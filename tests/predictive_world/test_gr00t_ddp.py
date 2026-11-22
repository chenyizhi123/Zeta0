"""Real two-rank native-reader updates with shared DDP gradient storage."""
import copy
from datetime import timedelta

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from psi.models.predictive_world.checkpoint_diagnostic import state_tree_sha256
from psi.models.predictive_world.phases import _response_velocity


class NativeReaderObjective(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert
        self.predictor = nn.Linear(4, 4)
        self.world_branch = nn.Linear(4, 4)

    def forward(self, kwargs, world):
        future = self.predictor(kwargs['z_raw'])
        if world:
            future = self.world_branch(future)
        arguments = {**kwargs, 'z_raw': future}
        ordinary = self.expert(**arguments)
        detached = {name: value.detach() for name, value in
                    list(self.expert.named_parameters()) + list(self.expert.named_buffers())}
        response = _response_velocity(self.expert, detached, arguments)
        return ordinary.square().mean() + response.square().mean()


def _bucket_view_worker(rank, rendezvous):
    from test_gr00t import setup, arguments

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        expert, batch = setup()
        expert.activation_checkpointing = True
        baseline = NativeReaderObjective(expert)
        optimized = copy.deepcopy(baseline)
        models = [DistributedDataParallel(model, find_unused_parameters=True,
                    gradient_as_bucket_view=views) for model, views in ((baseline, False), (optimized, True))]
        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in models]
        # Distinct inputs on each rank exercise actual gradient averaging.
        torch.manual_seed(37 + rank)
        for world in (False, True, False, True):
            kwargs = arguments(expert, batch)
            rng = torch.get_rng_state()
            results = []
            for model, optimizer in zip(models, optimizers):
                torch.set_rng_state(rng)
                optimizer.zero_grad(set_to_none=True)
                loss = model(kwargs, world)
                loss.backward()
                results.append((loss.detach().clone(), torch.get_rng_state().clone(),
                    {name: None if value.grad is None else value.grad.detach().clone()
                     for name, value in model.named_parameters()}))
                optimizer.step()
            assert torch.equal(results[0][0], results[1][0])
            assert torch.equal(results[0][1], results[1][1])
            for name, expected in results[0][2].items():
                actual = results[1][2][name]
                assert (expected is None) == (actual is None)
                if expected is not None:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert state_tree_sha256(baseline.state_dict()) == state_tree_sha256(optimized.state_dict())
            assert state_tree_sha256(optimizers[0].state_dict()) == state_tree_sha256(optimizers[1].state_dict())
        assert all(parameter.grad._base is not None for parameter in optimized.parameters()
                   if parameter.grad is not None)
    finally:
        dist.destroy_process_group()


def test_bucket_views_preserve_native_checkpointed_ddp_updates(tmp_path):
    from test_gr00t import setup

    setup()  # Apply the established dependency/source-pin skips in the parent.
    torch.multiprocessing.spawn(_bucket_view_worker,
        args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
