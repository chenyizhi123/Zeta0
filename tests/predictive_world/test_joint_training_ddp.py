"""Two actual Gloo ranks train and recover the complete small Zeta graph."""
import copy
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def _joint_worker(rank, root, payload):
    from psi.config.predictive_world import predictive_world_config_from_payload, predictive_world_loss_contract
    from psi.data.predictive_world.cache import canonical_sha256
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle, load_predictive_world_bundle
    from psi.models.predictive_world.engine import EngineConfig, run_phase_training, save_engine_state, load_engine_state
    from psi.models.predictive_world.training import module_state_sha256
    from test_checkpoint import _joint_metadata
    from test_scratch_joint import build_small_scratch

    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        config, _ = predictive_world_config_from_payload(payload)
        torch.manual_seed(825)
        with pytest.MonkeyPatch.context() as patch:
            built, batch = build_small_scratch(patch, config)
            # Only the frozen VLM I/O is stubbed. E/EMA/D/T/H/F/G and every loss,
            # DDP reducer, global denominator and optimizer operation are real.
            batch["actions"] = replace(batch["actions"], values=batch["actions"].values + rank * .3)
            batch["future_measurements"] = batch["future_measurements"] + rank * .2
            batch["future_patches"] = batch["future_patches"] + rank * .1
            if rank:
                batch["group_valid"][0, -1] = False
            initial = module_state_sha256(built.model)
            wrapped = DistributedDataParallel(built.model, find_unused_parameters=True)
            optimizer = torch.optim.AdamW((p for p in built.model.parameters() if p.requires_grad), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
            logged = []
            def prepare(update, micros):
                for micro in micros:
                    micro.update(_optimizer_update=update, forecast_mode="prior" if update % 2 == 0 else "world")
            completed = run_phase_training(model=wrapped,
                batches=iter([copy.deepcopy(batch) for _ in range(4)]),
                optimizer=optimizer, scheduler=scheduler,
                config=EngineConfig(phase="joint", max_updates=2, accumulation_steps=2,
                                    max_grad_norm=1, log_every=1, checkpoint_every=2),
                validity_fn=lambda micro: (micro["group_valid"].any(dim=1), None),
                forward_fn=lambda model, micro: model(micro), device=torch.device("cpu"),
                prepare_update=prepare, on_update=lambda update, metrics: logged.append((update, metrics)))
            assert completed == 2
            hashes = [None, None]
            final = module_state_sha256(built.model)
            dist.all_gather_object(hashes, final)
            assert final != initial and hashes[0] == hashes[1]
            assert int(built.reference_modules["encoder"].last_target_update) == 1
            assert not bool(built.reference_modules["normalizer"].fitted)
            if rank == 0:
                assert [update for update, _ in logged] == [1, 2]
                assert all(metrics["global_valid_weight"] == 8 for _, metrics in logged)
            metadata = {**_joint_metadata(), "initialization": built.initialization,
                "optimization_contract": asdict(config.training), "resolved_config_sha256": canonical_sha256(payload),
                "loss_contract": predictive_world_loss_contract(config.training)}
            metadata.pop("parent_world_model")
            checkpoint = root / "checkpoint"
            if rank == 0:
                save_predictive_world_bundle(checkpoint, phase="joint", modules=built.checkpoint_modules,
                    completed_updates=completed, metadata=metadata)
            dist.barrier()
            save_engine_state(checkpoint, optimizer=optimizer, scheduler=scheduler,
                completed_updates=completed, next_occurrence=16, config_identity=canonical_sha256(payload))
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())
            expected_random = torch.rand(4)
            with torch.no_grad():
                for module in built.checkpoint_modules.values():
                    for parameter in module.parameters():
                        parameter.zero_()
                built.reference_modules["encoder"].last_target_update.fill_(-1)
            optimizer.state.clear()
            torch.manual_seed(rank + 999)
            load_predictive_world_bundle(checkpoint, phase="joint", modules=built.checkpoint_modules,
                expected_metadata={"loss_contract": metadata["loss_contract"]})
            progress = load_engine_state(checkpoint, optimizer=optimizer, scheduler=scheduler,
                config_identity=canonical_sha256(payload), device=torch.device("cpu"))
            assert progress == (2, 16)
            assert module_state_sha256(built.model) == final
            torch.testing.assert_close(optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
            assert scheduler.state_dict() == expected_scheduler
            torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
            (root / f"rank{rank}.json").write_text(json.dumps(dict(completed_updates=completed,
                parameters_equal=True, raw_statistics_unused=True, checkpoint_recovered=True)))
    finally:
        dist.destroy_process_group()


def test_full_joint_ddp_trains_both_modes_with_accumulation_and_recovers_all_rank_state(tmp_path):
    from test_scratch_joint import scratch_config
    config = scratch_config(tmp_path)
    training = replace(config.training, objective_contract_version=6, latent_prediction_space="raw",
        preserve_control_sampling=True, normalizer_calibration_updates=0, forecast_warmup_updates=0,
        forecast_ramp_updates=0, effective_global_batch=8, gradient_accumulation_steps=2)
    config = replace(config, training=training, runtime=replace(config.runtime, gpus_per_node=2))
    config.validate()
    payload = json.loads(json.dumps(asdict(config)))
    mp.spawn(_joint_worker, args=(str(tmp_path), payload), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text())["checkpoint_recovered"]
