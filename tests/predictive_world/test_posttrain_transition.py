import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from psi.models.predictive_world.latent_transition import TemporalLatentTransition
from psi.models.predictive_world.posttrain_transition import (
    add_output_coordinate_tensors, install_calibrated_coordinates, calibrate_posttrain,
    make_posttrain_transition, verify_posttrain_transition,
)


def affine_copy(source):
    target = TemporalLatentTransition(
        {"toy": {"dim": 2, "mean": [0., 0.], "std": [1., 1.]}},
        context_dim=16, latent_shape=source.latent_shape, hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32, output_affine=True,
    )
    values = add_output_coordinate_tensors(
        {"latent_transition." + k: v for k, v in source.state_dict().items()}, target)
    target.load_state_dict({k.removeprefix("latent_transition."): v for k, v in values.items()}, strict=True)
    return target.eval()


def moments(latent):
    latent = latent.double()
    return torch.full_like(latent[0], len(latent)), latent.sum(0), latent.square().sum(0)


def test_calibration_preserves_raw_predictions_and_policy_actions_and_deploy_roundtrip(tmp_path):
    from test_policy import _policy, _observed
    from test_checkpoint import _metadata
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle, load_predictive_world_bundle

    torch.manual_seed(26)
    original = _policy()
    nn.init.normal_(original.latent_transition.output_projection[-1].weight, std=.1)
    for adapter in original.student.adapters.values():
        nn.init.normal_(adapter.output[-1].weight, std=.1)
    target = copy.deepcopy(original)
    target.latent_transition = affine_copy(original.latent_transition)
    context = torch.randn(2, 3, 16)
    with pytest.raises(RuntimeError, match="calibrated"):
        target.latent_transition(context, mode="prior")
    candidate = torch.randn(19, 3, 2, 4) * torch.linspace(.1, 3, 24).reshape(3, 2, 4) + 4
    install_calibrated_coordinates(target.latent_normalizer, target.latent_transition, *moments(candidate))
    for mode in ("prior", "world"):
        condition = None
        if mode == "world":
            from test_latent_transition import _condition
            condition = _condition(torch.randn(2, 4, 2))
        before = original.latent_normalizer.unwhiten(original.latent_transition(context, mode=mode, action_condition=condition))
        after = target.latent_normalizer.unwhiten(target.latent_transition(context, mode=mode, action_condition=condition))
        torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)
    noise = torch.randn(1, 4, 2)
    expected = original.sample(_observed(), action_noise=noise)
    torch.testing.assert_close(target.sample(_observed(), action_noise=noise), expected, atol=2e-6, rtol=2e-6)
    names = ("history_state_adapters", "history_encoder", "latent_transition", "latent_normalizer", "student")
    metadata = _metadata()
    save_predictive_world_bundle(tmp_path / "deploy", phase="deploy", completed_updates=1,
        modules={n: getattr(target, n) for n in names}, metadata=metadata)
    restored = copy.deepcopy(target)
    restored.latent_transition.output_scale.zero_()
    restored.latent_transition.output_calibrated.fill_(False)
    load_predictive_world_bundle(tmp_path / "deploy", phase="deploy",
        modules={n: getattr(restored, n) for n in names}, expected_metadata=metadata)
    torch.testing.assert_close(restored.sample(_observed(), action_noise=noise), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("damage", ["empty", "nan", "shape", "repeat"])
def test_bad_calibration_cannot_partially_change_coordinates(damage):
    from test_policy import _policy
    policy = _policy()
    transition = affine_copy(policy.latent_transition)
    normalizer = policy.latent_normalizer
    stats = list(moments(torch.randn(9, 3, 2, 4)))
    if damage == "empty": stats[0][0] = 0
    elif damage == "nan": stats[1][0] = float("nan")
    elif damage == "shape": stats[2] = stats[2][:1]
    else: install_calibrated_coordinates(normalizer, transition, *stats)
    old = copy.deepcopy(normalizer.state_dict())
    old_transition = copy.deepcopy(transition.state_dict())
    with pytest.raises(ValueError):
        install_calibrated_coordinates(normalizer, transition, *stats)
    for key, value in normalizer.state_dict().items(): assert torch.equal(value, old[key])
    for key, value in transition.state_dict().items(): assert torch.equal(value, old_transition[key])


@pytest.mark.parametrize("mask_first_body_query", [False, True])
def test_real_encoder_calibration_changes_only_statistics_and_output_coordinates(mask_first_body_query):
    from test_phases import _components, _batch
    from test_policy import _ObservationEncoder
    from psi.models.predictive_world.representation import EMAFutureTrajectoryEncoder
    from psi.models.predictive_world.phases import _encoder_arguments

    online, _, _, _, _, flow, normalizer = _components()
    encoder = EMAFutureTrajectoryEncoder(online)
    encoder.normalizer_frozen.fill_(True)
    flow = affine_copy(flow)
    model = nn.Module()
    model.observation_encoder = _ObservationEncoder()
    model.encoder = encoder
    model.normalizer = normalizer
    model.flow = flow
    model.train()
    batches = [_batch(), _batch()]
    if mask_first_body_query:
        for batch in batches:
            batch["future_measurement_valid"][:, 0, 0] = False
            batch["future_measurements"][:, 0, 0] = float("nan")
            batch["future_measurement_times"] = batch["future_measurement_times"].clone()
            batch["future_measurement_times"][:, 0, 0] = 0
    batches[1]["group_valid"][0, 2] = False
    state = {k: v.clone() for k, v in encoder.state_dict().items()}
    parameters = {k: v.clone() for k, v in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    built = SimpleNamespace(model=model, reference_modules={"encoder": encoder, "normalizer": normalizer},
                            forecast_modules={"latent_transition": flow})
    report = calibrate_posttrain(built, batches, batch_count=2, device=torch.device("cpu"), seed=7)
    assert torch.equal(rng, torch.random.get_rng_state()) and model.training
    assert report["optimizer_updates"] == 0
    assert report["samples_per_coordinate_min"] == 3
    assert report["samples_per_coordinate_max"] == 4
    for key, value in encoder.state_dict().items(): assert torch.equal(value, state[key])
    for key, value in model.named_parameters(): assert torch.equal(value, parameters[key]) and value.grad is None
    values, masks = [], []
    with torch.no_grad():
        for batch in batches:
            prepared = dict(batch)
            prepared["anchor_patches"] = model.observation_encoder(batch["observed"]).current_visual
            values.append(encoder(**_encoder_arguments(prepared)))
            masks.append(batch["group_valid"])
    from psi.models.predictive_world.latent_transition import LatentNormalizer
    expected = LatentNormalizer(normalizer.latent_shape)
    expected.fit(torch.cat(values), torch.cat(masks))
    torch.testing.assert_close(normalizer.mean, expected.mean)
    torch.testing.assert_close(normalizer.std, expected.std)


def test_posttrain_record_checks_bindings_and_preserves_orthogonal_contracts():
    from psi.data.predictive_world.cache import canonical_sha256
    payload = {"data": {}, "model": {"latent_transition": {}}}
    source = dict(phase="joint", weights_sha256="a" * 64, metadata=dict(
        resolved_config_sha256=canonical_sha256(payload), normalization_id="b" * 64))
    record = make_posttrain_transition(source, payload, current_visual_pooling="content_area_v1", calibration_batches=16)
    expected = dict(architecture=dict(latent_transition=dict(hidden_dim=16, output_affine=True)),
                    temporal_abi=dict(action_horizon=50, current_visual_pooling="content_area_v1"))
    old = verify_posttrain_transition(record, source, payload, expected)
    assert old["temporal_abi"] == {"action_horizon": 50}
    assert old["architecture"] == {"latent_transition": {"hidden_dim": 16}}
    for key in ("source_weights_sha256", "source_normalization_id", "calibration_split"):
        broken = {**record, key: "bad"}
        with pytest.raises(ValueError): verify_posttrain_transition(broken, source, payload, expected)


def _distributed_calibration_worker(rank, rendezvous, output):
    import json
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from test_phases import _components, _batch
    from test_policy import _ObservationEncoder
    from psi.models.predictive_world.representation import EMAFutureTrajectoryEncoder

    torch.set_num_threads(1)
    torch.manual_seed(42)
    online, _, _, _, _, flow, normalizer = _components()
    encoder = EMAFutureTrajectoryEncoder(online)
    flow = affine_copy(flow)
    model = nn.Module()
    model.observation_encoder = _ObservationEncoder()
    model.encoder = encoder
    batch = _batch()
    batch["future_measurements"] += rank * 3
    built = SimpleNamespace(model=model, reference_modules={"encoder": encoder, "normalizer": normalizer},
                            forecast_modules={"latent_transition": flow})
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        report = calibrate_posttrain(built, [batch], batch_count=1, device=torch.device("cpu"), seed=5,
                                    distributed=True)
        Path(output, f"rank{rank}.json").write_text(json.dumps(report, sort_keys=True))
    finally:
        dist.destroy_process_group()


def test_two_rank_calibration_publishes_identical_global_statistics(tmp_path):
    import json
    from ddp_test_utils import spawn_with_timeout

    spawn_with_timeout(_distributed_calibration_worker,
                       args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)))
    first = json.loads((tmp_path / "rank0.json").read_text())
    second = json.loads((tmp_path / "rank1.json").read_text())
    assert first == second
    assert first["samples_per_coordinate_min"] == first["samples_per_coordinate_max"] == 4
    assert first["world_size"] == 2
