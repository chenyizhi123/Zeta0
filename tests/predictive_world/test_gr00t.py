"""Native N1.7 integration boundaries, using real (small) upstream modules."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from psi.models.predictive_world.gr00t import Gr00tN1d7ActionExpert, native_types
from psi.models.predictive_world.action_expert import clone_independent_expert, make_committed_action_prefix
from psi.models.predictive_world.phases import _expert_arguments, _response_velocity


def setup():
    from test_phases import _batch

    pytest.importorskip("diffusers", reason="native head checks require the gr00t dependency group")
    if not (Path(__file__).resolve().parents[2] / "third_party/Isaac-GR00T/gr00t").is_dir():
        pytest.skip("native head checks require the pinned Isaac-GR00T checkout")
    torch.manual_seed(292285)
    config_type, head_type = native_types()
    config = config_type(
        backbone_embedding_dim=6, input_embedding_dim=16, hidden_size=16,
        max_state_dim=4, max_action_dim=4, max_num_embodiments=1,
        action_horizon=8, state_history_length=1, state_dropout_prob=0.2,
        strict_vl_padding_mask=True, strict_action_padding_mask=True,
        diffusion_model_cfg={"num_attention_heads": 2, "attention_head_dim": 8,
            "num_layers": 4, "output_dim": 16, "dropout": 0.2,
            "final_dropout": True, "norm_type": "ada_norm",
            "interleave_self_attention": True, "positional_embeddings": None},
        vl_self_attention_cfg={"num_attention_heads": 2, "attention_head_dim": 3,
            "num_layers": 1, "dropout": 0.2, "final_dropout": True,
            "positional_embeddings": None},
    )
    expert = Gr00tN1d7ActionExpert(head_type(config),
        {"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}}, {"state": 3},
        state_statistics={"state": (torch.zeros(3), torch.ones(3))}, latent_shape=(3, 8, 4))
    batch = _batch(action_steps=8)
    observed = batch["observed"]
    batch["observed"] = replace(observed, raw_states=observed.raw_states[:, -1:],
        state_valid=observed.state_valid[:, -1:], state_times=observed.state_times[:, -1:])
    batch["features"] = replace(batch["features"],
        image_mask=torch.tensor([[False, True, True, False, False]] * 2))
    return expert, batch


def arguments(expert, batch, z=None):
    return dict(_expert_arguments(batch["observed"], batch["features"],
        torch.randn(2, 3, 8, 4) if z is None else z, batch["group_valid"]),
        normalized_action_samples=batch["action_noise"], action_valid=batch["actions"].valid,
        sigma_a=torch.tensor([.5, .75]))


def test_disabled_future_matches_native_solver_and_flow_direction(monkeypatch):
    from transformers.feature_extraction_utils import BatchFeature

    expert, batch = setup()
    expert.eval()
    obs, features = batch["observed"], batch["features"]
    noise = batch["action_noise"]
    native_noise = torch.nn.functional.pad(noise, (0, 2))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: native_noise.clone())
    native = expert.head.get_action(BatchFeature(data={
        "backbone_features": features.qwen_context,
        "backbone_attention_mask": features.context_valid, "image_mask": features.image_mask}),
        BatchFeature(data={"state": torch.nn.functional.pad(obs.raw_states, (0, 1)),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
            "action_mask": torch.nn.functional.pad(batch["actions"].valid, (0, 2))}))["action_pred"][..., :2]
    kwargs = _expert_arguments(obs, features, torch.zeros(2, 3, 8, 4), torch.zeros(2, 3, dtype=torch.bool))
    result = expert.sample(**kwargs, native_dims=obs.action_native_dims, noise=noise, steps=4)
    expected = expert.denormalize_actions(native, obs.action_protocol_ids, obs.action_native_dims)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)


def test_future_reaches_image_and_text_blocks_and_ignores_padding():
    expert, batch = setup()
    expert.eval()
    batch["group_valid"][:, 1] = False
    z = torch.randn(2, 3, 8, 4, requires_grad=True)
    args = arguments(expert, batch, z)
    calls = []
    handles = [block.register_forward_pre_hook(
        lambda module, a, kw: calls.append(kw["encoder_attention_mask"]), with_kwargs=True)
        for block in expert.head.model.transformer_blocks[::2]]
    output = expert(**args)
    for handle in handles:
        handle.remove()
    assert len(calls) == 2
    # Text sees second view; image sees first. Both contain exactly 16 valid futures.
    assert calls[0][:, 5:29].sum() == 0 and calls[0][:, 29:].sum() == 32
    assert calls[1][:, 5:29].sum() == 32 and calls[1][:, 29:].sum() == 0
    output.square().sum().backward()
    assert z.grad[:, [0, 2]].abs().sum() > 0
    assert z.grad[:, 1].count_nonzero() == 0
    corrupt = z.detach().clone()
    corrupt[:, 1] = float("nan")
    torch.testing.assert_close(expert(**{**args, "z_raw": corrupt}), output.detach())


def test_response_is_deterministic_detached_and_restores_training_modes():
    expert, batch = setup()
    expert.train()
    z = torch.randn(2, 3, 8, 4, requires_grad=True)
    args = arguments(expert, batch, z)
    state = {name: tensor.detach() for name, tensor in
             list(expert.named_parameters()) + list(expert.named_buffers())}
    rng = torch.get_rng_state().clone()
    first = _response_velocity(expert, state, args)
    second = _response_velocity(expert, state, args)
    assert torch.equal(first, second) and torch.equal(torch.get_rng_state(), rng)
    first.square().sum().backward()
    assert z.grad.abs().sum() > 0
    assert all(p.grad is None for p in expert.parameters())
    assert expert.training and expert.head.training


def test_rtc_prefix_preserved_and_suffix_conditioned():
    expert, batch = setup()
    expert.eval()
    obs = batch["observed"]
    kwargs = _expert_arguments(obs, batch["features"], torch.zeros(2, 3, 8, 4), batch["group_valid"])
    prefix = make_committed_action_prefix(batch["actions"], torch.tensor([2, 2]))
    out = expert.sample(**kwargs, native_dims=obs.action_native_dims, noise=batch["action_noise"],
                        committed_prefix=prefix, steps=4)
    torch.testing.assert_close(out[:, :2], prefix.values[:, :2], rtol=0, atol=5e-7)
    changed = replace(prefix, values=prefix.values + prefix.valid * 2)
    other = expert.sample(**kwargs, native_dims=obs.action_native_dims, noise=batch["action_noise"],
                          committed_prefix=changed, steps=4)
    assert not torch.equal(other[:, 2:], out[:, 2:])
    args = arguments(expert, batch)
    sample = args["normalized_action_samples"].detach().requires_grad_()
    suffix = expert(**{**args, "normalized_action_samples": sample})[:, 2:]
    prefix_gradient = torch.autograd.grad(suffix.sum(), sample)[0][:, :2]
    assert torch.isfinite(prefix_gradient).all() and prefix_gradient.abs().sum() > 0
    with pytest.raises(ValueError, match="common suffix"):
        expert(**{**arguments(expert, batch), "sigma_a": torch.linspace(.1, .9, 8).expand(2, -1)})


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_joint_objective_updates_native_heads_and_response_only_updates_forecast(tmp_path, activation_checkpointing):
    from test_phases import _components
    from psi.models.predictive_world.phases import (
        ForecastObjective, JointObjective, PolicyObjective, RepresentationObjective, WorldModelObjective)

    teacher, batch = setup()
    teacher.activation_checkpointing = activation_checkpointing
    encoder, readout, _, adapters, history, transition, normalizer = _components(action_steps=8)
    history.state_history_count = 1
    history.state_history_position = torch.nn.Parameter(history.state_history_position[-1:].detach().clone())
    student = clone_independent_expert(teacher)
    with torch.no_grad():
        torch.nn.init.normal_(transition.output_projection[-1].weight, std=.05)
    common = dict(reference_encoder=encoder, normalizer=normalizer, history_state_adapters=adapters,
        history_encoder=history, latent_transition=transition, control_teacher=teacher,
        control_reader="both", control_weight=1., freeze_reference=False, latent_prediction_space="raw")
    response = ForecastObjective(**common, prior_weight=0., world_weight=0.)
    result = response(batch, control_student=student)
    result.per_sample.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in transition.parameters())
    assert all(p.grad is None for p in (*teacher.parameters(), *student.parameters(), *encoder.parameters()))
    for module in (adapters, history, transition):
        module.zero_grad(set_to_none=True)
    world = WorldModelObjective(RepresentationObjective(encoder, readout, teacher),
        ForecastObjective(**common), normalizer, normalizer_calibration_updates=0,
        forecast_warmup_updates=0, forecast_ramp_updates=0)
    policy = PolicyObjective(student=student, normalizer=normalizer, history_state_adapters=adapters,
        history_encoder=history, latent_transition=transition, freeze_world=False, latent_prediction_space="raw")
    joint = JointObjective(world, policy)
    result = joint(batch)
    assert torch.isfinite(result.per_sample).all()
    result.per_sample.mean().backward()
    for module in (teacher, student, encoder, transition):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    # Exercise the same immutable bundle and optimizer/RNG resume boundaries as
    # the trainer, with actual native modules and both response readers.
    import copy
    from dataclasses import asdict
    from test_checkpoint import _metadata
    from psi.config.predictive_world import PredictiveWorldTrainingConfig
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle, load_predictive_world_bundle
    from psi.models.predictive_world.engine import save_engine_state, load_engine_state
    from psi.models.predictive_world.gr00t_assets import EXPORT_FORMAT, initialization_contract

    modules = dict(encoder=encoder, readout=readout, teacher=teacher, student=student,
        history_state_adapters=adapters, history_encoder=history, latent_transition=transition, normalizer=normalizer)
    def optimizer_for(parts):
        return torch.optim.AdamW([p for module in parts.values() for p in module.parameters() if p.requires_grad], lr=1e-3)
    optimizer = optimizer_for(modules)
    before = teacher.future_projection[1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, teacher.future_projection[1].weight)
    training = PredictiveWorldTrainingConfig(phase="joint", joint_from_gr00t=True,
        objective_contract_version=5, forecast_control_reader="both", latent_prediction_space="raw",
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0,
        max_updates=10, micro_batch=2, effective_global_batch=2)
    source = dict(format=EXPORT_FORMAT, source_config_sha256="a" * 64,
        source_weights_sha256={"base.safetensors": "b" * 64},
        exported_weights_sha256={"backbone.safetensors": "c" * 64})
    metadata = {**_metadata(), "initialization": initialization_contract(training, source),
        "optimization_contract": asdict(training),
        "architecture": {"host_architecture": "gr00t_n1d7", "gr00t_config_sha256": "a" * 64}}
    checkpoint = tmp_path / "checkpoint"
    save_predictive_world_bundle(checkpoint, phase="joint", modules=modules, completed_updates=1, metadata=metadata)
    save_engine_state(checkpoint, optimizer=optimizer, scheduler=None, completed_updates=1,
        next_occurrence=2, config_identity="native-test")
    restored_joint, restored = copy.deepcopy((joint, modules))
    with torch.no_grad():
        restored["teacher"].future_projection[1].weight.add_(10)
    restored_optimizer = optimizer_for(restored)
    load_predictive_world_bundle(checkpoint, phase="joint", modules=restored,
        expected_metadata={"architecture": metadata["architecture"]})
    assert load_engine_state(checkpoint, optimizer=restored_optimizer, scheduler=None,
        config_identity="native-test", device=torch.device("cpu")) == (1, 2)
    rng = torch.get_rng_state()
    optimizer.zero_grad(set_to_none=True)
    expected = joint(batch).per_sample.mean()
    expected.backward()
    optimizer.step()
    torch.set_rng_state(rng)
    restored_optimizer.zero_grad(set_to_none=True)
    actual = restored_joint(batch).per_sample.mean()
    actual.backward()
    restored_optimizer.step()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(restored[name].state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="metadata"):
        load_predictive_world_bundle(checkpoint, phase="joint", modules=restored,
            expected_metadata={"architecture": {"host_architecture": "zeta_native"}})


@pytest.mark.parametrize("response_only", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpointed_native_outputs_gradients_dropout_rng_and_state_keys_match(response_only, device):
    import copy
    from psi.models.predictive_world.engine import move_to_device
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA RNG/recomputation equivalence needs a GPU")
    ordinary, batch = setup()
    ordinary.to(device)
    batch = move_to_device(batch, torch.device(device))
    optimized = copy.deepcopy(ordinary)
    optimized.activation_checkpointing = True
    assert ordinary.state_dict().keys() == optimized.state_dict().keys()
    z = torch.randn(2, 3, 8, 4, device=device)
    results = []
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    for expert in (ordinary, optimized):
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        latent = z.clone().requires_grad_()
        args = move_to_device(arguments(expert, batch, latent), torch.device(device))
        state = {name: tensor.detach() for name, tensor in
                 list(expert.named_parameters()) + list(expert.named_buffers())}
        # Both call types coexist before backward; response recomputation must
        # not accidentally restore trainable reader weights or enable dropout.
        response = _response_velocity(expert, state, args)
        output = response if response_only else response + expert(**args)
        output.square().mean().backward()
        assert expert.training and all(module.training for module in expert.modules())
        if response_only:
            assert all(parameter.grad is None for parameter in expert.parameters())
        states = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()) if cuda_rng is not None else (torch.get_rng_state().clone(),)
        results.append((output.detach(), latent.grad, states,
                        {k: p.grad for k, p in expert.named_parameters()}))
    for a, b in zip(results[0][:3], results[1][:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, a in results[0][3].items():
        b = results[1][3][key]
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
