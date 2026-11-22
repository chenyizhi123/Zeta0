import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from psi.models.predictive_world.gr00t_assets import export_backbone, verify_backbone_export


def test_export_is_exact_truncated_offline_and_source_bound(tmp_path):
    base, processor, output = (tmp_path / name for name in ("base", "processor", "output"))
    base.mkdir()
    processor.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", "select_layer": 2,
                                                "backbone_embedding_dim": 6}))
    (processor / "config.json").write_text(json.dumps({"model_type": "qwen3_vl",
        "text_config": {"num_hidden_layers": 4, "hidden_size": 6}}))
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (processor / name).write_text('{}')
    tensors = {"backbone.model.model.language_model.layers.0.weight": torch.randn(6, 6),
        "backbone.model.model.language_model.layers.1.weight": torch.randn(6, 6),
        "backbone.model.model.visual.weight": torch.randn(6, 6),
        "action_head.model.weight": torch.randn(6, 6)}
    save_file(tensors, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    manifest = export_backbone(base, processor, output, shard_bytes=150)
    assert json.loads((output / "config.json").read_text())["text_config"]["num_hidden_layers"] == 2
    actual = {k: v for path in output.glob('*.safetensors') for k, v in load_file(path).items()}
    expected = {k.removeprefix('backbone.model.'): v for k, v in tensors.items() if k.startswith('backbone.model.')}
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert verify_backbone_export(output, base, expected_config_sha256=manifest["source_config_sha256"]) == manifest
    with pytest.raises(FileExistsError):
        export_backbone(base, processor, output)
    tensors['action_head.model.weight'].add_(1)
    save_file(tensors, base / "model.safetensors")
    with pytest.raises(ValueError, match="base weight identity"):
        verify_backbone_export(output, base, expected_config_sha256=manifest["source_config_sha256"])


def test_load_native_head_preserves_shared_weights_and_initializes_new_embodiment(tmp_path):
    from test_gr00t import setup
    from psi.models.predictive_world.gr00t import load_action_expert, PROJECTOR_PREFIXES

    expert, _ = setup()
    expert.head.config.save_pretrained(tmp_path)
    state = {"action_head." + key: value for key, value in expert.head.state_dict().items()}
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in state}}))
    loaded = load_action_expert(tmp_path, {"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}},
        {"state": 3}, {"state": (torch.zeros(3), torch.ones(3))}, latent_shape=(3, 8, 4))
    for key, value in loaded.head.state_dict().items():
        if not key.startswith(PROJECTOR_PREFIXES):
            torch.testing.assert_close(value, state["action_head." + key], rtol=0, atol=0)
    assert not torch.equal(loaded.head.action_encoder.W1.W, expert.head.action_encoder.W1.W)


def test_phase_factory_builds_independent_native_readers_with_explicit_initialization(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_gr00t import setup
    from test_scratch_joint import scratch_config
    from psi.data.predictive_world.cache import file_sha256
    from psi.models.predictive_world import factory, gr00t_assets
    import psi.data.predictive_world.provenance as provenance

    native, _ = setup()
    base = tmp_path / "base"
    base.mkdir()
    native.head.config.action_horizon = 40
    native.head.config.save_pretrained(base)
    tensors = {"action_head." + key: value for key, value in native.head.state_dict().items()}
    save_file(tensors, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in tensors}}))
    source = dict(format=gr00t_assets.EXPORT_FORMAT, source_config_sha256=file_sha256(base / "config.json"),
        source_weights_sha256={"model.safetensors": file_sha256(base / "model.safetensors")},
        exported_weights_sha256={"backbone.safetensors": "b" * 64})
    config = scratch_config(tmp_path)
    config = replace(config, data=replace(config.data, action_horizon=40),
        runtime=replace(config.runtime, gr00t_activation_checkpointing=True),
        model=replace(config.model, host_architecture="gr00t_n1d7", gr00t_model=str(base),
            gr00t_config_sha256=source["source_config_sha256"], action_solver_steps=4),
        training=replace(config.training, joint_from_scratch=False, joint_from_gr00t=True,
            latent_prediction_space="raw", normalizer_calibration_updates=0,
            forecast_warmup_updates=0, forecast_ramp_updates=0))
    config.validate()
    frozen = torch.nn.Linear(1, 1)
    frozen.config = SimpleNamespace(hidden_size=6)

    class Observation(torch.nn.Module):
        def __init__(self, backbone, *, include_image_mask):
            super().__init__()
            assert include_image_mask
            self.backbone = backbone.requires_grad_(False)

    # Only frozen VLM I/O is replaced; the factory, native head loading, Zeta
    # modules, cloning and initialization contract execute unchanged.
    monkeypatch.setattr(gr00t_assets, "verify_backbone_export", lambda *a, **k: source)
    monkeypatch.setattr(gr00t_assets, "load_backbone", lambda *a, **k: frozen)
    monkeypatch.setattr(provenance, "verify_qwen_provenance", lambda *a, **k: {})
    monkeypatch.setattr(factory, "FrozenQwenObservationEncoder", Observation)
    built = factory.build_phase(config, device=torch.device("cpu"))
    assert built.initialization == gr00t_assets.initialization_contract(config.training, source)
    assert built.parent_world_model is None and not built.reference_modules["normalizer"].fitted
    teacher, student = (built.checkpoint_modules[key] for key in ("teacher", "student"))
    assert teacher.activation_checkpointing and student.activation_checkpointing
    for a, b in zip(teacher.parameters(), student.parameters()):
        assert a.data_ptr() != b.data_ptr()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(teacher.head.position_embedding.weight, native.head.position_embedding.weight)
