"""Standalone (no private weights) real-robot preparation tests for scratch and GR00T N1.7."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_module():
    path = ROOT / "scripts/train/predictive_world/prepare_sonic_task.py"
    spec = importlib.util.spec_from_file_location("prepare_sonic_task_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_vlm_stubs(monkeypatch, tmp_path):
    import psi.data.lerobot.action_aligned_cache as aligned
    import psi.data.predictive_world.provenance as provenance
    import psi.data.predictor_pretrain.runtime as runtime
    import psi.models.action_aligned_vlm_features as vlm_features
    import psi.models.predictive_world.gr00t_assets as gr00t_assets

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(vlm_features, "resolve_qwen3vl_visual_feature_spec", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(provenance, "load_qwen_processor", lambda path: SimpleNamespace(chat_template="tpl"))
    monkeypatch.setattr(aligned, "build_action_aligned_vlm_processor_fingerprint", lambda p: {"stub": True})
    monkeypatch.setattr(runtime, "fingerprint_model_directory", lambda path: "stub-vlm")
    monkeypatch.setattr(gr00t_assets, "verify_backbone_export", lambda *a, **k: {"verified": True})

    qwen_dir = tmp_path / "qwen_backbone"
    qwen_dir.mkdir(exist_ok=True)
    (qwen_dir / "config.json").write_text("{}")
    gr00t_dir = tmp_path / "gr00t_base"
    gr00t_dir.mkdir(exist_ok=True)
    (gr00t_dir / "config.json").write_text('{"model_type": "Gr00tN1d7"}')
    return qwen_dir, gr00t_dir


@pytest.mark.parametrize("init_mode,expected_horizon", [("scratch", 50), ("gr00t_n1d7", 40)])
def test_prepare_sonic_lerobot_standalone_modes(tmp_path, monkeypatch, init_mode, expected_horizon):
    from test_sonic_lerobot_post_training import recording

    dataset = tmp_path / "place_chair"
    recording(dataset)
    qwen_dir, gr00t_dir = _install_vlm_stubs(monkeypatch, tmp_path)
    out = tmp_path / f"prep_{init_mode}"
    argv = [
        "prepare_sonic_lerobot_task.py",
        "--init-mode", init_mode,
        "--dataset", str(dataset),
        "--task", "real/place_chair",
        "--output", str(out),
        "--qwen-model", str(qwen_dir),
        "--current-state-only",
        "--fixed-action-grid",
        "--whole-body-target",
        "--gpus", "1",
        "--batch", "4",
        "--updates", "20",
        "--checkpoint-every", "10",
        "--first-checkpoint-update", "10",
    ]
    if init_mode == "gr00t_n1d7":
        argv.extend(["--gr00t-model", str(gr00t_dir)])
    monkeypatch.setattr(sys, "argv", argv)
    mod = _load_prepare_module()
    mod.main(dataset_format="sonic_lerobot")

    config = json.loads((out / "config.json").read_text())
    assert config["data"]["action_hz"] == 30.0
    assert config["data"]["action_horizon"] == expected_horizon
    assert config["data"]["state_history_count"] == 1
    assert config["data"]["current_visual_pooling"] == "content_area_v1"
    assert config["training"]["objective_contract_version"] == 6
    assert config["training"]["latent_prediction_space"] == "raw"
    assert config["training"].get("init_joint") is None
    assert config["training"]["joint_from_scratch"] is (init_mode == "scratch")
    assert config["training"]["joint_from_gr00t"] is (init_mode == "gr00t_n1d7")
    assert config["model"]["host_architecture"] == ("gr00t_n1d7" if init_mode == "gr00t_n1d7" else "zeta_native")
    assert (out / "artifacts/source_audit.json").is_file()
    measurements = json.loads((out / "artifacts/measurement_registry.json").read_text())
    assert list(measurements) == ["real_g1_measured_whole_body_q29_brainco12_v2"]
    if init_mode == "gr00t_n1d7":
        prov = json.loads((out / "artifacts/feature_provenance.json").read_text())
        assert prov["gr00t_source"] == {"verified": True}


@pytest.mark.parametrize("init_mode,expected_horizon", [("scratch", 30), ("gr00t_n1d7", 40)])
def test_prepare_real_sonic_50hz_standalone_modes(tmp_path, monkeypatch, init_mode, expected_horizon):
    from test_real_sonic_post_training import recording

    dataset = tmp_path / "place_doll"
    recording(dataset, fps=50)
    qwen_dir, gr00t_dir = _install_vlm_stubs(monkeypatch, tmp_path)
    out = tmp_path / f"prep_doll_{init_mode}"
    argv = [
        "prepare_real_sonic_task.py",
        "--init-mode", init_mode,
        "--dataset", str(dataset),
        "--task", "real/place_doll",
        "--output", str(out),
        "--qwen-model", str(qwen_dir),
        "--current-state-only",
        "--fixed-action-grid",
        "--independent-future-components",
        "--gpus", "1",
        "--batch", "4",
        "--updates", "20",
        "--checkpoint-every", "10",
        "--first-checkpoint-update", "10",
    ]
    if init_mode == "gr00t_n1d7":
        argv.extend(["--gr00t-model", str(gr00t_dir)])
    monkeypatch.setattr(sys, "argv", argv)
    mod = _load_prepare_module()
    mod.main(dataset_format="real_sonic")

    config = json.loads((out / "config.json").read_text())
    assert config["data"]["action_hz"] == 50.0
    assert config["data"]["action_horizon"] == expected_horizon
    assert config["data"]["state_history_count"] == 1
    assert config["training"]["objective_contract_version"] == 6
    assert config["training"]["joint_from_gr00t"] is (init_mode == "gr00t_n1d7")
    assert (out / "artifacts/source_audit.json").is_file()
