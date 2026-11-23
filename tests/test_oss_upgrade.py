"""Tests for Zeta0 v0.2 layered hosts, public SDK, CLI extensions, mock G1 & eval bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "psi.cli", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=40,
    )


def test_public_zeta0_sdk_is_lazy_and_matches_psi_version():
    code = f"""import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'transformers'):
            raise AssertionError('Heavy import on top-level zeta0: ' + fullname)
sys.meta_path.insert(0, Reject())
import psi, zeta0
assert zeta0.__version__ == psi.__version__
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

    import zeta0

    assert hasattr(zeta0.core, "PredictiveWorldPolicy")
    assert hasattr(zeta0.core, "JointObjective")
    assert hasattr(zeta0.core, "FutureTrajectoryEncoder")
    assert set(zeta0.hosts.list_hosts()) >= {"zeta_native", "gr00t_n1d7", "pi05"}


def test_layered_host_registry_contracts_and_pi05_placeholder():
    from zeta0.core import ObservationFeatures, FutureTrajectoryEncoder, WorldReadout
    from zeta0.hosts import describe_hosts, get_host

    hosts = describe_hosts()
    assert hosts["zeta_native"]["status"] == "supported"
    assert hosts["gr00t_n1d7"]["status"] == "supported"
    assert hosts["gr00t_n1d7"]["requires_image_mask"] is True
    assert hosts["pi05"]["status"] == "planned"
    assert hosts["pi05"]["supports_kv_cache"] is True

    # pi05 fails explicitly with actionable guidance rather than silently misbehaving.
    pi05 = get_host("pi05")
    with pytest.raises(NotImplementedError, match="pi05"):
        pi05.load_backbone(backbone_path=ROOT, device=torch.device("cpu"), dtype=torch.float32)

    # ObservationFeatures supports backward-compatible qwen_context + future kv_cache.
    ctx = torch.zeros(1, 4, 16)
    feat = ObservationFeatures(
        qwen_context=ctx,
        context_valid=torch.ones(1, 4, dtype=torch.bool),
        current_visual=ctx,
        kv_cache=("dummy_kv",),
    )
    assert feat.context is ctx
    assert feat.kv_cache == ("dummy_kv",)

    # Representation modules accept custom visual_dim (e.g., SigLIP 1152 for pi05).
    enc = FutureTrajectoryEncoder(visual_dim=64, measurement_dim=6, latent_shape=(3, 8, 16), hidden_dim=32, num_blocks=1, num_heads=4, ff_dim=64)
    ro = WorldReadout(visual_dim=64, measurement_dim=6, latent_shape=(3, 8, 16), max_scene_patches=8, body_points_per_group=2, hidden_dim=32, num_blocks=1, num_heads=4, ff_dim=64)
    assert enc.visual_projection[1].in_features == 64
    assert ro.scene_output[1].out_features == 64


def test_real_sonic_protocol_accepts_gr00t_action_grids_and_health_reports_host():
    from psi.deploy.real_sonic_protocol import validate_action_grid
    from psi.deploy.real_sonic_server import InferenceService

    assert validate_action_grid(50, 40) == (50.0, 40)
    assert validate_action_grid(30, 40) == (30.0, 40)
    manifest = {
        "architecture": {"host_architecture": "gr00t_n1d7"},
        "temporal_abi": {"action_hz": 50, "action_horizon": 40, "state_history_count": 1},
        "task": "Push the box",
        "source_completed_updates": 40000,
        "weights_sha256": "0" * 64,
        "rtc_max_prefix_steps": 6,
    }
    service = InferenceService(lambda req, seed: np.zeros((40, 76)), manifest, rtc_max_prefix_steps=6)
    health = service.health()
    assert health["host_architecture"] == "gr00t_n1d7"
    assert (health["action_hz"], health["action_horizon"]) == (50.0, 40)


def test_cli_hosts_init_and_data_pull_plan(tmp_path):
    res_hosts = _cli("--json", "hosts")
    assert res_hosts.returncode == 0, res_hosts.stderr
    assert "gr00t_n1d7" in json.loads(res_hosts.stdout)["hosts"]

    # Plan pull for official simulation and real datasets (offline, zero network calls).
    res_pull = _cli(
        "--json", "data", "pull", "humanoidarena",
        "--tasks", "HOI_pp_box", "--output", str(tmp_path / "sim"),
    )
    assert res_pull.returncode == 0, res_pull.stderr
    plan = json.loads(res_pull.stdout)
    assert plan["repo_id"] == "SII-yizhi/HumanoidArenaV3.1_SONIC40"
    assert plan["allow_patterns"] == ["README.md", "HOI_pp_box/**"]
    assert plan["executed"] is False

    res_real = _cli(
        "--json", "data", "pull", "real",
        "--tasks", "place_chair,flatten_box", "--output", str(tmp_path / "real"),
    )
    assert res_real.returncode == 0, res_real.stderr
    real_plan = json.loads(res_real.stdout)
    assert real_plan["repo_id"] == "SII-yizhi/zeta0-real-dataset"
    assert real_plan["allow_patterns"] == [
        "README.md", "SHA256SUMS", "real/place_chair.zip", "additional/flatten_box.zip",
    ]

    # Init GR00T config and verify automatic SHA-256 computation from config.json.
    gr00t_dir = tmp_path / "gr00t_base"
    gr00t_dir.mkdir()
    cfg_bytes = b'{"model_type": "Gr00tN1d7"}\n'
    (gr00t_dir / "config.json").write_bytes(cfg_bytes)
    out_cfg = tmp_path / "gr00t_run.json"
    res_init = _cli(
        "--root", str(ROOT), "--json", "init",
        "--host", "gr00t_n1d7",
        "--qwen-model", str(tmp_path / "qwen"),
        "--gr00t-model", str(gr00t_dir),
        "--prepared-dir", str(tmp_path / "prepared"),
        "--output-dir", str(tmp_path / "run_out"),
        "--max-updates", "1000",
        "--gpus", "2",
        "--micro-batch", "16",
        "--output", str(out_cfg),
    )
    assert res_init.returncode == 0, res_init.stderr
    init_report = json.loads(res_init.stdout)
    assert init_report["ready_for_dry_run"] is True
    generated = json.loads(out_cfg.read_text())
    assert generated["model"]["gr00t_config_sha256"] == hashlib.sha256(cfg_bytes).hexdigest()
    assert generated["training"]["effective_global_batch"] == 32


def test_mock_g1_client_logs_and_visualize_pipeline(tmp_path):
    trace = tmp_path / "mock_trace.jsonl"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/deploy/mock_g1_client.py"),
         "--in-process", "--cycles", "5", "--simulated-latency-ms", "1", "--output", str(trace)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["accepted"] == 5

    res_sum = _cli("--json", "logs", "summarize", str(trace))
    assert res_sum.returncode == 0, res_sum.stderr
    summary = json.loads(res_sum.stdout)
    assert summary["audit_counts_balanced"] is True
    assert summary["metrics"]["inference_seconds"]["samples"] == 5

    dashboard = tmp_path / "rtc_dashboard.png"
    res_viz = _cli("--json", "visualize", "log", str(trace), "--output", str(dashboard))
    assert res_viz.returncode == 0, res_viz.stderr
    assert dashboard.is_file() and dashboard.stat().st_size > 1000


def test_humanoidarena_policy_bridge_closed_loop(tmp_path):
    from benchmarks.humanoidarena.policy_bridge import (
        HUMANOIDARENA_ACTION_DIM,
        HUMANOIDARENA_STATE_DIM,
        HumanoidArenaPolicyBridge,
        write_eval_summary,
    )

    class _DummyPolicy:
        def sample(self, observed, *, action_noise):
            assert observed.raw_states.shape == (1, 1, HUMANOIDARENA_STATE_DIM)
            return torch.zeros_like(action_noise)

    class _DummyEnv:
        def __init__(self):
            self.t = 0

        def reset(self, seed=0):
            self.t = 0
            return {"state": np.zeros(HUMANOIDARENA_STATE_DIM, dtype=np.float32)}

        def step(self, action):
            assert action.shape == (HUMANOIDARENA_ACTION_DIM,)
            self.t += 1
            done = self.t >= 12
            return {"state": np.zeros(HUMANOIDARENA_STATE_DIM, dtype=np.float32)}, 0.0, done, {"success": done}

    bridge = HumanoidArenaPolicyBridge(
        policy=_DummyPolicy(),
        prepare_qwen_inputs=lambda obs, inst: {},
        action_horizon=40,
        execute_steps=6,
        host_architecture="gr00t_n1d7",
    )
    ep = bridge.run_episode(_DummyEnv(), task="HOI_pp_box", instruction="Push box", episode_index=0, seed=7, max_steps=20)
    assert ep.success is True and ep.steps == 12 and ep.chunks_inferred == 2
    out = tmp_path / "eval_summary.json"
    summary = write_eval_summary([ep], out)
    assert summary["mean_success_rate"] == 1.0 and out.is_file()


def test_inference_benchmark_runs_on_cpu():
    import os

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/benchmark_inference.py"),
         "--preset", "tiny-cpu", "--device", "cpu", "--warmup", "1", "--iterations", "2"],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["format"] == "zeta0.inference_benchmark.v1"
    assert report["latency_ms"]["mean"] > 0


def test_host_load_backbone_preserves_factory_observation_encoder_monkeypatch(monkeypatch, tmp_path):
    from types import ModuleType, SimpleNamespace
    from psi.models.predictive_world import factory, gr00t_assets
    from psi.models.predictive_world.hosts import get_host

    class _DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=6)

    class _Psi0Obs(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

    class _GrootObs(torch.nn.Module):
        def __init__(self, backbone, *, include_image_mask):
            super().__init__()
            assert include_image_mask is True
            self.backbone = backbone

    fake_transformers = ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=lambda *a, **k: _DummyBackbone()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(gr00t_assets, "load_backbone", lambda *a, **k: _DummyBackbone())

    psi0_res = get_host("zeta_native").load_backbone(
        backbone_path=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        observation_encoder_cls=_Psi0Obs,
    )
    assert isinstance(psi0_res.observation_encoder, _Psi0Obs)
    assert (psi0_res.context_dim, psi0_res.visual_dim) == (6, 2048)

    groot_res = get_host("gr00t_n1d7").load_backbone(
        backbone_path=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        observation_encoder_cls=_GrootObs,
    )
    assert isinstance(groot_res.observation_encoder, _GrootObs)
    assert (groot_res.context_dim, groot_res.visual_dim) == (6, 2048)
    assert hasattr(factory, "FrozenQwenObservationEncoder")


def test_finetune_launcher_dry_run_supports_scratch_and_gr00t_n1d7(tmp_path):
    import os

    dataset = tmp_path / "place_chair"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    for rel in ("qwen/config.json", "gr00t_base/config.json", "gr00t_backbone/config.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")

    env_scratch = {
        **os.environ,
        "INIT_MODE": "scratch",
        "DATASET_FORMAT": "sonic_lerobot",
        "DATASET": str(dataset),
        "OUTPUT_DIR": str(tmp_path / "run_scratch"),
        "QWEN_MODEL": str(tmp_path / "qwen"),
        "GPUS": "1",
        "BATCH_SIZE": "16",
        "UPDATES": "100",
        "PYTHON_BIN": sys.executable,
    }
    res = subprocess.run(
        ["bash", str(ROOT / "scripts/train/predictive_world/finetune-real-sonic.sh"), "--dry-run"],
        env=env_scratch,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--init-mode scratch" in res.stdout
    assert "--action-horizon 50" in res.stdout
    assert "--init-joint" not in res.stdout

    env_gr00t = {
        **os.environ,
        "INIT_MODE": "gr00t_n1d7",
        "DATASET_FORMAT": "sonic_lerobot",
        "DATASET": str(dataset),
        "OUTPUT_DIR": str(tmp_path / "run_gr00t"),
        "QWEN_MODEL": str(tmp_path / "gr00t_backbone"),
        "GR00T_MODEL": str(tmp_path / "gr00t_base"),
        "GPUS": "1",
        "BATCH_SIZE": "16",
        "UPDATES": "100",
        "GR00T_ACTIVATION_CHECKPOINTING": "1",
        "PYTHON_BIN": sys.executable,
    }
    res_g = subprocess.run(
        ["bash", str(ROOT / "scripts/train/predictive_world/finetune-real-sonic.sh"), "--dry-run"],
        env=env_gr00t,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--init-mode gr00t_n1d7" in res_g.stdout
    assert "--action-horizon 40" in res_g.stdout
    assert "--gr00t-model " + str(tmp_path / "gr00t_base") in res_g.stdout
    assert "--gr00t-activation-checkpointing" in res_g.stdout
    assert "--init-joint" not in res_g.stdout


def test_real_sonic_runtime_attaches_content_boxes_for_content_area_v1(tmp_path):
    import base64
    from io import BytesIO
    from PIL import Image
    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.deploy.real_sonic_protocol import CAMERAS, FORMAT, PROTOCOL, STATE_SCHEMA
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    config = {
        "data": {
            "state_max_age_s": 0.04,
            "visual_max_age_s": 0.08,
            "current_visual_pooling": "content_area_v1",
        },
        "training": {"rtc_training_max_delay": 6, "action_condition_time_mode": "fixed_action_grid_v1"},
    }
    source = {
        "metadata": {
            "resolved_config_sha256": canonical_sha256(config),
            "optimization_contract": {
                "rtc_training_max_delay": 6,
                "action_condition_time_mode": "fixed_action_grid_v1",
            },
        }
    }
    (tmp_path / "original_run_config.json").write_text(json.dumps(config))
    source_path = tmp_path / "source_checkpoint.json"
    source_path.write_text(json.dumps(source))
    (tmp_path / "feature_provenance.json").write_text(json.dumps({"image_preprocessing": "aspect_v1", "feature_shape": [16, 2048]}))
    (tmp_path / "registries").mkdir()
    (tmp_path / "registries/action_protocol_registry.json").write_text(
        json.dumps({PROTOCOL: {"dim": 76, "normalization_space": "native"}})
    )
    manifest = {
        "format": "zeta0.research_inference_weights.v1",
        "action_protocol_id": PROTOCOL,
        "state_schema_id": STATE_SCHEMA,
        "visual_keys": list(CAMERAS),
        "task": "Place the chair",
        "source_checkpoint_sha256": file_sha256(source_path),
        "temporal_abi": {"action_hz": 30, "action_horizon": 40, "state_query_hz": 50, "state_history_count": 1},
        "architecture": {"host_architecture": "gr00t_n1d7"},
    }
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest))

    class _DummyProcessor:
        def apply_chat_template(self, conv, *, tokenize, add_generation_prompt):
            return "prompt"

        def __call__(self, *, text, images, padding, return_tensors):
            return {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "pixel_values": torch.zeros((len(images), 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long),
            }

    buf = BytesIO()
    Image.new("RGB", (640, 480), (40, 80, 120)).save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode("ascii")
    req = {
        "format": FORMAT,
        "session_id": "s1",
        "request_seq": 0,
        "queue_revision": 1,
        "anchor_time": 1.0,
        "action_protocol_id": PROTOCOL,
        "state_schema_id": STATE_SCHEMA,
        "state": {"values": [[0.0] * 41], "valid": [[True] * 41], "times": [0.0]},
        "images": {cam: {"time": 0.0, "encoded": encoded_img} for cam in CAMERAS},
        "committed_prefix": [],
    }
    runtime = ResearchInferenceRuntime(tmp_path)
    observed = runtime.prepare_observation(req, _DummyProcessor(), device="cpu")
    assert "image_content_boxes" in observed.qwen_inputs
    assert tuple(observed.qwen_inputs["image_content_boxes"].shape) == (3, 4)
