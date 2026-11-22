import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "scripts/train/predictive_world/submit-two-stage-5n40g.sh"
ONE_COMMAND_SUBMIT = ROOT / "scripts/train/predictive_world/submit-pretrain-5n40g.sh"


def _configs(tmp_path: Path) -> tuple[Path, Path]:
    shared = {
        "route": "predictive_world",
        "data": {"index_path": "/data/index.jsonl"},
        "model": {"history_hidden_dim": 1024},
    }
    world_output = tmp_path / "world"
    world = {
        **shared,
        "training": {"phase": "world_model"},
        "runtime": {"output_dir": str(world_output)},
    }
    joint = {
        **shared,
        "training": {
            "phase": "joint",
            "world_model": str(world_output / "world-model-final"),
        },
        "runtime": {"output_dir": str(tmp_path / "joint")},
    }
    world_path = tmp_path / "world.json"
    joint_path = tmp_path / "joint.json"
    world_path.write_text(json.dumps(world), encoding="utf-8")
    joint_path.write_text(json.dumps(joint), encoding="utf-8")
    return world_path, joint_path


def _dry_run(world: Path, joint: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "WORLD_MODEL_CONFIG": str(world),
        "JOINT_CONFIG": str(joint),
    }
    return subprocess.run(
        ["bash", str(SUBMIT)],
        check=False,
        capture_output=True,
        cwd=ROOT,
        env=environment,
        text=True,
    )


def test_two_stage_submit_validates_complete_chain_without_submitting(tmp_path):
    world, joint = _configs(tmp_path)
    result = _dry_run(world, joint)
    assert result.returncode == 0, result.stderr
    assert "validated world_model=" in result.stdout
    assert "validated joint=" in result.stdout


def test_two_stage_submit_rejects_a_moving_or_wrong_world_model_target(tmp_path):
    world, joint = _configs(tmp_path)
    payload = json.loads(joint.read_text(encoding="utf-8"))
    payload["training"]["world_model"] = str(tmp_path / "wrong-bundle")
    joint.write_text(json.dumps(payload), encoding="utf-8")
    result = _dry_run(world, joint)
    assert result.returncode != 0
    assert "joint training.world_model must equal" in result.stderr


def _release(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "release"
    artifact.mkdir()
    for name in (
        "measurement_registry.json",
        "capability_registry.json",
        "action_protocol_registry.json",
        "state_schema_registry.json",
        "source_audit.json",
    ):
        (artifact / name).write_text("{}\n", encoding="utf-8")
    (artifact / "predictive_world_index.jsonl").write_text("\n", encoding="utf-8")
    feature_cache = artifact / "feature_cache"
    feature_cache.mkdir()
    (feature_cache / "feature_manifest.json").write_text("{}\n", encoding="utf-8")
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text("{}\n", encoding="utf-8")
    return artifact, qwen


def _one_command_dry_run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    artifact, qwen = (
        (tmp_path / "release", tmp_path / "qwen")
        if (tmp_path / "release").exists() else _release(tmp_path)
    )
    environment = {
        **os.environ,
        "ARTIFACT_ROOT": str(artifact),
        "QWEN_MODEL": str(qwen),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "RUN_ID": "test-run",
        "LAUNCH_BACKEND": "slurm",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "MICRO_BATCH_SIZE": "256",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "JOINT_MICRO_BATCH_SIZE": "220",
        "JOINT_GRADIENT_ACCUMULATION_STEPS": "1",
        "JOINT_RUN_ROOT": "",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(ONE_COMMAND_SUBMIT)],
        check=False,
        capture_output=True,
        cwd=ROOT,
        env=environment,
        text=True,
    )


def test_one_command_submit_resolves_and_submits_both_preflights_and_stages(tmp_path):
    result = _one_command_dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("DRY_RUN sbatch stage=") == 4
    run = tmp_path / "runs/test-run"
    joint_run = run / "joint-b220-a1"
    world = json.loads((run / "configs/world.json").read_text(encoding="utf-8"))
    joint = json.loads((joint_run / "configs/joint.json").read_text(encoding="utf-8"))
    assert world["training"]["max_updates"] == 15_000
    assert joint["training"]["max_updates"] == 20_000
    assert world["training"]["micro_batch"] == 256
    assert world["training"]["gradient_accumulation_steps"] == 1
    assert world["training"]["effective_global_batch"] == 10_240
    assert joint["training"]["micro_batch"] == 220
    assert joint["training"]["gradient_accumulation_steps"] == 1
    assert joint["training"]["effective_global_batch"] == 8_800
    assert joint["training"]["phase"] == "joint"
    assert joint["runtime"]["output_dir"] == str(joint_run / "joint")
    assert world["model"]["latent_transition"]["semantics"] == "parallel_process"
    assert world["training"]["forecast_control_loss_weight"] == 1.0
    assert joint["training"]["forecast_control_loss_weight"] == 1.0
    assert world["training"]["forecast_control_scope"] == "prior"
    assert joint["training"]["forecast_control_scope"] == "prior"
    assert world["data"]["observed_visual_mode"] == "current_multiview"
    assert world["data"]["history_visual_offsets_s"] == [-0.4, -0.2, 0.0]
    assert world["data"]["state_history_count"] == 21
    assert world["data"] == joint["data"]
    assert world["model"] == joint["model"]
    assert joint["training"]["world_model"] == str(
        run / "world-model/world-model-final"
    )
    assert (run / "configs/world_preflight.json").is_file()
    joint_preflight = json.loads((joint_run / "configs/joint_preflight.json").read_text())
    assert joint_preflight["training"] == joint["training"]
    assert joint_preflight["runtime"]["output_dir"] == str(joint_run / "preflight/joint")


def test_joint_batch_change_preserves_completed_world_and_old_joint_configs(tmp_path):
    first = _one_command_dry_run(tmp_path, JOINT_MICRO_BATCH_SIZE="256")
    assert first.returncode == 0, first.stderr
    run = tmp_path / "runs/test-run"
    original_configs = {path: path.read_bytes() for path in (run / "configs").glob("*.json")}
    (run / "world-model/world-model-final").mkdir(parents=True)
    (run / "world-model/world-model-final/.complete").write_text("complete\n")
    (run / "world-model/.train_complete").write_text("15000\n")
    (run / "preflight/world-model").mkdir(parents=True)
    (run / "preflight/world-model/memory_preflight_report.json").write_text('{"status":"passed"}\n')

    retry = _one_command_dry_run(tmp_path)
    assert retry.returncode == 0, retry.stderr
    assert "stage=world-preflight status=reused" in retry.stdout
    assert "stage=world-model status=already-complete" in retry.stdout
    assert retry.stdout.count("DRY_RUN sbatch stage=") == 2
    assert "stage=joint-preflight dependency=none" in retry.stdout
    assert "stage=joint dependency=dry-joint-preflight" in retry.stdout
    for path, original in original_configs.items():
        assert path.read_bytes() == original
    joint = json.loads((run / "joint-b220-a1/configs/joint.json").read_text())
    assert joint["training"]["world_model"] == str(run / "world-model/world-model-final")
    assert joint["training"]["micro_batch"] == 220
    repeated = _one_command_dry_run(tmp_path)
    assert repeated.returncode == 0, repeated.stderr


def test_joint_accumulation_uses_a_separate_contract_without_changing_world(tmp_path):
    result = _one_command_dry_run(
        tmp_path, JOINT_MICRO_BATCH_SIZE="128", JOINT_GRADIENT_ACCUMULATION_STEPS="2"
    )
    assert result.returncode == 0, result.stderr
    run = tmp_path / "runs/test-run"
    world = json.loads((run / "configs/world.json").read_text())
    joint = json.loads((run / "joint-b128-a2/configs/joint.json").read_text())
    assert world["training"]["micro_batch"] == 256
    assert world["training"]["gradient_accumulation_steps"] == 1
    assert joint["training"]["micro_batch"] == 128
    assert joint["training"]["gradient_accumulation_steps"] == 2
    assert joint["training"]["effective_global_batch"] == 10_240


def test_submit_does_not_export_world_batch_overrides_to_joint(tmp_path):
    executable = tmp_path / "bin/sbatch"
    executable.parent.mkdir()
    executable.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'row = {"args":sys.argv[1:], "batch_env":{k:os.environ[k] for k in '
        '("MICRO_BATCH_SIZE","GRADIENT_ACCUMULATION_STEPS","EFFECTIVE_GLOBAL_BATCH") if k in os.environ}}\n'
        'with Path(os.environ["SUBMISSION_CAPTURE"]).open("a") as f: f.write(json.dumps(row)+"\\n")\n'
        'print(123)\n'
    )
    executable.chmod(0o755)
    capture = tmp_path / "submissions.jsonl"
    result = _one_command_dry_run(
        tmp_path, DRY_RUN="0", PATH=f"{executable.parent}:{os.environ['PATH']}",
        SUBMISSION_CAPTURE=str(capture), MICRO_BATCH_SIZE="256", EFFECTIVE_GLOBAL_BATCH="10240",
    )
    assert result.returncode == 0, result.stderr
    submissions = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(submissions) == 4
    assert all(not item["batch_env"] for item in submissions)
    for item in submissions[-2:]:
        export = next(arg for arg in item["args"] if arg.startswith("--export="))
        assert "joint-b220-a1/configs/joint" in export


def test_one_command_submit_keeps_resolved_run_immutable(tmp_path):
    first = _one_command_dry_run(tmp_path)
    assert first.returncode == 0, first.stderr
    environment = {
        **os.environ,
        "ARTIFACT_ROOT": str(tmp_path / "release"),
        "QWEN_MODEL": str(tmp_path / "qwen"),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "RUN_ID": "test-run",
        "LAUNCH_BACKEND": "slurm",
        "DRY_RUN": "1",
        "MICRO_BATCH_SIZE": "64",
    }
    second = subprocess.run(
        ["bash", str(ONE_COMMAND_SUBMIT)], check=False, capture_output=True,
        cwd=ROOT, env=environment, text=True,
    )
    assert second.returncode != 0
    assert "immutable resolved config already exists with different content" in second.stderr


def test_current_multiview_default_uses_separate_run_directory(tmp_path):
    result = _one_command_dry_run(tmp_path, RUN_ID='')
    assert result.returncode == 0, result.stderr
    runs = list((tmp_path / 'runs').iterdir())
    assert len(runs) == 1
    assert runs[0].name.endswith('-b256-a1-current-multiview-v1-checkpoint-fix-v1-early20')
