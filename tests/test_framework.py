"""User-facing entrypoints: no downloads, shell interpolation or robot resources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from psi.framework.bundle import inspect_bundle
from psi.framework.data import inspect_data
from psi.framework.doctor import doctor
from psi.framework.logs import summarize_log
from psi.framework.workflow import WORKFLOWS, execute_plan, workflow_plan

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "psi.cli", *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=40)


def test_dependency_light_imports_and_contract_export_compatibility():
    # Block tensor/runtime imports to prove help/metadata tools work without them.
    code = f"""import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'lerobot', 'pyarrow', 'transformers'):
            raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, Reject())
from psi.cli import parser
parser().format_help()
from psi.framework.doctor import doctor
from psi.framework.data import inspect_data
from psi.framework.bundle import inspect_bundle
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    from psi.data.predictive_world import ObservedHistory
    from psi.data.predictive_world.contracts import ObservedHistory as original
    assert ObservedHistory is original


def test_doctor_reuses_the_selected_environment_and_relative_asset_map(tmp_path):
    asset = tmp_path / "model"
    asset.mkdir()
    mapping = tmp_path / "assets.json"
    mapping.write_text(json.dumps({"/original/model": "model"}))
    before = mapping.read_bytes()
    report = doctor(ROOT, selected=[sys.executable], profile="core", asset_path_map=str(mapping))
    assert report["ok"]
    assert report["reuse_candidates"] == [str(Path(sys.executable).absolute())]
    assert report["environments"][0]["prefix"] == sys.prefix
    assert report["assets"][0]["local_path"] == str(asset)
    assert report["downloads"] == 0 and not report["environments_modified"]
    assert mapping.read_bytes() == before
    asset.rmdir()
    assert not doctor(ROOT, selected=[sys.executable], profile="core", asset_path_map=str(mapping))["ok"]


def test_missing_interpreter_is_reported_without_installation(tmp_path):
    report = doctor(None, selected=[str(tmp_path / "missing")], profile="train", asset_path_map=None)
    assert not report["ok"] and not report["reuse_candidates"]
    assert report["environments"][0]["error"]
    assert list(tmp_path.iterdir()) == []


def test_data_metadata_is_not_training_admission(tmp_path):
    (tmp_path / "meta").mkdir()
    path = tmp_path / "meta/info.json"
    path.write_text(json.dumps(dict(codebase_version="v3.0", robot_type="sonic_g1", fps=30,
                                    total_episodes=1, total_frames=20, features={})))
    before = path.read_bytes()
    report = inspect_data(str(tmp_path))
    assert report["candidate_adapter"] == "sonic_lerobot"
    assert not report["training_admission_decided"] and not report["numerical_contract_verified"]
    assert path.read_bytes() == before
    assert len(list(tmp_path.rglob("*"))) == 2


def test_metadata_cli_report_cannot_overwrite_previous_audit(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"features":{}}')
    report = tmp_path / "report.json"
    result = cli("--json", "data", "inspect", tmp_path, "--output", report)
    assert result.returncode == 0, result.stderr
    before = report.read_bytes()
    result = cli("--json", "data", "inspect", tmp_path, "--output", report)
    assert result.returncode == 2
    assert report.read_bytes() == before


@pytest.mark.parametrize("name", WORKFLOWS)
def test_all_plans_preserve_arguments_without_running_anything(name, tmp_path):
    values = ["--output", str(tmp_path / "a b;$(touch bad)"), "--help"]
    result = cli("--root", ROOT, "--json", "workflow", "--python", sys.executable, name, "--", *values, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["argv"][2:] == values
    assert report["entrypoint_exists"] and not report["executed"]
    assert list(tmp_path.iterdir()) == []


def test_real_demo_runs_from_another_directory_and_preserves_resume(tmp_path):
    output = tmp_path / "demo result"
    result = cli("--root", ROOT, "workflow", "--python", sys.executable, "--execute", "demo",
                 "--", "--device", "cpu", "--output", output, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["resumed_at"] == 4 and report["completed_updates"] == 8
    assert report["final_mse"] < report["initial_mse"]
    assert not report["robot_commands_sent"]


def test_dispatch_preserves_exit_code_and_separates_hardware_pythonpath(tmp_path, monkeypatch):
    root = tmp_path / "source"
    entry = root / "real/g1/scripts/deploy_online_model.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("import os,sys\nassert 'PYTHONPATH' not in os.environ\nraise SystemExit(7)\n")
    monkeypatch.setenv("PYTHONPATH", "/wrong/training/source")
    plan = workflow_plan(root, "deploy", [], sys.executable)
    assert execute_plan(plan, root) == 7


def test_bundle_full_verification_rejects_corruption(tmp_path):
    from psi.deploy.research_inference_weights import INFERENCE_FORMAT, INFERENCE_MODULES
    weights = tmp_path / "inference.safetensors"
    weights.write_bytes(b"file hash fixture; not a loadable tensor model")
    manifest = {"format": INFERENCE_FORMAT, "module_names": sorted(INFERENCE_MODULES),
                "weights_file": weights.name, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest))
    assert not inspect_bundle(str(tmp_path))["hashes_verified"]
    report = inspect_bundle(str(tmp_path), full=True)
    assert report["hashes_verified"] and report["legacy_package"]
    assert not report["source_contract_verified"] and not report["hardware_acceptance_verified"]
    weights.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        inspect_bundle(str(tmp_path), full=True)


def test_logs_keep_missing_metrics_unknown_and_report_truncation(tmp_path):
    log = tmp_path / "trace.jsonl"
    values = [{"event": "audit_start"},
              {"event": "accepted", "inference_seconds": .1, "anchor_to_accept_s": .2},
              {"event": "server_error", "request_seq": 2},
              {"event": "accepted", "inference_seconds": .3},
              {"event": "audit_end", "dropped": 4}]
    log.write_text("\n".join(json.dumps(item) for item in values))
    report = summarize_log(str(log))
    assert report["audit_counts_balanced"] and report["reported_dropped_events"] == 4
    assert report["metrics"]["inference_seconds"]["p95"] == .3
    assert "queue_depth" not in report["metrics"]
    assert report["first_errors"][0]["line"] == 3
    assert not report["robot_commands_sent"]
    partial = summarize_log(str(log), max_events=2)
    assert partial["truncated"] and not partial["audit_counts_balanced"]
    log.write_text(log.read_text() + '\n{"event":')
    with pytest.raises(ValueError, match="line 6"):
        summarize_log(str(log))


def test_logs_read_nested_events_from_the_real_client_audit_writer(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_writer", ROOT / "real/g1/scripts/runtime_audit.py")
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    log = tmp_path / "client.jsonl"
    with writer.RuntimeAudit(log) as audit:
        audit.record("client_event", data={"event": "reply_rejected", "request_seq": 7,
                                          "session_id": "session", "reason": "invalid reply"})
        audit.record("client_event", data={"event": "model_running", "queue_depth": 12})
        audit.record("inference_accepted", inference_seconds=.2)
    report = summarize_log(str(log))
    assert report["events_read"] == 5
    assert report["event_counts"]["client_event"] == 2
    assert report["client_event_counts"] == {"reply_rejected": 1, "model_running": 1}
    assert report["first_errors"] == [{"line": 2, "event": "reply_rejected",
                                        "session_id": "session", "request_seq": 7}]
    assert report["metrics"]["queue_depth"] == {"samples": 1, "p50": 12, "p95": 12, "max": 12}
    assert report["metrics"]["inference_seconds"]["samples"] == 1
    assert report["audit_counts_balanced"]


@pytest.mark.parametrize("data", [None, {}, {"event": 7}])
def test_logs_reject_malformed_client_events(tmp_path, data):
    log = tmp_path / "client.jsonl"
    log.write_text(json.dumps({"event": "client_event", "data": data}))
    with pytest.raises(ValueError, match="client event name at log line 1"):
        summarize_log(str(log))


def test_custom_task_configuration_in_isolated_runtime(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "inference_manifest.json").write_text(json.dumps({
        "task": "Place the object", "temporal_abi": {"action_hz": 50, "action_horizon": 40}}))
    config = tmp_path / "models.json"
    payload = {"format": "local_zeta0_model_paths.v1", "runtime_profile": "unified",
               "qwen_model": "encoder", "tasks": {"new_task": {
                   "weights_dir": "weights", "task": "Place the object", "action_hz": 50, "action_horizon": 40}}}
    config.write_text(json.dumps(payload))
    before = config.read_bytes()
    code = f"""import sys,os
sys.path.insert(0,{str(ROOT / 'real/g1/scripts')!r})
from policy_profile import model_paths, runtime_kit
paths=model_paths('new_task',{str(config)!r})
assert paths['tasks']['new_task']['weights_dir']=={str(weights)!r}
assert runtime_kit().name=='unified'
import online_runtime
root=online_runtime.import_runtime()
import psi
assert str(root) in psi.__file__
assert 'torch' not in sys.modules
"""
    env = dict(os.environ)
    env.pop("CHENYZ_POLICY_PROFILE", None)
    env.pop("ZETA0_POLICY_PROFILE", None)
    result = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert config.read_bytes() == before
    payload["tasks"]["new_task"]["action_hz"] = 30
    payload["tasks"]["new_task"]["action_horizon"] = 50
    config.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "disagrees" in result.stderr
