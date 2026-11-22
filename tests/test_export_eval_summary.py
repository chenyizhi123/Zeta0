"""CLI contracts for public exports of private evaluation logs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_eval_summary.py"
FIELDS = {
    "task", "model_label", "seed", "repeat_idx", "episode_seed", "episode_index",
    "success", "failure_reason", "episode_steps", "max_steps", "returncode", "worker_id",
}


def episode(**overrides):
    record = {
        "task": "task-a", "model_label": "model-a", "seed": 0, "repeat_idx": 0,
        "episode_seed": 123, "episode_index": 0, "success": True,
        "failure_reason": "success", "episode_steps": 3, "max_steps": 10,
        "returncode": 0, "worker_id": 0,
        "model_path": "/private/checkpoint", "server_url": "http://private.invalid",
        "args": {"token": "do-not-export"},
    }
    return {**record, **overrides}


def invoke(tmp_path, records, *, raw=None, run_id="diagnostic-run", expected_tasks=None):
    source = tmp_path / "summary.jsonl"
    source.write_text(
        raw if raw is not None else "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    output = tmp_path / "export.json"
    command = [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output),
               "--run-id", run_id]
    for task in expected_tasks or []:
        command.extend(["--expected-task", task])
    result = subprocess.run(
        command,
        capture_output=True, text=True, timeout=10, check=False,
    )
    return result, source, output


def test_cli_exports_allowlist_and_computes_actual_denominators(tmp_path):
    records = [
        episode(),
        episode(repeat_idx=1, success=False, failure_reason="fall"),
        episode(repeat_idx=2, success=False, failure_reason="worker_error", returncode=-1),
        episode(seed=1, success=False, failure_reason="sim_error"),
        # A success marker cannot make a failed process count as completed.
        episode(seed=1, repeat_idx=1, returncode=2),
        episode(seed=2),
    ]
    result, source, output = invoke(tmp_path, records)
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    report = json.loads(text)
    assert report["schema_version"] == 2
    assert report["aggregation_contract"]["version"] == 2
    assert report["diagnostic_only"] is True
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    aggregate = report["aggregate"]
    assert aggregate["attempted_episodes"] == 6
    assert aggregate["completed_episodes"] == 3
    assert aggregate["successes"] == 2
    assert aggregate["policy_failures"] == 1
    assert aggregate["infrastructure_failures"] == 3
    assert aggregate["nonzero_returncodes"] == 2
    assert aggregate["success_rate"] == pytest.approx(2 / 3)
    assert aggregate["policy_failure_reason_counts"] == {"fall": 1}
    assert report["per_seed"]["0"]["success_rate"] == 0.5
    assert report["per_seed"]["1"]["success_rate"] is None
    assert report["cohorts"][0]["aggregate"] == aggregate
    assert all(set(row) == FIELDS for row in report["episodes"])
    for private in ("/private", "http://", "do-not-export", str(tmp_path), "model_path"):
        assert private not in text


@pytest.mark.parametrize("change", [
    {"success": 1}, {"seed": True}, {"returncode": "0"}, {"episode_steps": -1},
    {"episode_steps": 11}, {"max_steps": 0}, {"task": "/private/task"},
    {"model_label": "https://private.invalid"}, {"failure_reason": "private/path"},
    {"failure_reason": "timeout"},
])
def test_rejects_invalid_types_values_and_sensitive_strings(tmp_path, change):
    result, _, output = invoke(tmp_path, [episode(**change)])
    assert result.returncode == 1
    assert not output.exists()
    assert "private" not in result.stderr


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_requires_every_exported_field(tmp_path, field):
    row = episode()
    del row[field]
    result, _, output = invoke(tmp_path, [row])
    assert result.returncode == 1
    assert f"missing required field {field}" in result.stderr
    assert not output.exists()


def test_rejects_duplicate_episode_with_same_model(tmp_path):
    result, _, output = invoke(tmp_path, [episode(), episode()])
    assert result.returncode == 1
    assert "duplicate model_label/task/seed/repeat_idx" in result.stderr
    assert not output.exists()


def test_same_episode_is_allowed_for_distinct_models(tmp_path):
    result, _, output = invoke(tmp_path, [episode(), episode(model_label="another-model")])
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert len(report["cohorts"]) == 2
    assert len(report["model_summaries"]) == 2
    assert report["aggregate"]["success_rate_ci95"] is None
    assert report["per_seed"]["0"]["success_rate_ci95"] is None


def test_sim_stopped_is_infrastructure_even_with_successful_worker_exit(tmp_path):
    result, _, output = invoke(tmp_path, [
        episode(), episode(repeat_idx=1, success=False, failure_reason="sim_stopped"),
    ])
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    aggregate = report["aggregate"]
    assert aggregate["attempted_episodes"] == 2
    assert aggregate["completed_episodes"] == 1
    assert aggregate["infrastructure_failures"] == 1
    assert aggregate["nonzero_returncodes"] == 0
    assert aggregate["policy_failures"] == 0
    assert aggregate["success_rate"] == 1
    assert aggregate["infrastructure_failure_reason_counts"] == {"sim_stopped": 1}
    assert "sim_stopped" in report["aggregation_contract"]["infrastructure_reasons"]


@pytest.mark.parametrize("field", ["returncode", "worker_id"])
@pytest.mark.parametrize("missing", [False, True])
def test_pending_worker_records_are_rejected_without_becoming_failures(tmp_path, field, missing):
    row = episode()
    if missing:
        del row[field]
    else:
        row[field] = None
    result, _, output = invoke(tmp_path, [row])
    assert result.returncode == 1
    assert "pending record" in result.stderr and field in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("successes,lower,upper", [
    (0, 0.0, 0.2775328), (5, 0.2365931, 0.7634069), (10, 0.7224672, 1.0),
])
def test_cohort_wilson_interval_and_explicit_seed_zero(tmp_path, successes, lower, upper):
    rows = [
        episode(
            repeat_idx=index, success=index < successes,
            failure_reason="success" if index < successes else "timeout",
        )
        for index in range(10)
    ]
    result, _, output = invoke(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    interval = report["cohorts"][0]["aggregate"]["success_rate_ci95"]
    assert interval["lower"] == pytest.approx(lower, abs=1e-7)
    assert interval["upper"] == pytest.approx(upper, abs=1e-7)
    assert report["cohorts"][0]["per_seed"]["0"]["success_rate_ci95"] == interval
    assert all(row["seed"] == 0 for row in report["episodes"])


def test_task_macro_weights_tasks_equally_and_does_not_pool_wilson_interval(tmp_path):
    rows = [episode(task="task-a", repeat_idx=index) for index in range(3)]
    rows.append(episode(task="task-b", success=False, failure_reason="timeout"))
    result, _, output = invoke(tmp_path, rows, expected_tasks=["task-a", "task-b"])
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["aggregate"]["success_rate"] == 0.75
    assert report["aggregate"]["success_rate_ci95"] is None
    assert report["per_seed"]["0"]["success_rate_ci95"] is None
    model = report["model_summaries"][0]
    assert model["aggregate"]["success_rate_ci95"] is None
    assert model["task_macro"]["success_rate"] == 0.5
    assert model["task_macro"]["task_success_rates"] == {"task-a": 1, "task-b": 0}
    assert model["task_macro"]["missing_completed_tasks"] == []
    assert "success_rate_ci95" not in model["task_macro"]
    assert report["aggregation_contract"]["task_set_source"] == "explicit_expected_tasks"


@pytest.mark.parametrize("absent", [False, True])
def test_missing_completed_task_blocks_macro_without_silently_dropping_it(tmp_path, absent):
    rows = [episode()]
    if not absent:
        rows.append(episode(task="task-b", success=False, failure_reason="sim_stopped"))
    result, _, output = invoke(tmp_path, rows, expected_tasks=["task-a", "task-b"])
    assert result.returncode == 0, result.stderr
    macro = json.loads(output.read_text())["model_summaries"][0]["task_macro"]
    assert macro["task_count"] == 2
    assert macro["tasks_with_completed_episodes"] == 1
    assert macro["missing_completed_tasks"] == ["task-b"]
    assert macro["task_success_rates"] == {"task-a": 1, "task-b": None}
    assert macro["success_rate"] is None


def test_observed_task_union_is_shared_across_models(tmp_path):
    rows = [episode(), episode(model_label="model-b", task="task-b")]
    result, _, output = invoke(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["aggregation_contract"]["task_set_source"] == "observed_union"
    assert report["aggregation_contract"]["task_set"] == ["task-a", "task-b"]
    for model in report["model_summaries"]:
        assert model["task_macro"]["task_count"] == 2
        assert model["task_macro"]["success_rate"] is None


def test_no_completed_episodes_has_null_rate_and_interval(tmp_path):
    result, _, output = invoke(tmp_path, [episode(success=False, failure_reason="sim_stopped")])
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    for aggregate in (report["aggregate"], report["cohorts"][0]["aggregate"]):
        assert aggregate["attempted_episodes"] == 1
        assert aggregate["completed_episodes"] == 0
        assert aggregate["success_rate"] is None
        assert aggregate["success_rate_ci95"] is None
    assert report["model_summaries"][0]["task_macro"]["success_rate"] is None


@pytest.mark.parametrize("expected_tasks", [["other-task"], ["task-a", "task-a"], ["/private/task"]])
def test_expected_tasks_fail_closed(tmp_path, expected_tasks):
    result, _, output = invoke(tmp_path, [episode()], expected_tasks=expected_tasks)
    assert result.returncode == 1
    assert not output.exists()
    assert "/private" not in result.stderr


@pytest.mark.parametrize("raw", ["", "\n", "[]\n", "{invalid private data}\n"])
def test_rejects_empty_or_malformed_input_without_echoing_contents(tmp_path, raw):
    result, _, output = invoke(tmp_path, [], raw=raw)
    assert result.returncode == 1
    assert "private data" not in result.stderr
    assert not output.exists()


def test_rejects_private_run_identifier(tmp_path):
    result, _, output = invoke(tmp_path, [episode()], run_id="/private/run")
    assert result.returncode == 1
    assert "/private" not in result.stderr
    assert not output.exists()


def test_preserves_existing_output(tmp_path):
    output = tmp_path / "export.json"
    output.write_text("existing artifact\n")
    result, _, _ = invoke(tmp_path, [episode()])
    assert result.returncode == 1
    assert output.read_text() == "existing artifact\n"
    assert str(tmp_path) not in result.stderr
