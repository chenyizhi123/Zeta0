"""Export a traceable, path-free diagnostic summary from evaluator JSONL records.

Only identifier-valued strings and the documented scalar fields are published.
Completed episodes exclude process/simulator errors, including errors whose outer
worker returned zero. This export does not certify benchmark comparability.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


FIELDS = (
    "task", "model_label", "seed", "repeat_idx", "episode_seed", "episode_index",
    "success", "failure_reason", "episode_steps", "max_steps", "returncode", "worker_id",
)
STRING_FIELDS = ("task", "model_label", "failure_reason")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
INFRASTRUCTURE_REASONS = frozenset({"process_error", "sim_error", "sim_stopped", "worker_error"})


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a nonempty identifier without paths or URLs")
    return value


def _record(value: object, line: int) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"line {line}: expected a JSON object")
    for field in FIELDS:
        if field not in value:
            if field in {"returncode", "worker_id"}:
                raise ValueError(
                    f"line {line}: pending record: missing required field {field}; "
                    "wait for worker finalization"
                )
            raise ValueError(f"line {line}: missing required field {field}")
        if field in {"returncode", "worker_id"} and value[field] is None:
            raise ValueError(f"line {line}: pending record: {field} has not been finalized")
    result = {field: value[field] for field in FIELDS}
    for field in STRING_FIELDS:
        _identifier(result[field], f"line {line}: {field}")
    if type(result["success"]) is not bool:
        raise ValueError(f"line {line}: success must be boolean")
    for field in set(FIELDS) - set(STRING_FIELDS) - {"success"}:
        item = result[field]
        if type(item) is not int:
            raise ValueError(f"line {line}: {field} must be an integer")
        if field != "returncode" and item < 0:
            raise ValueError(f"line {line}: {field} must be nonnegative")
    if result["max_steps"] <= 0:
        raise ValueError(f"line {line}: max_steps must be positive")
    if result["episode_steps"] > result["max_steps"]:
        raise ValueError(f"line {line}: episode_steps exceeds max_steps")
    if result["success"] != (result["failure_reason"] == "success"):
        raise ValueError(f"line {line}: success disagrees with failure_reason")
    return result


def _wilson_interval(successes: int, completed: int) -> dict | None:
    """Binomial episode uncertainty; not uncertainty across training seeds."""
    if completed == 0:
        return None
    z = 1.959963984540054
    rate = successes / completed
    denominator = 1 + z * z / completed
    center = (rate + z * z / (2 * completed)) / denominator
    radius = z * math.sqrt(
        rate * (1 - rate) / completed + z * z / (4 * completed * completed)
    ) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def _aggregate(records: list[dict]) -> dict:
    infrastructure = [
        row for row in records
        if row["returncode"] != 0 or row["failure_reason"] in INFRASTRUCTURE_REASONS
    ]
    completed = [
        row for row in records
        if row["returncode"] == 0 and row["failure_reason"] not in INFRASTRUCTURE_REASONS
    ]
    successes = sum(row["success"] for row in completed)
    failures = [row for row in completed if not row["success"]]
    single_cohort = len({(row["task"], row["model_label"]) for row in records}) == 1
    return {
        "attempted_episodes": len(records),
        "completed_episodes": len(completed),
        "successes": successes,
        "policy_failures": len(failures),
        "infrastructure_failures": len(infrastructure),
        "nonzero_returncodes": sum(row["returncode"] != 0 for row in records),
        "success_rate": successes / len(completed) if completed else None,
        "success_rate_ci95": _wilson_interval(successes, len(completed)) if single_cohort else None,
        "policy_failure_reason_counts": dict(sorted(Counter(
            row["failure_reason"] for row in failures
        ).items())),
        "infrastructure_failure_reason_counts": dict(sorted(Counter(
            row["failure_reason"] for row in infrastructure
        ).items())),
    }


def _per_seed(records: list[dict]) -> dict:
    return {
        str(seed): _aggregate([row for row in records if row["seed"] == seed])
        for seed in sorted({row["seed"] for row in records})
    }


def _task_macro(records: list[dict], tasks: list[str]) -> dict:
    rates = {
        task: _aggregate([row for row in records if row["task"] == task])["success_rate"]
        for task in tasks
    }
    missing = [task for task, rate in rates.items() if rate is None]
    return {
        "task_count": len(tasks),
        "tasks_with_completed_episodes": len(tasks) - len(missing),
        "missing_completed_tasks": missing,
        "task_success_rates": rates,
        "success_rate": sum(rates.values()) / len(tasks) if not missing else None,
    }


def export_summary(source: Path, *, run_id: str, expected_tasks: list[str] | None = None) -> dict:
    """Validate one source before producing an allowlisted diagnostic document."""
    _identifier(run_id, "run_id")
    if expected_tasks is not None:
        if not expected_tasks:
            raise ValueError("expected_tasks must contain at least one task")
        for task in expected_tasks:
            _identifier(task, "expected task")
        if len(set(expected_tasks)) != len(expected_tasks):
            raise ValueError("expected_tasks contains duplicates")
    content = source.read_bytes()
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("input must be UTF-8 JSONL") from error
    records = []
    identities = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"line {number}: empty JSONL record")
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"line {number}: invalid JSON") from error
        row = _record(value, number)
        identity = (row["model_label"], row["task"], row["seed"], row["repeat_idx"])
        if identity in identities:
            raise ValueError(f"line {number}: duplicate model_label/task/seed/repeat_idx")
        identities.add(identity)
        records.append(row)
    if not records:
        raise ValueError("input contains no episode records")
    observed_tasks = {row["task"] for row in records}
    if expected_tasks is not None and not observed_tasks <= set(expected_tasks):
        raise ValueError("input contains a task outside expected_tasks")
    tasks = sorted(expected_tasks if expected_tasks is not None else observed_tasks)
    cohorts = []
    for task, model in sorted({(row["task"], row["model_label"]) for row in records}):
        rows = [row for row in records if row["task"] == task and row["model_label"] == model]
        cohorts.append({
            "task": task,
            "model_label": model,
            "aggregate": _aggregate(rows),
            "per_seed": _per_seed(rows),
        })
    models = []
    for model in sorted({row["model_label"] for row in records}):
        rows = [row for row in records if row["model_label"] == model]
        models.append({
            "model_label": model,
            "aggregate": _aggregate(rows),
            "task_macro": _task_macro(rows, tasks),
        })
    return {
        "schema_version": 2,
        "run_id": run_id,
        "diagnostic_only": True,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_format": "summary.jsonl",
        "aggregation_contract": {
            "version": 2,
            "success_denominator": "completed_episodes",
            "infrastructure": "nonzero returncode or listed infrastructure failure reason",
            "infrastructure_reasons": sorted(INFRASTRUCTURE_REASONS),
            "overall_weighting": "episode_micro_average",
            "model_macro_weighting": "equal_task_average; null if any task has no completed episodes",
            "task_set": tasks,
            "task_set_source": "explicit_expected_tasks" if expected_tasks is not None else "observed_union",
            "uncertainty": (
                "95% Wilson binomial interval only within one model/task cohort; "
                "null for mixed tasks/models or no completed episodes; not training-seed uncertainty"
            ),
            "comparability": "not established by this export",
        },
        "aggregate": _aggregate(records),
        "per_seed": _per_seed(records),
        "cohorts": cohorts,
        "model_summaries": models,
        "episodes": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Evaluator summary.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="New diagnostic JSON file")
    parser.add_argument("--run-id", required=True, help="Public identifier for this run")
    parser.add_argument(
        "--expected-task", action="append", dest="expected_tasks",
        help="Expected task identifier (repeat for all tasks); absent tasks block the macro average",
    )
    args = parser.parse_args(argv)
    try:
        document = export_summary(args.input, run_id=args.run_id, expected_tasks=args.expected_tasks)
        # Never overwrite source data or an earlier published diagnostic.
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(document, output, indent=2, ensure_ascii=True, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as error:
        # OSError messages can include private local paths; schema errors never
        # interpolate untrusted field values or raw JSON into the diagnostic.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Export failed: {message}", file=sys.stderr)
        return 1
    print(f"Exported {len(document['episodes'])} diagnostic episode records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
