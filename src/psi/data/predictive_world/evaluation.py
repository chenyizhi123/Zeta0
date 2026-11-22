"""Deterministic validation views and fail-closed quality gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from psi.config.predictive_world import predictive_world_architecture_contract


REQUIRED_VIEWS = frozenset({"all", "prior", "world", "policy"})
EVALUATION_FORMAT = "zeta0.predictive_world.evaluation.v2"
EVALUATION_RELEASE_FORMAT = "zeta0.predictive_world.evaluation_release.v1"


def evaluation_release_from_index(
    index_path: str | Path,
    *,
    split: str,
    count: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Select one immutable held-out release eligible for every required view."""

    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.data.predictive_world.index import (
        predictive_world_eligibility,
        verify_predictive_world_index,
    )

    if split not in {"validation", "test"}:
        raise ValueError("release evaluation only permits validation or test split")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("evaluation release count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evaluation release seed must be an integer")
    index = Path(index_path).expanduser().resolve()
    verify_predictive_world_index(index)
    candidates: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"predictive-world index row {line_number} is not an object")
        if str(row.get("split")) != split:
            continue
        eligibility = predictive_world_eligibility(row)
        if row.get("eligibility") != eligibility:
            raise ValueError("evaluation index eligibility disagrees with audited masks")
        # world implies prior; policy additionally requires complete future and
        # measurement supervision.  Their intersection is valid for all four
        # release views without silently changing the cohort per view.
        if not eligibility["world"] or not eligibility["policy"]:
            continue
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in candidates:
            raise ValueError("evaluation index sample IDs must be non-empty and unique")
        candidates[sample_id] = row
    if len(candidates) < count:
        raise ValueError(
            f"evaluation release requires {count} all-view eligible {split} samples, "
            f"but only {len(candidates)} are available"
        )
    selected = tuple(sorted(
        candidates,
        key=lambda sample_id: hashlib.sha256(f"{seed}:{sample_id}".encode()).digest(),
    )[:count])
    release = {
        "format": EVALUATION_RELEASE_FORMAT,
        "index_sha256": file_sha256(index),
        "split": split,
        "seed": seed,
        "sample_count": count,
        "eligibility": ["policy", "world"],
        "sample_ids_sha256": canonical_sha256(list(selected)),
    }
    return selected, release


def validate_release_record_ids(
    records: Sequence[Mapping[str, Any]], expected_sample_ids: Sequence[str]
) -> None:
    expected = {str(value) for value in expected_sample_ids}
    actual = {str(record.get("sample_id", "")) for record in records}
    if len(expected) != len(expected_sample_ids) or actual != expected:
        raise ValueError("evaluation records do not exactly match the immutable release sample IDs")


def fixed_sample(
    records: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
    *,
    required_views: frozenset[str] | None = None,
) -> list[Mapping[str, Any]]:
    if count <= 0:
        raise ValueError("evaluation sample count must be positive")
    unique: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["sample_id"]), str(record["view"]))
        if key in unique:
            raise ValueError(f"duplicate evaluation record {key}")
        unique[key] = record
    if required_views is None:
        return sorted(
            unique.values(),
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['sample_id']}:{row['view']}".encode()
            ).digest(),
        )[:count]
    if not required_views:
        raise ValueError("required evaluation views cannot be empty")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for (sample_id, view), record in unique.items():
        if view not in required_views:
            raise ValueError(f"unsupported evaluation view {view!r}")
        grouped[sample_id][view] = record
    incomplete = {
        sample_id: sorted(required_views - rows.keys())
        for sample_id, rows in grouped.items()
        if rows.keys() != required_views
    }
    if incomplete:
        example = next(iter(sorted(incomplete.items())))
        raise ValueError(f"evaluation samples lack required views; example={example}")
    if len(grouped) < count:
        raise ValueError(
            f"evaluation requires {count} complete samples, but only {len(grouped)} are present"
        )
    selected_ids = sorted(
        grouped,
        key=lambda sample_id: hashlib.sha256(f"{seed}:{sample_id}".encode()).digest(),
    )[:count]
    return [
        grouped[sample_id][view]
        for sample_id in selected_ids
        for view in sorted(required_views)
    ]


def checkpoint_evaluation_binding(
    checkpoint: str | Path,
    config_path: str | Path,
    config: Any,
    resolved_config: Mapping[str, Any],
    *,
    code_revision: str,
    code_fingerprint: str,
    evaluation_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and bind evaluation to one policy/config/data/code identity."""

    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.models.predictive_world.checkpoint import (
        predictive_world_bundle_identity,
        read_predictive_world_bundle,
    )

    sidecar = read_predictive_world_bundle(checkpoint)
    if sidecar["phase"] not in {"joint", "policy"}:
        raise ValueError("quality evaluation must target an immutable action-policy bundle")
    if sidecar["phase"] != config.training.phase:
        raise ValueError("quality evaluation checkpoint phase disagrees with its config")
    feature_manifest = Path(config.data.feature_cache)
    if feature_manifest.is_dir():
        feature_manifest = feature_manifest / "feature_manifest.json"
    feature_payload = json.loads(feature_manifest.read_text(encoding="utf-8"))
    data_fingerprint = canonical_sha256({
        "index": file_sha256(Path(config.data.index_path)),
        "feature_cache": feature_payload["cache_identity"],
        "capabilities": file_sha256(Path(config.data.canonical_future_capabilities)),
    })
    resolved_config_sha256 = canonical_sha256(dict(resolved_config))
    expected_metadata = {
        "architecture": predictive_world_architecture_contract(config.model),
        "temporal_abi": {
            "history_visual_offsets_s": list(config.data.history_visual_offsets_s),
            **config.data.observation_contract,
            "state_history_count": config.data.state_history_count,
            "state_query_hz": config.data.state_query_hz,
            "future_group_ends_s": list(config.data.future_group_ends_s),
            "action_horizon": config.data.action_horizon,
            "action_hz": config.data.action_hz,
        },
        "encoder_id": str(feature_payload["provenance"]["encoder_id"]),
        "processor_id": str(feature_payload["provenance"]["processor_id"]),
        "protocol_registry_sha256": file_sha256(Path(config.data.action_protocol_registry)),
        "measurement_registry_sha256": file_sha256(Path(config.data.measurement_registry)),
        "data_fingerprint": data_fingerprint,
        "resolved_config_sha256": resolved_config_sha256,
        "code_revision": str(code_revision),
        "code_fingerprint": str(code_fingerprint),
    }
    metadata = sidecar["metadata"]
    mismatches = {
        name: {"checkpoint": metadata.get(name), "evaluation": expected}
        for name, expected in expected_metadata.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"evaluation checkpoint/config/data/code mismatch: {mismatches}")
    binding = {
        "checkpoint_identity": predictive_world_bundle_identity(checkpoint),
        "checkpoint_weights_sha256": sidecar["weights_sha256"],
        "checkpoint_completed_updates": int(sidecar["completed_updates"]),
        "resolved_config_sha256": resolved_config_sha256,
        "config_file_sha256": file_sha256(Path(config_path)),
        "data_fingerprint": data_fingerprint,
        "code_revision": str(code_revision),
        "code_fingerprint": str(code_fingerprint),
    }
    if evaluation_release is not None:
        if evaluation_release.get("format") != EVALUATION_RELEASE_FORMAT:
            raise ValueError("evaluation release binding has an unsupported format")
        if evaluation_release.get("index_sha256") != file_sha256(Path(config.data.index_path)):
            raise ValueError("evaluation release does not belong to the configured index")
        binding["evaluation_release"] = dict(evaluation_release)
    return binding


def validate_record_bindings(
    records: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> None:
    if not records:
        raise ValueError("evaluation records cannot be empty")
    for index, record in enumerate(records):
        binding = record.get("evaluation_binding")
        if not isinstance(binding, Mapping) or dict(binding) != dict(expected):
            raise ValueError(
                f"evaluation record {index} is unbound or stale for the requested bundle"
            )


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        view = str(record["view"])
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("each evaluation record needs non-empty scalar metrics")
        groups = [view]
        if record.get("source_id") is not None:
            groups.append(f"{view}|source={record['source_id']}")
        if record.get("motion_bucket") is not None:
            groups.append(f"{view}|motion={record['motion_bucket']}")
        for name, raw in metrics.items():
            value = float(raw)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite evaluation metric {view}/{name}")
            for group in groups:
                values[group][str(name)].append(value)
    result = {}
    for view, metrics in values.items():
        result[view] = {}
        for name, rows in metrics.items():
            array = np.asarray(rows, dtype=np.float64)
            result[view][f"{name}.mean"] = float(array.mean())
            result[view][f"{name}.p50"] = float(np.quantile(array, 0.50))
            result[view][f"{name}.p95"] = float(np.quantile(array, 0.95))
            result[view][f"{name}.count"] = float(len(array))
            standard_error = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
            result[view][f"{name}.ci95_low"] = float(array.mean() - 1.96 * standard_error)
            result[view][f"{name}.ci95_high"] = float(array.mean() + 1.96 * standard_error)
    return result


def evaluate_gates(
    metrics: Mapping[str, Mapping[str, float]],
    gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for gate in gates:
        view, name = str(gate["view"]), str(gate["metric"])
        if view not in metrics or name not in metrics[view]:
            raise KeyError(f"quality gate metric is missing: {view}/{name}")
        value = float(metrics[view][name])
        minimum = gate.get("min")
        maximum = gate.get("max")
        if (minimum is None) == (maximum is None):
            raise ValueError("each quality gate must declare exactly one of min/max")
        passed = value >= float(minimum) if minimum is not None else value <= float(maximum)
        results.append({**dict(gate), "value": value, "passed": passed})
    return results


def validate_passed_evaluation_report(
    report: Mapping[str, Any], expected_binding: Mapping[str, Any]
) -> None:
    """Fail closed before export; this validates records, not model inference."""

    if report.get("format") != EVALUATION_FORMAT:
        raise ValueError("evaluation report format is unsupported")
    if report.get("evaluation_binding") != dict(expected_binding):
        raise ValueError("evaluation report does not bind the exact policy bundle/config/data/code")
    release = expected_binding.get("evaluation_release")
    if (
        not isinstance(release, Mapping)
        or release.get("format") != EVALUATION_RELEASE_FORMAT
    ):
        raise ValueError("evaluation report is not bound to an immutable held-out release")
    if report.get("required_views") != sorted(REQUIRED_VIEWS):
        raise ValueError("evaluation report does not enforce every required view")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or not REQUIRED_VIEWS.issubset(metrics):
        raise ValueError("evaluation report is missing required-view metrics")
    outcomes = report.get("gates")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("evaluation report has no quality gates")
    gate_views = {
        str(outcome.get("view"))
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    }
    if not REQUIRED_VIEWS.issubset(gate_views):
        raise ValueError("evaluation report quality gates do not cover every required view")
    gate_specs = [
        {key: value for key, value in outcome.items() if key not in {"value", "passed"}}
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    ]
    if len(gate_specs) != len(outcomes):
        raise ValueError("evaluation report contains malformed quality gates")
    recomputed = evaluate_gates(metrics, gate_specs)
    if recomputed != outcomes:
        raise ValueError("evaluation gate outcomes disagree with the reported metrics")
    if report.get("passed") is not True or not all(item["passed"] for item in outcomes):
        raise ValueError("evaluation report did not pass every quality gate")
    sample_count = report.get("sample_count")
    record_count = report.get("record_count")
    if (
        not isinstance(sample_count, int)
        or sample_count <= 0
        or release.get("sample_count") != sample_count
        or record_count != sample_count * len(REQUIRED_VIEWS)
    ):
        raise ValueError("evaluation report sample/required-view record counts are inconsistent")


__all__ = [
    "EVALUATION_FORMAT",
    "EVALUATION_RELEASE_FORMAT",
    "REQUIRED_VIEWS",
    "aggregate_metrics",
    "checkpoint_evaluation_binding",
    "evaluate_gates",
    "evaluation_release_from_index",
    "fixed_sample",
    "validate_passed_evaluation_report",
    "validate_release_record_ids",
    "validate_record_bindings",
]
