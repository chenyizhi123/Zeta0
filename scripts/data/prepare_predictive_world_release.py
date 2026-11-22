#!/usr/bin/env python3
"""Publish the immutable predictive-world registries and physical-time index.

This converts the already audited HE/Psi0/HIW episode release into the
predictive-world ABI.  No raw observations are rewritten.  Every timestamp is
resolved against its native stream, episode tails are retained, and missing
future/action slots remain present with validity false.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import (
    build_episode_windows,
    predictive_world_temporal_abi,
    publish_predictive_world_index,
    verify_predictive_world_index,
)
from psi.data.predictive_world.measurement_schema import (
    load_capability_registry,
    load_measurement_registry,
)


HISTORY_VISUAL_OFFSETS = (-0.4, -0.2, 0.0)
STATE_HISTORY_COUNT = 21
STATE_HZ = 50.0
FUTURE_GROUPS = (0.2, 0.4, 0.6)
ACTION_HORIZON = 30
ACTION_HZ = 50.0
VISUAL_MAX_AGE = 0.08
STATE_MAX_AGE = 0.04
ACTION_MAX_AGE = 0.04
NORMALIZATION_SPACE = "source_native_raw_with_registry_zscore_v1"
CANONICAL_LAYOUT = "humanoid_upper_body_and_height_29_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable release artifact exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _slots() -> list[str]:
    return [
        *(f"left_arm_joint_{index}" for index in range(7)),
        *(f"right_arm_joint_{index}" for index in range(7)),
        *(f"left_hand_joint_{index}" for index in range(7)),
        *(f"right_hand_joint_{index}" for index in range(7)),
        "base_height",
    ]


def _canonical_mapping(
    raw_indices: list[int | None], statistics: Mapping[str, Any]
) -> dict[str, Any]:
    native_mean = tuple(float(value) for value in statistics["mean"])
    native_std = tuple(float(value) for value in statistics["std"])
    mean = [0.0 if index is None else native_mean[index] for index in raw_indices]
    std = [1.0 if index is None else native_std[index] for index in raw_indices]
    return {
        "layout_id": CANONICAL_LAYOUT,
        "slots": _slots(),
        "raw_indices": raw_indices,
        "mean": mean,
        "std": std,
        "normalization_space": "per_source_registry_zscore_v1",
    }


def _measurement_registries(
    state_schemas: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    g1_indices: list[int | None] = [*range(14), *range(29, 43), None]
    h1_indices: list[int | None] = [
        *range(14), *range(27, 33), None, *range(33, 39), None, None
    ]
    hiw_indices: list[int | None] = [*range(15, 29), *([None] * 15)]
    psi_indices: list[int | None] = [*range(14, 28), *range(14), 31]
    if not all(len(value) == 29 for value in (g1_indices, h1_indices, hiw_indices, psi_indices)):
        raise AssertionError("canonical body mappings must contain 29 slots")

    def field(
        name: str,
        start: int,
        stop: int,
        *,
        kind: str = "measurement",
        unit: str | None = "source_native_joint_units",
        time_key: str = "timestamp",
        statistics_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "start": start,
            "stop": stop,
            "kind": kind,
            "unit": unit,
            "coordinate_frame": "robot_native",
            "rotation_representation": None,
            "time_key": time_key,
            "statistics_id": statistics_id,
        }

    measurements = {
        "he_g1_state43": {
            "width": 43,
            "world_action_eligible": True,
            "fields": [
                field("arm_joints", 0, 14, statistics_id="g1"),
                field("leg_joints", 14, 29, statistics_id="g1"),
                field("hand_joints", 29, 43, statistics_id="g1"),
            ],
            "canonical_body": _canonical_mapping(g1_indices, state_schemas["g1"]),
        },
        "he_h1_state39": {
            "width": 39,
            "world_action_eligible": True,
            "fields": [
                field("arm_joints", 0, 14, statistics_id="h1"),
                field("leg_joints", 14, 27, statistics_id="h1"),
                field("hand_joints", 27, 39, statistics_id="h1"),
            ],
            "canonical_body": _canonical_mapping(h1_indices, state_schemas["h1"]),
        },
        "hiw_state52": {
            "width": 52,
            "world_action_eligible": True,
            "fields": [
                field(
                    "joint_observation",
                    0,
                    29,
                    time_key="timestamp.state",
                    statistics_id="hiw_g1_q29_wbc23",
                ),
                field(
                    "wbc_context",
                    29,
                    52,
                    kind="command",
                    unit=None,
                    time_key="timestamp.wbc",
                ),
            ],
            "canonical_body": _canonical_mapping(
                hiw_indices, state_schemas["hiw_g1_q29_wbc23"]
            ),
        },
        "psi0_real_state32": {
            "width": 32,
            "world_action_eligible": True,
            "fields": [
                field("hand_joints", 0, 14, statistics_id="psi0_real_g1_states32"),
                field("arm_joints", 14, 28, statistics_id="psi0_real_g1_states32"),
                field(
                    "base_rpy",
                    28,
                    31,
                    unit="rad",
                    statistics_id="psi0_real_g1_states32",
                ),
                field(
                    "base_height",
                    31,
                    32,
                    unit="source_native_length_units",
                    statistics_id="psi0_real_g1_states32",
                ),
            ],
            "canonical_body": _canonical_mapping(
                psi_indices, state_schemas["psi0_real_g1_states32"]
            ),
        },
    }
    states = {
        name: {
            "dim": int(value["dim"]),
            "mean": [float(item) for item in value["mean"]],
            "std": [float(item) for item in value["std"]],
        }
        for name, value in state_schemas.items()
        if name in {"g1", "h1", "hiw_g1_q29_wbc23", "psi0_real_g1_states32"}
    }
    return measurements, states


def _logical_source(record: Mapping[str, Any]) -> str:
    source = str(record["source_id"])
    if source == "Humanoid-Everyday":
        robot = str(record.get("robot_type", "")).lower()
        if robot not in {"g1", "h1"}:
            raise ValueError(f"unsupported Humanoid-Everyday robot type: {robot!r}")
        return f"Humanoid-Everyday/{robot}"
    return source


def _source_spec(record: Mapping[str, Any]) -> tuple[str, str, list[str], list[str]]:
    source = _logical_source(record)
    if source == "HIW-500":
        return "hiw_state52", "hiw_g1_wbc23", [
            "observation.state", "observation.state.wbc"
        ], ["action"]
    if source == "Humanoid-Everyday/g1":
        return "he_g1_state43", "he_g1_joint_target28", [
            "observation.arm_joints", "observation.leg_joints", "observation.hand_joints"
        ], ["action"]
    if source == "Humanoid-Everyday/h1":
        return "he_h1_state39", "he_h1_joint_target26", [
            "observation.arm_joints", "observation.leg_joints", "observation.hand_joints"
        ], ["action"]
    if source.startswith("Psi0-Real/"):
        return "psi0_real_state32", "psi0_real_semantic36", ["states"], ["action"]
    raise ValueError(f"unsupported pretraining source: {source!r}")


def _capabilities(
    episodes: Iterable[Mapping[str, Any]], episode_manifest_sha: str
) -> dict[str, Any]:
    representatives: dict[str, Mapping[str, Any]] = {}
    revisions: dict[str, set[str]] = {}
    for record in episodes:
        source = _logical_source(record)
        representatives.setdefault(source, record)
        revisions.setdefault(source, set()).add(str(record["revision"]))
    result = {}
    for source in sorted(representatives):
        record = representatives[source]
        schema, protocol, _, _ = _source_spec(record)
        views = [
            value for value in record.get("input_views", ()) if bool(value.get("available"))
        ]
        visual_keys = [str(value["image_key"]) for value in views]
        primary = str(record["primary_image_key"])
        if primary not in visual_keys:
            raise ValueError(f"source {source!r} primary camera is unavailable")
        independent = len(visual_keys) > 1
        audit_revision = canonical_sha256({
            "episode_manifest_sha256": episode_manifest_sha,
            "source_id": source,
            "revisions": sorted(revisions[source]),
            "measurement_schema_id": schema,
            "action_protocol_id": protocol,
            "visual_keys": visual_keys,
            "future_visual_keys": [primary],
        })
        result[source] = {
            "measurement_schema_id": schema,
            "action_protocol_id": protocol,
            "visual_keys": visual_keys,
            "future_visual_keys": [primary],
            "supports_future_visual": True,
            "supports_future_measurement": True,
            "supports_world_action": True,
            "audit_revision": audit_revision,
            "shared_frame_index_across_visual_keys": False,
            "independent_timestamps_across_visual_keys": independent,
            "visual_synchronization_audit_revision": None,
        }
    return result


def _strict_stream(values: Any, rows: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(values, dtype=np.float64).reshape(-1)
    positions = np.asarray(rows, dtype=np.int64).reshape(-1)
    if times.shape != positions.shape:
        raise ValueError("timestamp and row-index arrays disagree")
    finite = np.isfinite(times)
    times, positions = times[finite], positions[finite]
    if not len(times):
        raise ValueError("timestamp stream is empty")
    output_times: list[float] = []
    output_rows: list[int] = []
    for timestamp, row in zip(times.tolist(), positions.tolist()):
        if output_times and timestamp < output_times[-1] - 1e-9:
            raise ValueError("native timestamp stream is not monotonic")
        if output_times and math.isclose(timestamp, output_times[-1], rel_tol=0.0, abs_tol=1e-9):
            output_times[-1] = float(timestamp)
            output_rows[-1] = int(row)
        else:
            output_times.append(float(timestamp))
            output_rows.append(int(row))
    return np.asarray(output_times, dtype=np.float64), np.asarray(output_rows, dtype=np.int64)


def _column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))


def _episode_windows(
    record: Mapping[str, Any],
    *,
    capability: Any,
    measurement_schema: Any,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if bool(record.get("shared_data_file", False)):
        raise ValueError("the production converter expects episode-local parquet files")
    image_timestamp_columns = {
        str(key): str(value)
        for key, value in record.get("image_timestamp_columns", {}).items()
    }
    primary = str(record["primary_image_key"])
    visual_keys = tuple(capability.visual_keys)
    timestamp_columns = {
        image_timestamp_columns.get(key, "timestamp") for key in visual_keys
    }
    state_timestamp = str(record.get("state_timestamp_column", "timestamp"))
    action_timestamp = str(record.get("action_timestamp_column", "timestamp"))
    columns = sorted({*timestamp_columns, state_timestamp, action_timestamp, "frame_index"})
    table = pq.read_table(str(record["data_path"]), columns=columns)
    row_indices = np.arange(table.num_rows, dtype=np.int64)
    frame_indices = _column(table, "frame_index").astype(np.int64, copy=False)
    visual_streams = {}
    for camera in visual_keys:
        times, frames = _strict_stream(
            _column(table, image_timestamp_columns.get(camera, "timestamp")), frame_indices
        )
        visual_streams[camera] = {"timestamps": times, "row_indices": frames}
    visual = visual_streams[primary]
    state_times, state_rows = _strict_stream(_column(table, state_timestamp), row_indices)
    action_times, action_rows = _strict_stream(_column(table, action_timestamp), row_indices)
    nominal_fps = float(record.get("nominal_fps", 30.0))
    stride = max(1, int(round(nominal_fps / 2.0)))
    anchors = visual["timestamps"][::stride]
    schema_id, protocol_id, state_columns, action_columns = _source_spec(record)
    video_paths = {
        str(value["image_key"]): str(Path(value["video_path"]).expanduser().resolve())
        for value in record.get("input_views", ())
        if bool(value.get("available")) and str(value["image_key"]) in visual_keys
    }
    materialization = {
        "data_path": str(Path(record["data_path"]).expanduser().resolve()),
        "video_paths": video_paths,
        "primary_camera_id": primary,
        "instruction": str(record.get("instruction", "")),
        "state_columns": state_columns,
        "action_columns": action_columns,
        "measurement_schema_id": schema_id,
        "state_schema_id": {
            "he_g1_state43": "g1",
            "he_h1_state39": "h1",
            "hiw_state52": "hiw_g1_q29_wbc23",
            "psi0_real_state32": "psi0_real_g1_states32",
        }[schema_id],
        "action_protocol_id": protocol_id,
        "action_normalization_space": NORMALIZATION_SPACE,
        "measurement_indices": list(measurement_schema.canonical_body.raw_indices),
        "legacy_source_id": str(record["source_id"]),
        "legacy_episode_sample_id": str(record["sample_id"]),
    }
    episode = {
        "source_id": _logical_source(record),
        "episode_id": str(record["sample_id"]),
        "split": {
            "PT_train": "train",
            "PT_val": "validation",
            "PT_test": "test",
        }[str(record["split"])],
        "anchor_times": anchors,
        "visual_timestamps": visual["timestamps"],
        "visual_row_indices": visual["row_indices"],
        "visual_streams": visual_streams,
        "state_timestamps": state_times,
        "state_row_indices": state_rows,
        "action_timestamps": action_times,
        "action_row_indices": action_rows,
        "materialization": materialization,
    }
    return build_episode_windows(
        episode,
        history_visual_offsets_s=HISTORY_VISUAL_OFFSETS,
        state_offsets_s=tuple(
            np.linspace(-(STATE_HISTORY_COUNT - 1) / STATE_HZ, 0.0, STATE_HISTORY_COUNT)
        ),
        future_group_ends_s=FUTURE_GROUPS,
        action_offsets_s=tuple(
            np.arange(1, ACTION_HORIZON + 1, dtype=np.float64) / ACTION_HZ
        ),
        future_measurement_offsets_s=tuple(np.arange(1, 31, dtype=np.float64) / STATE_HZ),
        visual_max_age_s=VISUAL_MAX_AGE,
        state_max_age_s=STATE_MAX_AGE,
        action_max_age_s=ACTION_MAX_AGE,
        capability=capability,
        measurement_schema=measurement_schema,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress interval must be positive")
    legacy = Path(args.legacy_artifact_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    qwen = Path(args.qwen_model).expanduser().resolve()
    episode_manifest = legacy / "episode_split_manifest.jsonl"
    required = [
        episode_manifest,
        legacy / "action_statistics.json",
        legacy / "state_statistics.json",
        legacy / "anchor_index.sqlite",
        legacy / "stage_a_seed101_qk_norm_v1_runtime_v2_n5_mb256.json",
        qwen / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source release is incomplete: {missing}")
    episodes = [
        json.loads(line)
        for line in episode_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes:
        raise ValueError("source episode release is empty")
    episodes.sort(key=lambda row: (
        _logical_source(row), str(row["sample_id"])
    ))
    episode_manifest_sha = file_sha256(episode_manifest)
    state_payload = json.loads((legacy / "state_statistics.json").read_text(encoding="utf-8"))
    action_payload = json.loads((legacy / "action_statistics.json").read_text(encoding="utf-8"))
    measurements, states = _measurement_registries(state_payload["schemas"])
    protocols = {
        name: {**value, "normalization_space": NORMALIZATION_SPACE}
        for name, value in action_payload["protocols"].items()
        if name in {
            "he_g1_joint_target28",
            "he_h1_joint_target26",
            "hiw_g1_wbc23",
            "psi0_real_semantic36",
        }
    }
    capabilities_payload = _capabilities(episodes, episode_manifest_sha)
    paths = {
        "measurement": output / "measurement_registry.json",
        "capability": output / "capability_registry.json",
        "action": output / "action_protocol_registry.json",
        "state": output / "state_schema_registry.json",
    }
    _immutable_json(paths["measurement"], measurements)
    _immutable_json(paths["capability"], capabilities_payload)
    _immutable_json(paths["action"], protocols)
    _immutable_json(paths["state"], states)
    measurement_objects = load_measurement_registry(paths["measurement"])
    capability_objects = load_capability_registry(
        paths["capability"],
        measurement_schemas=measurement_objects,
        action_protocol_ids=set(protocols),
    )
    registry_binding = {
        "measurement_registry_sha256": file_sha256(paths["measurement"]),
        "capability_registry_sha256": file_sha256(paths["capability"]),
        "action_protocol_registry_sha256": file_sha256(paths["action"]),
        "state_schema_registry_sha256": file_sha256(paths["state"]),
    }
    temporal = predictive_world_temporal_abi(
        history_visual_offsets_s=HISTORY_VISUAL_OFFSETS,
        state_history_count=STATE_HISTORY_COUNT,
        state_query_hz=STATE_HZ,
        future_group_ends_s=FUTURE_GROUPS,
        action_horizon=ACTION_HORIZON,
        action_hz=ACTION_HZ,
        visual_max_age_s=VISUAL_MAX_AGE,
        state_max_age_s=STATE_MAX_AGE,
        action_max_age_s=ACTION_MAX_AGE,
    )
    index_path = output / "predictive_world_index.jsonl"
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
    lookup_path = index_path.with_suffix(index_path.suffix + ".sqlite3")
    if index_path.is_file() and meta_path.is_file() and lookup_path.is_file():
        index_meta = verify_predictive_world_index(index_path)
        if (
            index_meta.get("temporal_abi") != temporal
            or index_meta.get("provenance", {}).get("registry_binding") != registry_binding
            or index_meta.get("provenance", {}).get("source_episode_manifest_sha256")
            != episode_manifest_sha
        ):
            raise ValueError("existing predictive-world index has a different release identity")
        print(json.dumps({
            "event": "predictive_world_index_reused",
            "rows": index_meta["rows"],
        }, sort_keys=True), flush=True)
    else:
        if any(path.exists() for path in (index_path, meta_path, lookup_path)):
            raise ValueError(
                "partial predictive-world index publication exists; preserve it for diagnosis "
                "or move the release to a new output path"
            )
        episode_count = len(episodes)

        def rows() -> Iterable[dict[str, Any]]:
            emitted = 0
            for position, record in enumerate(episodes, 1):
                source = _logical_source(record)
                capability = capability_objects[source]
                schema = measurement_objects[capability.measurement_schema_id]
                values = _episode_windows(
                    record, capability=capability, measurement_schema=schema
                )
                emitted += len(values)
                yield from values
                if position % args.progress_every == 0 or position == episode_count:
                    print(json.dumps({
                        "event": "predictive_world_index_progress",
                        "episodes_done": position,
                        "episodes_total": episode_count,
                        "rows_emitted": emitted,
                    }, sort_keys=True), flush=True)

        _, published_meta_path = publish_predictive_world_index(
            index_path,
            rows=rows(),
            provenance={
                "source_episode_manifest": str(episode_manifest),
                "source_episode_manifest_sha256": episode_manifest_sha,
                "source_dataset_manifest_sha256": str(
                    action_payload["dataset_manifest_sha256"]
                ),
                "registry_binding": registry_binding,
                "anchor_selection": "native_primary_camera_every_0.5s_including_tail_v1",
            },
            temporal_abi=temporal,
            presorted=True,
        )
        # Publication computed and persisted both content hashes atomically.
        # The independent source audit below performs the full read-back check.
        index_meta = json.loads(published_meta_path.read_text(encoding="utf-8"))

    stage_config = json.loads(
        (legacy / "stage_a_seed101_qk_norm_v1_runtime_v2_n5_mb256.json").read_text(
            encoding="utf-8"
        )
    )
    configured_teacher = Path(stage_config["teacher_model"]).expanduser().resolve()
    if configured_teacher != qwen:
        raise ValueError(
            "Qwen model differs from the encoder used by the reusable legacy cache: "
            f"requested={qwen}, legacy={configured_teacher}"
        )
    processor_fingerprint = stage_config["teacher_processor_fingerprint"]
    feature_provenance = {
        "encoder_id": f"qwen3vl:{stage_config['teacher_sha256']}",
        "processor_id": f"qwen3vl-processor:{canonical_sha256(processor_fingerprint)}",
        "image_preprocessing": "aspect_v1",
        "feature_layer": "post_merger_content_area_overlap_4x4_v1",
        "dtype": "float16",
        "feature_shape": [16, 2048],
        "source_episode_manifest_sha256": episode_manifest_sha,
        "legacy_teacher_sha256": str(stage_config["teacher_sha256"]),
        "legacy_endpoint_group": 2,
        "legacy_endpoint_seconds": 0.6,
    }
    _immutable_json(output / "feature_provenance.json", feature_provenance)
    audit_path = output / "source_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_audit_binding = {
            "measurement_registry_sha256": registry_binding["measurement_registry_sha256"],
            "capability_registry_sha256": registry_binding["capability_registry_sha256"],
            "action_protocol_registry_sha256": registry_binding[
                "action_protocol_registry_sha256"
            ],
            "state_schema_registry_sha256": registry_binding["state_schema_registry_sha256"],
        }
        if (
            any(audit.get(key) != value for key, value in expected_audit_binding.items())
            or audit.get("index_identity") != index_meta["semantic_identity"]
        ):
            raise ValueError("existing source audit is bound to a different release")
        print(json.dumps({
            "event": "predictive_world_source_audit_reused",
            "rows": audit.get("rows"),
        }, sort_keys=True), flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("audit_predictive_world_sources.py")),
                "--measurement-registry", str(paths["measurement"]),
                "--capability-registry", str(paths["capability"]),
                "--action-protocol-registry", str(paths["action"]),
                "--state-schema-registry", str(paths["state"]),
                "--index", str(index_path),
                "--output", str(audit_path),
            ],
            check=True,
        )
    print(json.dumps({
        "event": "predictive_world_release_metadata_ready",
        "output": str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
