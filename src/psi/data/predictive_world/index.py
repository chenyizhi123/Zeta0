"""Immutable physical-time index construction for predictive-world windows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from psi.data.predictive_world.alignment import causal_floor_indices, select_predictive_window
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.measurement_schema import (
    MeasurementSchema,
    SourceCapability,
    validate_predictive_world_row_capability,
)


INDEX_FORMAT = "zeta0.predictive_world.index.v2"
LOOKUP_FORMAT = "zeta0.predictive_world.lookup.v3"
TEMPORAL_ABI_FORMAT = "zeta0.predictive_world.temporal_abi.v2"
INDEX_BINDING_FORMAT = "zeta0.predictive_world.index_binding.v1"


def predictive_world_temporal_abi(
    *,
    history_visual_offsets_s: Sequence[float],
    state_history_count: int,
    state_query_hz: float,
    future_group_ends_s: Sequence[float],
    action_horizon: int,
    action_hz: float,
    visual_max_age_s: float,
    state_max_age_s: float,
    action_max_age_s: float,
    future_measurement_max_age_s: float | None = None,
    future_visual_max_age_s: float | None = None,
    real_observation_mode: str | None = None,
    real_observation_contract: Mapping[str, Any] | None = None,
    real_snapshot_max_age_s: float | None = None,
    real_action_label_mode: str | None = None,
    real_action_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize every physical-time selection decision made by the index."""

    if (
        isinstance(state_history_count, bool)
        or isinstance(action_horizon, bool)
        or int(state_history_count) != state_history_count
        or int(action_horizon) != action_horizon
    ):
        raise ValueError("temporal ABI history/action counts must be integers")
    history = tuple(float(value) for value in history_visual_offsets_s)
    groups = tuple(float(value) for value in future_group_ends_s)
    state_count = int(state_history_count)
    state_hz = float(state_query_hz)
    action_count = int(action_horizon)
    native_action_hz = float(action_hz)
    ages = tuple(float(value) for value in (
        visual_max_age_s, state_max_age_s, action_max_age_s
    ))
    if (
        not history
        or history[-1] != 0.0
        or tuple(sorted(history)) != history
        or any(value > 0 or not np.isfinite(value) for value in history)
    ):
        raise ValueError("temporal ABI history visual offsets must be sorted, finite and end at zero")
    if (
        not groups
        or tuple(sorted(set(groups))) != groups
        or any(value <= 0 or not np.isfinite(value) for value in groups)
    ):
        raise ValueError("temporal ABI future group ends must be finite and strictly increasing")
    if (
        state_count <= 0
        or action_count <= 0
        or state_hz <= 0
        or native_action_hz <= 0
        or not np.isfinite(state_hz)
        or not np.isfinite(native_action_hz)
        or any(value < 0 or not np.isfinite(value) for value in ages)
    ):
        raise ValueError("temporal ABI rates, counts and age tolerances are invalid")
    state_offsets = np.linspace(-(state_count - 1) / state_hz, 0.0, state_count)
    # Actions supervise/condition the interval strictly after the observation
    # anchor.  Starting at zero made an asynchronous camera anchor select the
    # last command from before t0, invalidating every action-conditioned group.
    action_offsets = np.arange(1, action_count + 1, dtype=np.float64) / native_action_hz
    measurement_count = int(round(groups[-1] * state_hz))
    if measurement_count <= 0 or not np.isclose(
        measurement_count / state_hz, groups[-1], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "temporal ABI final future group must lie exactly on the state query grid"
        )
    measurement_offsets = np.arange(1, measurement_count + 1, dtype=np.float64) / state_hz
    measurement_age = ages[1] if future_measurement_max_age_s is None else float(future_measurement_max_age_s)
    future_visual_age = ages[0] if future_visual_max_age_s is None else float(future_visual_max_age_s)
    if not np.isfinite(future_visual_age) or future_visual_age < 0:
        raise ValueError("future visual age tolerance is invalid")
    if not np.isfinite(measurement_age) or measurement_age < 0:
        raise ValueError("future measurement age tolerance is invalid")
    if real_observation_mode not in (None, "collector_snapshot_v1") or real_action_label_mode not in (None, "held_command_v1"):
        raise ValueError("unsupported real observation/action temporal contract")
    if real_observation_mode is None and (real_observation_contract is not None or real_action_label_mode is not None):
        raise ValueError("real temporal contracts require collector snapshot mode")
    if real_observation_mode is not None:
        from psi.real_observation_contract import make_deployment_observation_contract
        deployment = make_deployment_observation_contract(dict(
            real_observation_mode=real_observation_mode, real_observation_contract=real_observation_contract,
            state_history_count=state_count, state_max_age_s=ages[1], visual_max_age_s=ages[0],
            real_snapshot_max_age_s=real_snapshot_max_age_s))
        real_observation_contract = deployment["collector"]
        if real_action_label_mode != "held_command_v1":
            raise ValueError("collector action slots require held commands")
    elif real_action_projection is not None:
        raise ValueError("recorded action projection requires collector snapshot mode")
    if real_snapshot_max_age_s is not None and real_observation_mode is None:
        raise ValueError("snapshot age limit requires collector snapshot mode")
    result = {
        "format": TEMPORAL_ABI_FORMAT,
        "history_visual_offsets_s": list(history),
        "state_history_count": state_count,
        "state_query_hz": state_hz,
        "state_offsets_s": state_offsets.tolist(),
        "future_group_ends_s": list(groups),
        "future_measurement_offsets_s": measurement_offsets.tolist(),
        "action_horizon": action_count,
        "action_hz": native_action_hz,
        "action_offsets_s": action_offsets.tolist(),
        "selection": {
            "history_visual": {"method": "causal_floor", "max_age_s": ages[0]},
            "future_visual": {
                "method": "causal_floor",
                "max_age_s": future_visual_age,
                "strictly_after_anchor": True,
            },
            "state_history": {"method": "causal_floor", "max_age_s": ages[1]},
            "future_measurement": {
                "method": "causal_floor",
                "max_age_s": measurement_age,
                "strictly_after_anchor": True,
            },
            "demonstrated_action": {
                "method": "causal_floor",
                "max_age_s": ages[2],
                "strictly_after_anchor": real_action_label_mode != "held_command_v1",
            },
        },
    }
    if future_measurement_max_age_s is not None:
        result["future_measurement_max_age_s"] = measurement_age
    if future_visual_max_age_s is not None:
        result["future_visual_max_age_s"] = future_visual_age
    if real_observation_mode is not None:
        result.update(real_observation_mode=real_observation_mode,
                      real_observation_contract=dict(real_observation_contract or {}),
                      real_action_label_mode=real_action_label_mode)
    if real_action_projection is not None:
        from psi.real_observation_contract import validate_action_projection
        result["real_action_projection"] = validate_action_projection(real_action_projection)
    if real_snapshot_max_age_s is not None:
        result["real_snapshot_max_age_s"] = real_snapshot_max_age_s
    return result


def temporal_abi_from_data_config(data: Any) -> dict[str, Any]:
    """Build the expected immutable index ABI from a typed training data config."""

    return predictive_world_temporal_abi(
        history_visual_offsets_s=data.history_visual_offsets_s,
        state_history_count=data.state_history_count,
        state_query_hz=data.state_query_hz,
        future_group_ends_s=data.future_group_ends_s,
        action_horizon=data.action_horizon,
        action_hz=data.action_hz,
        visual_max_age_s=data.visual_max_age_s,
        state_max_age_s=data.state_max_age_s,
        action_max_age_s=data.action_max_age_s,
        future_measurement_max_age_s=getattr(data, "future_measurement_max_age_s", None),
        future_visual_max_age_s=getattr(data, "future_visual_max_age_s", None),
        real_observation_mode=getattr(data, "real_observation_mode", None),
        real_observation_contract=getattr(data, "real_observation_contract", None),
        real_snapshot_max_age_s=getattr(data, "real_snapshot_max_age_s", None),
        real_action_label_mode=getattr(data, "real_action_label_mode", None),
        real_action_projection=getattr(data, "real_action_projection", None),
    )


def _validated_temporal_abi(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        selection = value["selection"]
        expected = predictive_world_temporal_abi(
            history_visual_offsets_s=value["history_visual_offsets_s"],
            state_history_count=int(value["state_history_count"]),
            state_query_hz=float(value["state_query_hz"]),
            future_group_ends_s=value["future_group_ends_s"],
            action_horizon=int(value["action_horizon"]),
            action_hz=float(value["action_hz"]),
            visual_max_age_s=float(selection["history_visual"]["max_age_s"]),
            state_max_age_s=float(selection["state_history"]["max_age_s"]),
            action_max_age_s=float(selection["demonstrated_action"]["max_age_s"]),
            future_measurement_max_age_s=value.get("future_measurement_max_age_s"),
            future_visual_max_age_s=value.get("future_visual_max_age_s"),
            real_observation_mode=value.get("real_observation_mode"),
            real_observation_contract=value.get("real_observation_contract"),
            real_snapshot_max_age_s=value.get("real_snapshot_max_age_s"),
            real_action_label_mode=value.get("real_action_label_mode"),
            real_action_projection=value.get("real_action_projection"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("index temporal ABI is incomplete") from error
    if dict(value) != expected:
        raise ValueError("index temporal ABI contains inconsistent derived/selection fields")
    return expected


def _different_paths(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        paths: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual or key not in expected:
                paths.append(child)
            else:
                paths.extend(_different_paths(actual[key], expected[key], child))
        return paths
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [prefix]
        paths = []
        for index, (left, right) in enumerate(zip(actual, expected)):
            paths.extend(_different_paths(left, right, f"{prefix}[{index}]"))
        return paths
    return [] if actual == expected else [prefix]


def _validate_row_temporal_shape(row: Mapping[str, Any], temporal: Mapping[str, Any]) -> None:
    from psi.data.predictive_world.action_labels import validate_action_component_index
    validate_action_component_index(row, int(temporal["action_horizon"]))
    expected_lengths = {
        "history_visual_indices": len(temporal["history_visual_offsets_s"]),
        "history_visual_valid": len(temporal["history_visual_offsets_s"]),
        "history_visual_actual_times": len(temporal["history_visual_offsets_s"]),
        "state_indices": int(temporal["state_history_count"]),
        "state_valid": int(temporal["state_history_count"]),
        "state_actual_times": int(temporal["state_history_count"]),
        "future_visual_indices": len(temporal["future_group_ends_s"]),
        "future_visual_valid": len(temporal["future_group_ends_s"]),
        "future_visual_actual_times": len(temporal["future_group_ends_s"]),
        "world_action_group_valid": len(temporal["future_group_ends_s"]),
        "future_measurement_indices": len(temporal["future_measurement_offsets_s"]),
        "future_measurement_valid": len(temporal["future_measurement_offsets_s"]),
        "future_measurement_actual_times": len(temporal["future_measurement_offsets_s"]),
        "action_indices": int(temporal["action_horizon"]),
        "action_valid": int(temporal["action_horizon"]),
        "action_actual_times": int(temporal["action_horizon"]),
    }
    for field, expected_length in expected_lengths.items():
        value = row.get(field)
        if not isinstance(value, list) or len(value) != expected_length:
            raise ValueError(
                f"predictive-world row {field} length disagrees with temporal ABI"
            )
    camera_fields = (
        "history_visual_indices_by_camera",
        "history_visual_valid_by_camera",
        "history_visual_actual_times_by_camera",
    )
    present = [field in row for field in camera_fields]
    if any(present) and not all(present):
        raise ValueError("per-camera history alignment publication is incomplete")
    if all(present):
        cameras = set(map(str, row["source_capability"].get("visual_keys", ())))
        expected = len(temporal["history_visual_offsets_s"])
        for field in camera_fields:
            value = row[field]
            if not isinstance(value, Mapping) or set(map(str, value)) != cameras:
                raise ValueError(f"predictive-world row {field} camera set disagrees")
            if any(not isinstance(items, list) or len(items) != expected for items in value.values()):
                raise ValueError(f"predictive-world row {field} length disagrees")


def predictive_world_eligibility(row: Mapping[str, Any]) -> dict[str, bool]:
    """Derive phase eligibility from masks gated by an immutable audit snapshot."""

    history = tuple(bool(value) for value in row.get("history_visual_valid", ()))
    future = tuple(bool(value) for value in row.get("future_visual_valid", ()))
    measurements = tuple(bool(value) for value in row.get("future_measurement_valid", ()))
    actions = tuple(bool(value) for value in row.get("action_valid", ()))
    world = tuple(bool(value) for value in row.get("world_action_group_valid", ()))
    capability = row.get("source_capability")
    if not isinstance(capability, Mapping):
        capability = {}
    supports_visual = bool(capability.get("supports_future_visual", False))
    supports_measurement = bool(capability.get("supports_future_measurement", False))
    supports_world = bool(capability.get("supports_world_action", False))
    current_valid = bool(history) and history[-1]
    prior = supports_visual and current_valid and bool(future) and any(future)
    world_valid = (
        supports_world
        and
        prior
        and len(world) == len(future)
        and any(image_valid and action_valid for image_valid, action_valid in zip(future, world))
    )
    policy = (
        current_valid
        and supports_visual
        and supports_measurement
        and bool(future)
        and all(future)
        and bool(measurements)
        and all(measurements)
        and bool(actions)
        and any(actions)
    )
    return {"prior": prior, "world": world_valid, "policy": policy}


def predictive_world_lookup_path(path: str | Path) -> Path:
    index = Path(path).expanduser().resolve()
    return index.with_suffix(index.suffix + ".sqlite3")


def _publish_lookup(path: Path, index_path: Path) -> None:
    """Publish a compact lookup without repeating each payload for every view."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE records (
                ordinal INTEGER PRIMARY KEY,
                sample_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            );
            CREATE TABLE occurrences (
                split TEXT NOT NULL,
                view TEXT NOT NULL,
                position INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                source_position INTEGER NOT NULL,
                record_ordinal INTEGER NOT NULL,
                PRIMARY KEY (split, view, position)
            ) WITHOUT ROWID;
            CREATE UNIQUE INDEX occurrence_source_position
                ON occurrences(split, view, source_id, source_position);
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", ("format", LOOKUP_FORMAT)
        )
        positions: dict[tuple[str, str], int] = {}
        source_positions: dict[tuple[str, str, str], int] = {}
        with index_path.open("r", encoding="utf-8") as handle:
            records = (json.loads(line) for line in handle if line.strip())
            for ordinal, record in enumerate(records):
                payload = json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                connection.execute(
                    "INSERT INTO records(ordinal, sample_id, payload) VALUES (?, ?, ?)",
                    (ordinal, str(record["sample_id"]), payload),
                )
                split = str(record["split"])
                source = str(record["source_id"])
                eligibility = predictive_world_eligibility(record)
                for view, eligible in eligibility.items():
                    if not eligible:
                        continue
                    key = (split, view)
                    source_key = (split, view, source)
                    position = positions.get(key, 0)
                    source_position = source_positions.get(source_key, 0)
                    connection.execute(
                        "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?)",
                        (split, view, position, source, source_position, ordinal),
                    )
                    positions[key] = position + 1
                    source_positions[source_key] = source_position + 1
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


def build_episode_windows(
    episode: Mapping[str, Any],
    *,
    history_visual_offsets_s: Sequence[float],
    state_offsets_s: Sequence[float],
    future_group_ends_s: Sequence[float],
    action_offsets_s: Sequence[float],
    visual_max_age_s: float,
    state_max_age_s: float,
    action_max_age_s: float,
    capability: SourceCapability,
    measurement_schema: MeasurementSchema,
    future_measurement_offsets_s: Sequence[float] | None = None,
    frame_grid_hz: float | None = None,
    future_measurement_max_age_s: float | None = None,
    future_visual_max_age_s: float | None = None,
) -> list[dict[str, Any]]:
    """Build rows from one episode; all returned indices address that episode only."""

    required = {"source_id", "episode_id", "split", "anchor_times", "visual_timestamps", "state_timestamps", "action_timestamps"}
    missing = sorted(required - episode.keys())
    if missing:
        raise ValueError(f"episode record is missing keys: {missing}")
    if str(episode["source_id"]) != capability.source_id:
        raise ValueError("episode source_id disagrees with the supplied capability")
    if measurement_schema.schema_id != capability.measurement_schema_id:
        raise ValueError("measurement schema disagrees with the supplied capability")
    capability_contract = capability.index_contract()
    visual = np.asarray(episode["visual_timestamps"], dtype=np.float64)
    state = np.asarray(episode["state_timestamps"], dtype=np.float64)
    actions = np.asarray(episode["action_timestamps"], dtype=np.float64)
    measurement_source = np.asarray(episode.get("future_measurement_timestamps", state), dtype=np.float64)
    measurement_source_rows = np.asarray(
        episode.get("future_measurement_row_indices", episode.get("state_row_indices", np.arange(len(state)))),
        dtype=np.int64,
    )
    if measurement_source_rows.shape != measurement_source.shape or (measurement_source_rows < 0).any():
        raise ValueError("future measurement stream and source row indices disagree")
    visual_rows = np.asarray(
        episode.get("visual_row_indices", np.arange(len(visual))), dtype=np.int64
    )
    state_rows = np.asarray(
        episode.get("state_row_indices", np.arange(len(state))), dtype=np.int64
    )
    action_rows = np.asarray(
        episode.get("action_row_indices", np.arange(len(actions))), dtype=np.int64
    )
    if (
        visual_rows.shape != visual.shape
        or state_rows.shape != state.shape
        or action_rows.shape != actions.shape
        or (visual_rows < 0).any()
        or (state_rows < 0).any()
        or (action_rows < 0).any()
    ):
        raise ValueError("timestamp row-index mappings must be aligned and non-negative")
    anchors = np.asarray(episode["anchor_times"], dtype=np.float64)
    if anchors.ndim != 1 or not np.isfinite(anchors).all() or len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchor_times must be unique and finite")
    def relative_times(values, anchor):
        delta = values - anchor
        if frame_grid_hz is not None:
            # Publish exact grid-relative seconds as well as exact row indices;
            # cancellation must not move a command across a group-end boundary.
            delta = np.rint(delta * frame_grid_hz) / frame_grid_hz
        return delta.tolist()
    output = []
    alignment_mode = capability_contract["visual_alignment_mode"]
    visual_streams = episode.get("visual_streams")
    independent_camera = alignment_mode in {"independent_camera_physical_time", "independent_camera_nominal_time"}
    if independent_camera:
        if not isinstance(visual_streams, Mapping) or set(map(str, visual_streams)) != set(
            capability.visual_keys
        ):
            raise ValueError(
                "independent multi-camera source requires every audited visual_stream"
            )
    for anchor_ordinal, anchor in enumerate(anchors.tolist()):
        window = select_predictive_window(
            anchor_time=anchor,
            visual_timestamps=visual,
            state_timestamps=state,
            history_visual_offsets_s=history_visual_offsets_s,
            state_offsets_s=state_offsets_s,
            future_group_ends_s=future_group_ends_s,
            visual_max_age_s=visual_max_age_s,
            state_max_age_s=state_max_age_s,
            frame_grid_hz=frame_grid_hz,
        )
        action_indices, action_valid, action_actual = causal_floor_indices(
            actions,
            anchor + np.asarray(action_offsets_s, dtype=np.float64),
            max_age_seconds=action_max_age_s,
            frame_grid_hz=frame_grid_hz,
        )
        # Demonstrations condition/target the open future interval (t0, tN].
        # A floor lookup may otherwise pull a command from before the anchor.
        action_valid &= action_actual > anchor + 1e-12
        action_offsets_array = np.asarray(action_offsets_s, dtype=np.float64)
        future_visual_valid = window.future_visual_valid & capability.supports_future_visual
        if future_visual_max_age_s is not None:
            future_visual_valid &= np.asarray(future_group_ends_s) + anchor - window.future_visual_times <= future_visual_max_age_s + 1e-12
        world_action_group_valid = [
            capability.supports_world_action
            and bool(action_valid[action_offsets_array <= end + 1e-12].all())
            and bool((action_offsets_array <= end + 1e-12).any())
            for end in future_group_ends_s
        ]
        measurement_offsets = (
            np.asarray(future_group_ends_s, dtype=np.float64)
            if future_measurement_offsets_s is None
            else np.asarray(future_measurement_offsets_s, dtype=np.float64)
        )
        measurement_indices, measurement_valid, measurement_actual = causal_floor_indices(
            measurement_source, anchor + measurement_offsets,
            max_age_seconds=state_max_age_s if future_measurement_max_age_s is None else future_measurement_max_age_s,
            frame_grid_hz=frame_grid_hz,
        )
        measurement_valid &= measurement_actual > anchor
        measurement_valid &= capability.supports_future_measurement
        history_indices_by_camera: dict[str, list[int]] = {}
        history_valid_by_camera: dict[str, list[bool]] = {}
        history_times_by_camera: dict[str, list[float]] = {}
        if independent_camera:
            assert isinstance(visual_streams, Mapping)
            for camera_id in capability.visual_keys:
                stream = visual_streams[camera_id]
                if not isinstance(stream, Mapping):
                    raise ValueError("visual_stream entry must be an object")
                timestamps = np.asarray(stream.get("timestamps"), dtype=np.float64)
                rows = np.asarray(
                    stream.get("row_indices", np.arange(len(timestamps))), dtype=np.int64
                )
                if rows.shape != timestamps.shape or (rows < 0).any():
                    raise ValueError("visual_stream row mapping is invalid")
                indices, valid, actual = causal_floor_indices(
                    timestamps,
                    anchor + np.asarray(history_visual_offsets_s, dtype=np.float64),
                    max_age_seconds=visual_max_age_s,
                    frame_grid_hz=frame_grid_hz,
                )
                history_indices_by_camera[camera_id] = rows[indices].tolist()
                history_valid_by_camera[camera_id] = valid.tolist()
                history_times_by_camera[camera_id] = relative_times(actual, anchor)
        else:
            for camera_id in capability.visual_keys:
                history_indices_by_camera[camera_id] = visual_rows[
                    window.history_visual_indices
                ].tolist()
                history_valid_by_camera[camera_id] = window.history_visual_valid.tolist()
                history_times_by_camera[camera_id] = relative_times(window.history_visual_times, anchor)
        identity = {
            "source_id": str(episode["source_id"]),
            "episode_id": str(episode["episode_id"]),
            "anchor_time": float(anchor),
            "anchor_ordinal": anchor_ordinal,
        }
        sample_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record = {
            **identity,
            "sample_id": sample_id,
            "split": str(episode["split"]),
            "history_visual_indices": visual_rows[window.history_visual_indices].tolist(),
            "history_visual_valid": window.history_visual_valid.tolist(),
            "history_visual_actual_times": relative_times(window.history_visual_times, anchor),
            "history_visual_indices_by_camera": history_indices_by_camera,
            "history_visual_valid_by_camera": history_valid_by_camera,
            "history_visual_actual_times_by_camera": history_times_by_camera,
            "state_indices": state_rows[window.state_indices].tolist(),
            "state_valid": window.state_valid.tolist(),
            "state_actual_times": relative_times(window.state_times, anchor),
            "future_visual_indices": visual_rows[window.future_visual_indices].tolist(),
            "future_visual_valid": future_visual_valid.tolist(),
            "future_visual_actual_times": relative_times(window.future_visual_times, anchor),
            "action_indices": action_rows[action_indices].tolist(),
            "action_valid": action_valid.tolist(),
            "action_actual_times": relative_times(action_actual, anchor),
            "world_action_group_valid": world_action_group_valid,
            "future_measurement_indices": measurement_source_rows[measurement_indices].tolist(),
            "future_measurement_valid": measurement_valid.tolist(),
            "future_measurement_actual_times": relative_times(measurement_actual, anchor),
            "materialization": dict(episode.get("materialization", {})),
            "source_capability": capability_contract,
        }
        validate_predictive_world_row_capability(
            record,
            capabilities={capability.source_id: capability},
            measurement_schemas={measurement_schema.schema_id: measurement_schema},
        )
        record["eligibility"] = predictive_world_eligibility(record)
        output.append(record)
    return output


def publish_predictive_world_index(
    output_path: str | Path,
    *,
    rows: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    temporal_abi: Mapping[str, Any],
    excluded_episode_ids: set[str] | frozenset[str] = frozenset(),
    presorted: bool = False,
) -> tuple[Path, Path]:
    path = Path(output_path).expanduser().resolve()
    if not provenance or any("REQUIRED_" in str(value) for value in provenance.values()):
        raise ValueError("index provenance is incomplete")
    registry_binding = provenance.get("registry_binding")
    required_registry_hashes = {
        "measurement_registry_sha256",
        "capability_registry_sha256",
        "action_protocol_registry_sha256",
        "state_schema_registry_sha256",
    }
    if (
        not isinstance(registry_binding, Mapping)
        or set(registry_binding) != required_registry_hashes
        or any(
            len(str(registry_binding[name])) != 64
            or any(character not in "0123456789abcdef" for character in str(registry_binding[name]))
            for name in required_registry_hashes
        )
    ):
        raise ValueError("index provenance requires exact registry SHA-256 bindings")
    temporal = _validated_temporal_abi(temporal_abi)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    lookup_path = predictive_world_lookup_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    record_iterable: Iterable[Mapping[str, Any]]
    if presorted:
        record_iterable = rows
    else:
        record_iterable = sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                str(row["source_id"]), str(row["episode_id"]), float(row["anchor_time"])
            ),
        )
    count = 0
    previous_sort_key: tuple[str, str, float] | None = None
    allowed_splits = {"train", "train_calibration", "validation", "test"}
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for ordinal, source_row in enumerate(record_iterable):
                row = dict(source_row)
                sort_key = (
                    str(row["source_id"]),
                    str(row["episode_id"]),
                    float(row["anchor_time"]),
                )
                if presorted and previous_sort_key is not None and sort_key < previous_sort_key:
                    raise ValueError("presorted predictive-world rows are out of order")
                previous_sort_key = sort_key
                if str(row.get("split")) not in allowed_splits:
                    raise ValueError(
                        f"predictive-world index has unsupported split: {row.get('split')}"
                    )
                if str(row["episode_id"]) in excluded_episode_ids:
                    raise ValueError(
                        f"predictive-world index overlaps downstream episode: {row['episode_id']}"
                    )
                expected = predictive_world_eligibility(row)
                if row.get("eligibility") not in (None, expected):
                    raise ValueError(
                        "predictive-world row eligibility disagrees with alignment masks"
                    )
                row["eligibility"] = expected
                row["feature_row_index"] = ordinal
                _validate_row_temporal_shape(row, temporal)
                handle.write(
                    json.dumps(
                        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    )
                    + "\n"
                )
                count += 1
        if count == 0:
            raise ValueError("predictive-world index cannot be empty")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    meta = {
        "format": INDEX_FORMAT,
        "rows": count,
        "provenance": dict(provenance),
        "temporal_abi": temporal,
        "semantic_identity": canonical_sha256({
            "provenance": dict(provenance), "temporal_abi": temporal,
        }),
    }
    if path.exists() or meta_path.exists() or lookup_path.exists():
        if not path.is_file() or not meta_path.is_file() or not lookup_path.is_file():
            temporary.unlink(missing_ok=True)
            raise ValueError("partial predictive-world index publication exists")
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        candidate_sha = file_sha256(temporary)
        if candidate_sha != file_sha256(path) or existing_meta != {
            **meta,
            "index_sha256": file_sha256(path),
            "lookup_file": lookup_path.name,
            "lookup_sha256": file_sha256(lookup_path),
        }:
            temporary.unlink(missing_ok=True)
            raise ValueError("immutable predictive-world index exists with different content")
        temporary.unlink(missing_ok=True)
        return path, meta_path
    temporary.replace(path)
    try:
        _publish_lookup(lookup_path, path)
    except BaseException:
        # The index is immutable and useful for resuming lookup publication;
        # the missing sidecar makes the release fail closed.
        raise
    meta["index_sha256"] = file_sha256(path)
    meta["lookup_file"] = lookup_path.name
    meta["lookup_sha256"] = file_sha256(lookup_path)
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_tmp.replace(meta_path)
    return path, meta_path


def verify_predictive_world_index(
    path: str | Path, *, verify_content_hashes: bool = True
) -> dict[str, Any]:
    index = Path(path).expanduser().resolve()
    meta_path = index.with_suffix(index.suffix + ".meta.json")
    if not index.is_file() or not meta_path.is_file():
        raise ValueError("predictive-world index publication is incomplete")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index_digest = str(meta.get("index_sha256", ""))
    lookup_digest = str(meta.get("lookup_sha256", ""))
    if (
        meta.get("format") != INDEX_FORMAT
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in (index_digest, lookup_digest)
        )
    ):
        raise ValueError("predictive-world index format/content binding is invalid")
    if verify_content_hashes and file_sha256(index) != index_digest:
        raise ValueError("predictive-world index format/content hash mismatch")
    temporal = meta.get("temporal_abi")
    if not isinstance(temporal, Mapping):
        raise ValueError("predictive-world index lacks a supported temporal ABI")
    temporal = _validated_temporal_abi(temporal)
    expected_identity = canonical_sha256({
        "provenance": meta.get("provenance"), "temporal_abi": dict(temporal),
    })
    if meta.get("semantic_identity") != expected_identity:
        raise ValueError("predictive-world index semantic identity mismatch")
    lookup = predictive_world_lookup_path(index)
    if not lookup.is_file() or (
        verify_content_hashes and file_sha256(lookup) != lookup_digest
    ):
        raise ValueError("predictive-world lookup content hash mismatch")
    connection = sqlite3.connect(f"file:{lookup}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='format'"
        ).fetchone()
        if row is None or row[0] != LOOKUP_FORMAT:
            raise ValueError("predictive-world lookup format mismatch")
    finally:
        connection.close()
    if verify_content_hashes:
        with index.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != int(meta.get("rows", -1)):
            raise ValueError("predictive-world index row count mismatch")
    elif int(meta.get("rows", 0)) <= 0:
        raise ValueError("predictive-world index row count is invalid")
    return meta


def assert_predictive_world_index_temporal_abi(
    path: str | Path, data: Any, *, verify_content_hashes: bool = True
) -> dict[str, Any]:
    """Fail before loading data when runtime time semantics differ from the index."""

    meta = verify_predictive_world_index(
        path, verify_content_hashes=verify_content_hashes
    )
    expected = temporal_abi_from_data_config(data)
    actual = meta["temporal_abi"]
    if actual != expected:
        differing = _different_paths(actual, expected)
        raise ValueError(
            "predictive-world training config temporal ABI disagrees with index: "
            f"{differing}"
        )
    return meta


def predictive_world_index_binding(
    path: str | Path, *, verify_content_hashes: bool = True
) -> dict[str, Any]:
    meta = verify_predictive_world_index(
        path, verify_content_hashes=verify_content_hashes
    )
    return {
        "format": INDEX_BINDING_FORMAT,
        "semantic_identity": str(meta["semantic_identity"]),
        "index_sha256": str(meta["index_sha256"]),
        "lookup_sha256": str(meta["lookup_sha256"]),
        "temporal_abi_sha256": canonical_sha256(meta["temporal_abi"]),
    }


__all__ = [
    "INDEX_FORMAT",
    "INDEX_BINDING_FORMAT",
    "LOOKUP_FORMAT",
    "TEMPORAL_ABI_FORMAT",
    "assert_predictive_world_index_temporal_abi",
    "build_episode_windows",
    "publish_predictive_world_index",
    "predictive_world_eligibility",
    "predictive_world_index_binding",
    "predictive_world_lookup_path",
    "predictive_world_temporal_abi",
    "temporal_abi_from_data_config",
    "verify_predictive_world_index",
]
