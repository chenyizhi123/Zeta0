"""LeRobot v2.x episode and physical-time anchor indexing for Stage A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .alignment import align_visual_anchors, causal_state_history_indices


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_lerobot_metadata(root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes_path = root / "meta/episodes.jsonl"
    if episodes_path.is_file():
        episodes = _jsonl(episodes_path)
    else:
        try:
            import pyarrow.dataset as ds
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("LeRobot v3 metadata requires pyarrow") from error
        episode_root = root / "meta/episodes"
        if not episode_root.is_dir():
            raise FileNotFoundError("missing meta/episodes.jsonl and meta/episodes/")
        episodes = ds.dataset(episode_root, format="parquet").to_table().to_pylist()
    tasks_path = root / "meta/tasks.jsonl"
    tasks = {}
    if tasks_path.is_file():
        for row in _jsonl(tasks_path):
            tasks[int(row["task_index"])] = str(row.get("description") or row.get("task") or "").strip()
    elif (root / "meta/tasks.parquet").is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("LeRobot v3 task metadata requires pyarrow") from error
        for row in pq.read_table(root / "meta/tasks.parquet").to_pylist():
            tasks[int(row["task_index"])] = str(row.get("description") or row.get("task") or "").strip()
    return info, episodes, tasks


def resolve_lerobot_path(
    root: str | Path,
    template: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
    chunk_index: int | None = None,
    file_index: int | None = None,
) -> Path:
    values = {
        "episode_index": int(episode_index),
        "episode_chunk": int(episode_index) // int(chunks_size),
        "video_key": video_key,
        "chunk_index": int(chunk_index) if chunk_index is not None else int(episode_index) // int(chunks_size),
        "file_index": int(file_index) if file_index is not None else int(episode_index) % int(chunks_size),
    }
    return Path(root) / template.format(**values)


def _episode_instruction(episode: Mapping[str, Any], tasks: Mapping[int, str]) -> str:
    raw = episode.get("tasks", episode.get("task_indices", []))
    if isinstance(raw, (str, int)):
        raw = [raw]
    descriptions = []
    for value in raw or []:
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            descriptions.append(tasks.get(int(value), str(value)))
        else:
            descriptions.append(str(value))
    # Episode-level descriptions are safe current context.  Timestamped future
    # completion annotations are deliberately not inspected here.
    return "; ".join(value for value in descriptions if value)


def _group_id(episode: Mapping[str, Any], group_fields: Sequence[str]) -> str:
    values = [str(episode.get(field, "")).strip() for field in group_fields]
    if group_fields and all(values):
        return "|".join(f"{field}={value}" for field, value in zip(group_fields, values))
    return f"episode={int(episode['episode_index'])}"


def build_lerobot_episode_registry(
    root: str | Path,
    *,
    source_id: str,
    revision: str,
    primary_image_key: str,
    group_fields: Sequence[str] = (),
    input_image_keys: Sequence[str] | None = None,
    camera_roles: Mapping[str, str] | None = None,
    protocol_version: str = "v1",
    action_timestamp_column: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    info, episodes, tasks = load_lerobot_metadata(root)
    if protocol_version not in {"v1", "v2", "v3"}:
        raise ValueError("protocol_version must be v1, v2 or v3")
    input_keys = list(input_image_keys or [primary_image_key])
    if len(set(input_keys)) != len(input_keys) or primary_image_key not in input_keys:
        raise ValueError("input_image_keys must be unique and include the target primary camera")
    roles = dict(camera_roles or {primary_image_key: "primary"})
    if protocol_version in {"v2", "v3"} and (
        set(roles) != set(input_keys) or any(not str(value).strip() for value in roles.values())
        or len(set(roles.values())) != len(roles)
    ):
        raise ValueError("V2 requires one explicit, unique camera role per input image key")
    is_v3 = str(info.get("codebase_version", "")).lower().startswith("v3")
    chunks_size = int(info.get("chunks_size", 1000))
    file_global_starts: dict[tuple[int, int], int] = {}
    for episode in episodes:
        if episode.get("data/chunk_index") is None or episode.get("data/file_index") is None:
            continue
        key = (int(episode["data/chunk_index"]), int(episode["data/file_index"]))
        start = int(episode["dataset_from_index"])
        file_global_starts[key] = min(start, file_global_starts.get(key, start))
    rows = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_chunk = episode.get("data/chunk_index")
        data_file = episode.get("data/file_index")
        data_path = resolve_lerobot_path(
            root,
            str(info["data_path"]),
            episode_index=episode_index,
            chunks_size=chunks_size,
            chunk_index=data_chunk,
            file_index=data_file,
        )
        video_prefix = f"videos/{primary_image_key}"
        video_chunk = episode.get(f"{video_prefix}/chunk_index", data_chunk)
        video_file = episode.get(f"{video_prefix}/file_index", data_file)
        video_path = resolve_lerobot_path(
            root,
            str(info["video_path"]),
            episode_index=episode_index,
            chunks_size=chunks_size,
            video_key=primary_image_key,
            chunk_index=video_chunk,
            file_index=video_file,
        )
        lineage = f"{source_id}@{revision}:episode:{episode_index}"
        rows.append(
            {
                "sample_id": lineage,
                "source_id": source_id,
                "revision": revision,
                "episode_id": str(episode_index),
                # Group identifiers live in one combined multi-source
                # manifest. Namespace them so unrelated datasets that both
                # use values such as ``episode=0`` or ``home=1`` are not
                # accidentally tied to the same split.
                "group_id": f"{source_id}|{_group_id(episode, group_fields)}",
                "grouping_method": "metadata_fields" if group_fields else "episode_fallback",
                "lineage_ids": [lineage],
                "content_hashes": [],
                "paths": [str(data_path), str(video_path)],
                "data_path": str(data_path),
                "video_path": str(video_path),
                "instruction": _episode_instruction(episode, tasks),
                # Humanoid Everyday mixes G1/H1 embodiments. Preserve this
                # declaration so unequal state vectors use distinct adapters.
                "robot_type": episode.get("robot_type"),
                "nominal_fps": float(info["fps"]),
                "primary_image_key": primary_image_key,
                "dataset_from_index": episode.get("dataset_from_index"),
                "dataset_to_index": episode.get("dataset_to_index"),
                "dataset_stop_exclusive": (
                    int(episode["dataset_to_index"]) if is_v3 else int(episode["dataset_to_index"]) + 1
                ) if episode.get("dataset_to_index") is not None else None,
                "shared_data_file": data_chunk is not None and data_file is not None,
                "data_file_global_start": file_global_starts.get(
                    (int(data_chunk), int(data_file))
                ) if data_chunk is not None and data_file is not None else None,
                "video_from_timestamp": episode.get(f"{video_prefix}/from_timestamp"),
                "video_to_timestamp": episode.get(f"{video_prefix}/to_timestamp"),
            }
        )
        for temporal_key in (
            "video_timestamp_column", "image_timestamp_columns",
            "state_timestamp_column", "action_timestamp_column",
        ):
            if info.get(temporal_key):
                rows[-1][temporal_key] = info[temporal_key]
        if action_timestamp_column:
            features = info.get("features") or {}
            if action_timestamp_column not in features:
                raise ValueError(
                    f"action timestamp column {action_timestamp_column!r} is not declared by {root}"
                )
            rows[-1]["action_timestamp_column"] = action_timestamp_column
        if protocol_version in {"v2", "v3"}:
            views = []
            for key in input_keys:
                prefix = f"videos/{key}"
                path = resolve_lerobot_path(
                    root, str(info["video_path"]), episode_index=episode_index,
                    chunks_size=chunks_size, video_key=key,
                    chunk_index=episode.get(f"{prefix}/chunk_index", data_chunk),
                    file_index=episode.get(f"{prefix}/file_index", data_file),
                )
                # LeRobot 2.x usually namespaces video features as
                # ``observation.images.<camera>`` while the video template and
                # this camera ABI use the short camera name.
                features = info.get("features") or {}
                declared = (
                    not features
                    or key in features
                    or f"observation.images.{key}" in features
                )
                views.append({
                    "image_key": key, "camera_role": roles[key], "video_path": str(path),
                    "available": declared and path.is_file(),
                    "video_from_timestamp": episode.get(f"{prefix}/from_timestamp"),
                    "video_to_timestamp": episode.get(f"{prefix}/to_timestamp"),
                })
            if not next(view["available"] for view in views if view["image_key"] == primary_image_key):
                raise FileNotFoundError(f"V2 primary input/target video is unavailable: {video_path}")
            rows[-1].update({
                "format_version": f"predictor_pretrain_episode_{protocol_version}",
                "input_image_keys": input_keys, "input_views": views,
                "target_image_key": primary_image_key,
                "paths": [str(data_path), *[view["video_path"] for view in views if view["available"]]],
            })
    return rows


def build_lerobot_anchor_rows(
    episode_rows: Iterable[Mapping[str, Any]],
    *,
    timestamp_column: str = "timestamp",
    anchor_hz: float = 2.0,
    future_seconds: float = 0.6,
    protocol_version: str = "v1",
    candidate_phases: int = 1,
    visual_history_seconds: float = 0.0,
    action_steps: int = 30,
) -> list[dict[str, Any]]:
    """Read only timestamps from episode parquet and materialize anchor identities."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("LeRobot anchor construction requires pyarrow") from error
    if protocol_version not in {"v1", "v2", "v3"} or candidate_phases <= 0:
        raise ValueError("invalid anchor protocol or candidate_phases")
    if protocol_version == "v1" and candidate_phases != 1:
        raise ValueError("expanded candidate phases require V2 sample identities")
    if not np.isfinite(visual_history_seconds) or visual_history_seconds < 0:
        raise ValueError("visual_history_seconds must be finite and nonnegative")
    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    anchors: list[dict[str, Any]] = []
    cached_path: Path | None = None
    cached_table = None
    for episode in episode_rows:
        data_path = Path(str(episode["data_path"]))
        if data_path != cached_path:
            time_columns = [timestamp_column]
            if episode.get("video_timestamp_column"):
                time_columns.append(episode["video_timestamp_column"])
            if episode.get("state_timestamp_column"):
                time_columns.append(episode["state_timestamp_column"])
            if episode.get("action_timestamp_column"):
                time_columns.append(episode["action_timestamp_column"])
            time_columns.extend((episode.get("image_timestamp_columns") or {}).values())
            cached_table = pq.read_table(data_path, columns=list(dict.fromkeys(time_columns)))
            cached_path = data_path
        table = cached_table
        assert table is not None
        start = episode.get("dataset_from_index")
        stop = episode.get("dataset_stop_exclusive")
        if start is not None and stop is not None:
            start, stop = int(start), int(stop)
            if bool(episode.get("shared_data_file")):
                global_file_start = int(episode.get("data_file_global_start") or 0)
                row_offset = start - global_file_start
            else:
                row_offset = 0
            table = table.slice(row_offset, stop - start)
            global_row_offset = start
        else:
            row_offset = 0
            global_row_offset = 0
        timestamps = np.asarray(table[timestamp_column].to_numpy(), dtype=np.float64)
        video_times = np.asarray(table[episode["video_timestamp_column"]].to_numpy(), dtype=np.float64) if episode.get("video_timestamp_column") else timestamps
        if not np.isfinite(video_times).all() or np.any(np.diff(video_times) <= 0):
            raise ValueError("video timestamps must be finite and strictly increasing")
        aligned = {}
        for phase in range(candidate_phases):
            for item in align_visual_anchors(
                timestamps, anchor_hz=anchor_hz, future_seconds=future_seconds,
                nominal_fps=float(episode["nominal_fps"]),
                phase_seconds=phase / (anchor_hz * candidate_phases),
            ):
                aligned.setdefault(item.current_index, (item, phase))
        for item, phase in sorted(aligned.values(), key=lambda value: value[0].current_index):
            state_indices, state_valid, state_timestamps = causal_state_history_indices(
                timestamps,
                current_timestamp=item.current_timestamp,
                nominal_state_hz=float(episode["nominal_fps"]),
            )
            episode_lineage = str(episode["lineage_ids"][0])
            if episode.get("state_timestamp_column"):
                physical_states = np.asarray(table[episode["state_timestamp_column"]].to_numpy(), dtype=np.float64)
                state_timestamps = np.where(state_valid, physical_states[state_indices], np.nan)
                if np.any(state_timestamps[state_valid] > timestamps[state_indices[state_valid]] + 1e-9):
                    raise ValueError("converted state timestamp is noncausal")
            current_row = row_offset + item.current_index
            target_row = row_offset + item.target_index
            sample_id = f"{episode_lineage}:frame:{item.current_index}"
            legacy_sample_id = sample_id
            if protocol_version in {"v2", "v3"}:
                sample_id = f"{episode_lineage}:v2:frame:{item.current_index}:target:{item.target_index}"
                if protocol_version == "v3":
                    sample_id = f"{episode_lineage}:v3:frame:{item.current_index}:target:{item.target_index}"
            video_offset = float(episode.get("video_from_timestamp") or 0.0)
            anchors.append(
                {
                    **dict(episode),
                    "sample_id": sample_id,
                    "lineage_ids": [episode_lineage, sample_id],
                    "current_row_index": current_row,
                    "target_row_index": target_row,
                    "current_global_index": global_row_offset + item.current_index,
                    "target_global_index": global_row_offset + item.target_index,
                    "current_timestamp": item.current_timestamp,
                    "target_timestamp": item.target_timestamp,
                    "actual_delta_t": item.actual_delta_t,
                    "current_video_timestamp": video_offset + float(video_times[item.current_index]),
                    "target_video_timestamp": video_offset + float(video_times[item.target_index]),
                    "state_row_indices": (state_indices + row_offset).tolist(),
                    "state_history_mask": state_valid.tolist(),
                    "state_timestamps": [None if np.isnan(value) else float(value) for value in state_timestamps],
                }
            )
            if protocol_version in {"v2", "v3"}:
                views = episode.get("input_views")
                if not views:
                    raise ValueError("V2 anchors require V2 episode input_views metadata")
                history_query = item.current_timestamp - visual_history_seconds
                history_index = int(np.searchsorted(timestamps, history_query + 1e-9, side="right")) - 1
                history_valid = bool(history_index >= 0 and (
                    history_query - timestamps[max(history_index, 0)] <= 1.5 / float(episode["nominal_fps"])
                ))
                slots = []
                for view in views:
                    primary = view["image_key"] == episode["primary_image_key"]
                    offsets = [-visual_history_seconds, 0.0] if primary and visual_history_seconds > 0 else [0.0]
                    for offset in offsets:
                        historical = offset < 0
                        local_index = history_index if historical else item.current_index
                        available = bool(view["available"]) and (not historical or history_valid)
                        actual_time = float(timestamps[local_index]) if available else None
                        camera_time_column = (episode.get("image_timestamp_columns") or {}).get(view["image_key"])
                        if available and camera_time_column:
                            actual_time = float(table[camera_time_column][local_index].as_py())
                            if not np.isfinite(actual_time) or actual_time > timestamps[local_index] + 1e-9:
                                raise ValueError("converted camera timestamp is noncausal")
                        slots.append({
                            **dict(view), "offset_seconds": offset, "available": available,
                            "timestamp": actual_time,
                            "actual_offset_seconds": actual_time - item.current_timestamp if available else None,
                            "row_index": row_offset + local_index if available else None,
                            "video_timestamp": float(view.get("video_from_timestamp") or 0.0) + float(video_times[local_index]) if available else None,
                            "is_primary_current": primary and not historical,
                        })
                anchors[-1].update({
                    "format_version": f"predictor_pretrain_anchor_{protocol_version}",
                    "legacy_sample_id": legacy_sample_id,
                    "visual_inputs": slots,
                    "anchor_phase": phase, "candidate_phases": candidate_phases,
                    "anchor_hz_per_phase": anchor_hz,
                    "visual_history_seconds": visual_history_seconds,
                    "future_seconds": future_seconds,
                })
            if protocol_version == "v3":
                explicit_action_clock = bool(episode.get("action_timestamp_column"))
                action_times = np.asarray(
                    table[episode["action_timestamp_column"]].to_numpy()
                    if explicit_action_clock else timestamps,
                    dtype=np.float64,
                )
                # Converted control streams are commonly sampled causally and
                # held until the next command.  Their physical timestamps are
                # therefore nondecreasing (with repeats), not strictly
                # increasing like the camera clock.
                if not np.isfinite(action_times).all() or np.any(np.diff(action_times) < 0):
                    raise ValueError("action timestamps must be finite and nondecreasing")
                requested = item.current_timestamp + (
                    np.arange(action_steps, dtype=np.float64) * future_seconds / action_steps
                )
                if explicit_action_clock:
                    # A command must already exist at the requested physical
                    # time.  Use the last causal command and retain a mask for
                    # any uncovered boundary instead of leaking a future one.
                    causal = np.searchsorted(action_times, requested + 1e-9, side="right") - 1
                    action_valid = causal >= 0
                    action_indices = np.clip(causal, 0, len(action_times) - 1)
                else:
                    # Frame-aligned LeRobot actions have no independent clock.
                    # Nearest-row resampling is appropriate, with the same
                    # dropped-frame tolerance used by visual alignment.
                    right = np.searchsorted(action_times, requested, side="left")
                    right = np.clip(right, 0, len(action_times) - 1)
                    left = np.clip(right - 1, 0, len(action_times) - 1)
                    choose_left = (
                        np.abs(action_times[left] - requested)
                        <= np.abs(action_times[right] - requested)
                    )
                    action_indices = np.where(choose_left, left, right)
                    tolerance = 1.5 / float(episode["nominal_fps"]) + 1e-6
                    action_valid = np.abs(action_times[action_indices] - requested) <= tolerance
                anchors[-1].update({
                    "action_row_indices": (action_indices + row_offset).tolist(),
                    "action_timestamps": action_times[action_indices].tolist(),
                    "action_requested_timestamps": requested.tolist(),
                    "action_time_mask": action_valid.tolist(),
                    "action_steps": action_steps,
                })
    return anchors


__all__ = [
    "build_lerobot_anchor_rows",
    "build_lerobot_episode_registry",
    "load_lerobot_metadata",
    "resolve_lerobot_path",
]
