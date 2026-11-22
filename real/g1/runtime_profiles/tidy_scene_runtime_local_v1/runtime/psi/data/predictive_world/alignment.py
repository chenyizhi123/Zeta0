"""Causal physical-time selection for observed and future windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _strict_timestamps(values: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D sequence")
    if (np.diff(result) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing within one episode")
    return result


def causal_floor_indices(
    timestamps: Sequence[float],
    query_times: Sequence[float],
    *,
    max_age_seconds: float,
    frame_grid_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the newest real sample not later than each requested time.

    Audited synchronous sources may opt into a canonical frame clock. Source
    times must already equal integer frame ticks / Hz; only float64 arithmetic
    roundoff in on-grid queries is removed. The default asynchronous path and
    the max-age/episode-tail hold policy are unchanged.
    """

    source = _strict_timestamps(timestamps, name="timestamps")
    queries = np.asarray(query_times, dtype=np.float64)
    if queries.ndim != 1 or not np.isfinite(queries).all():
        raise ValueError("query_times must be a finite 1-D sequence")
    if max_age_seconds < 0 or not np.isfinite(max_age_seconds):
        raise ValueError("max_age_seconds must be finite and non-negative")
    if frame_grid_hz is not None:
        if isinstance(frame_grid_hz, bool) or not np.isfinite(frame_grid_hz) or frame_grid_hz <= 0:
            raise ValueError("frame_grid_hz must be finite and positive")
        if not np.array_equal(source, np.rint(source * frame_grid_hz) / frame_grid_hz):
            raise ValueError("timestamps must use the canonical frame grid before selection")
        ticks = queries * frame_grid_hz
        rounded = np.rint(ticks)
        tolerance = 16 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(ticks))
        if np.any(np.abs(ticks - rounded) > tolerance):
            raise ValueError("query_times must lie on the declared frame grid")
        queries = rounded / frame_grid_hz
    indices = np.zeros(len(queries), dtype=np.int64)
    actual = np.full(len(queries), np.nan, dtype=np.float64)
    valid = np.zeros(len(queries), dtype=bool)
    for slot, query in enumerate(queries):
        index = int(np.searchsorted(source, query, side="right")) - 1
        if index < 0:
            continue
        age = float(query - source[index])
        if age <= max_age_seconds + 1e-12:
            indices[slot] = index
            actual[slot] = source[index]
            valid[slot] = True
    return indices, valid, actual


@dataclass(frozen=True)
class PredictiveWindowIndices:
    history_visual_indices: np.ndarray
    history_visual_valid: np.ndarray
    history_visual_times: np.ndarray
    state_indices: np.ndarray
    state_valid: np.ndarray
    state_times: np.ndarray
    future_visual_indices: np.ndarray
    future_visual_valid: np.ndarray
    future_visual_times: np.ndarray


def select_predictive_window(
    *,
    anchor_time: float,
    visual_timestamps: Sequence[float],
    state_timestamps: Sequence[float],
    history_visual_offsets_s: Sequence[float] = (-0.4, -0.2, 0.0),
    state_offsets_s: Sequence[float] = tuple(np.linspace(-0.4, 0.0, 21)),
    future_group_ends_s: Sequence[float] = (0.2, 0.4, 0.6),
    visual_max_age_s: float,
    state_max_age_s: float,
    frame_grid_hz: float | None = None,
) -> PredictiveWindowIndices:
    if not np.isfinite(anchor_time):
        raise ValueError("anchor_time must be finite")
    history_offsets = np.asarray(history_visual_offsets_s, dtype=np.float64)
    state_offsets = np.asarray(state_offsets_s, dtype=np.float64)
    future_offsets = np.asarray(future_group_ends_s, dtype=np.float64)
    if (history_offsets > 0).any() or (state_offsets > 0).any():
        raise ValueError("observed offsets must be causal")
    if (future_offsets <= 0).any():
        raise ValueError("future offsets must be positive")
    hi, hv, ht = causal_floor_indices(
        visual_timestamps, anchor_time + history_offsets, max_age_seconds=visual_max_age_s,
        frame_grid_hz=frame_grid_hz,
    )
    si, sv, st = causal_floor_indices(
        state_timestamps, anchor_time + state_offsets, max_age_seconds=state_max_age_s,
        frame_grid_hz=frame_grid_hz,
    )
    fi, fv, ft = causal_floor_indices(
        visual_timestamps, anchor_time + future_offsets, max_age_seconds=visual_max_age_s,
        frame_grid_hz=frame_grid_hz,
    )
    # A floor sample at/before t0 is not future evidence even if a generous
    # age threshold would otherwise accept it.
    fv &= np.isfinite(ft) & (ft > anchor_time)
    return PredictiveWindowIndices(hi, hv, ht, si, sv, st, fi, fv, ft)


def unique_measurement_weights(indices: Sequence[int], valid: Sequence[bool]) -> np.ndarray:
    """Give each unique real measurement total weight one despite resampling."""

    index_array = np.asarray(indices, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=bool)
    if index_array.ndim != 1 or valid_array.shape != index_array.shape:
        raise ValueError("indices and valid must be aligned 1-D arrays")
    weights = np.zeros(len(index_array), dtype=np.float64)
    for index in np.unique(index_array[valid_array]):
        rows = valid_array & (index_array == index)
        weights[rows] = 1.0 / int(rows.sum())
    return weights


def action_prefix_visibility(
    action_times: np.ndarray,
    action_valid: np.ndarray,
    group_ends_s: Sequence[float],
) -> np.ndarray:
    """Return [B,G,T] visibility using the strict command_time < group_end rule."""

    times = np.asarray(action_times, dtype=np.float64)
    valid = np.asarray(action_valid, dtype=bool)
    ends = np.asarray(group_ends_s, dtype=np.float64)
    if times.ndim != 2 or valid.shape != times.shape:
        raise ValueError("action times and step validity must share shape [B,T]")
    if ends.ndim != 1 or (ends <= 0).any() or not np.isfinite(ends).all():
        raise ValueError("group ends must be finite and positive")
    if not np.isfinite(times[valid]).all():
        raise ValueError("valid action times must be finite")
    return valid[:, None, :] & (times[:, None, :] < ends[None, :, None])


__all__ = [
    "PredictiveWindowIndices",
    "action_prefix_visibility",
    "causal_floor_indices",
    "select_predictive_window",
    "unique_measurement_weights",
]
