"""Physical-time alignment shared by all predictor pretraining sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AlignedAnchor:
    current_index: int
    target_index: int
    current_timestamp: float
    target_timestamp: float
    actual_delta_t: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _timestamps(values: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if (np.diff(result) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing within one episode")
    return result


def _nearest_index(timestamps: np.ndarray, query: float) -> int:
    right = int(np.searchsorted(timestamps, query, side="left"))
    candidates = [index for index in (right - 1, right) if 0 <= index < len(timestamps)]
    return min(candidates, key=lambda index: (abs(float(timestamps[index]) - query), index))


def align_visual_anchors(
    frame_timestamps: Sequence[float],
    *,
    anchor_hz: float = 2.0,
    future_seconds: float = 0.6,
    nominal_fps: float,
    tolerance_seconds: float | None = None,
    phase_seconds: float = 0.0,
) -> list[AlignedAnchor]:
    """Select real current/target frames without crossing an episode boundary."""

    timestamps = _timestamps(frame_timestamps, name="frame_timestamps")
    if anchor_hz <= 0 or future_seconds <= 0 or nominal_fps <= 0:
        raise ValueError("anchor_hz, future_seconds and nominal_fps must be positive")
    tolerance = (
        0.5 / nominal_fps + 0.001
        if tolerance_seconds is None
        else float(tolerance_seconds)
    )
    if tolerance < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    step = 1.0 / anchor_hz
    if not np.isfinite(phase_seconds) or not 0 <= phase_seconds < step:
        raise ValueError("phase_seconds must be within one anchor period")
    first_query = float(timestamps[0]) + phase_seconds
    last_query = float(timestamps[-1]) - future_seconds
    if last_query < first_query:
        return []
    count = int(np.floor((last_query - first_query) / step + 1e-9)) + 1
    result: list[AlignedAnchor] = []
    seen_current: set[int] = set()
    for occurrence in range(count):
        current_query = first_query + occurrence * step
        current_index = _nearest_index(timestamps, current_query)
        if current_index in seen_current:
            continue
        target_query = float(timestamps[current_index]) + future_seconds
        target_index = _nearest_index(timestamps, target_query)
        error = abs(float(timestamps[target_index]) - target_query)
        if target_index <= current_index or error > tolerance:
            continue
        seen_current.add(current_index)
        actual = float(timestamps[target_index] - timestamps[current_index])
        result.append(
            AlignedAnchor(
                current_index=current_index,
                target_index=target_index,
                current_timestamp=float(timestamps[current_index]),
                target_timestamp=float(timestamps[target_index]),
                actual_delta_t=actual,
            )
        )
    return result


def causal_state_history_indices(
    state_timestamps: Sequence[float],
    *,
    current_timestamp: float,
    offsets_seconds: Sequence[float] = (-0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0),
    nominal_state_hz: float,
    max_age_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the most recent causal state for each fixed physical-time slot."""

    timestamps = _timestamps(state_timestamps, name="state_timestamps")
    if nominal_state_hz <= 0:
        raise ValueError("nominal_state_hz must be positive")
    max_age = 2.0 / nominal_state_hz if max_age_seconds is None else float(max_age_seconds)
    if max_age < 0:
        raise ValueError("max_age_seconds must be non-negative")
    indices = np.zeros(len(offsets_seconds), dtype=np.int64)
    valid = np.zeros(len(offsets_seconds), dtype=bool)
    selected_timestamps = np.full(len(offsets_seconds), np.nan, dtype=np.float64)
    previous_offset = float("-inf")
    for slot, offset in enumerate(offsets_seconds):
        offset = float(offset)
        if offset < previous_offset or offset > 0:
            raise ValueError("state offsets must be sorted and non-positive")
        previous_offset = offset
        query = float(current_timestamp) + offset
        index = int(np.searchsorted(timestamps, query, side="right")) - 1
        if index < 0:
            continue
        age = query - float(timestamps[index])
        if age < -1e-9 or age > max_age + 1e-12:
            continue
        indices[slot] = index
        valid[slot] = True
        selected_timestamps[slot] = timestamps[index]
    return indices, valid, selected_timestamps


__all__ = ["AlignedAnchor", "align_visual_anchors", "causal_state_history_indices"]
