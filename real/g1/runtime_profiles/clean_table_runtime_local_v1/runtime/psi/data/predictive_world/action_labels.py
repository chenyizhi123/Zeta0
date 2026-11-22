"""Materialize indexed native labels, retaining independent component masks."""
from collections.abc import Mapping

import numpy as np


COMPONENT_FIELDS = (
    "action_indices_by_component", "action_valid_by_component", "action_actual_times_by_component",
)


def validate_action_component_index(row, horizon):
    label_mode = row.get("materialization", {}).get("real_action_label_mode")
    if label_mode not in (None, "held_command_v1"):
        raise ValueError("unsupported native action label mode")
    present = [name in row for name in COMPONENT_FIELDS]
    materialization = row.get("materialization", {})
    mode = materialization.get("future_component_alignment")
    if mode not in (None, "independent_components_v1"):
        raise ValueError("unsupported future_component_alignment in action labels")
    layout = materialization.get("action_component_slices")
    if mode is not None and (not all(present) or layout is None):
        raise ValueError("independent action component publication is incomplete")
    if not any(present) and layout is None:
        return None
    if not all(present) or not isinstance(layout, Mapping) or not layout:
        raise ValueError("independent action component publication is incomplete")
    keys = set(layout)
    if any(not isinstance(row[field], Mapping) or set(row[field]) != keys for field in COMPONENT_FIELDS):
        raise ValueError("action component index fields and slices disagree")
    cursor = 0
    for bounds in layout.values():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)):
            raise ValueError("action component slices must partition the native width")
    ordered = sorted(layout, key=lambda key: layout[key][0])
    for key in ordered:
        bounds = layout[key]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or bounds[0] != cursor or bounds[1] <= cursor):
            raise ValueError("action component slices must partition the native width")
        cursor = bounds[1]
        for field in COMPONENT_FIELDS:
            if not isinstance(row[field][key], list) or len(row[field][key]) != horizon:
                raise ValueError("action component index length disagrees with the action horizon")
        if any(type(index) is not int or index < 0 for index in row[COMPONENT_FIELDS[0]][key]):
            raise ValueError("action component source indices must be nonnegative integers")
        valid = row[COMPONENT_FIELDS[1]][key]
        if any(type(value) is not bool for value in valid):
            raise ValueError("action component validity must be boolean")
        times = np.asarray(row[COMPONENT_FIELDS[2]][key], dtype=np.float64)
        if not np.isfinite(times[valid]).all() or (label_mode is None and (times[valid] <= 0).any()):
            raise ValueError("valid action component times must lie after the observation anchor")
    complete = np.logical_and.reduce([row[COMPONENT_FIELDS[1]][key] for key in ordered])
    if not np.array_equal(complete, np.asarray(row["action_valid"], dtype=bool)):
        raise ValueError("complete action slots must equal the AND of component validity")
    return layout


def materialize_action_labels(row, read_vectors):
    """Read row vectors through one callback; assemble each component at its source row."""
    horizon = len(row["action_indices"])
    layout = validate_action_component_index(row, horizon)
    if layout is None:
        values = np.asarray(read_vectors(row["action_indices"]), dtype=np.float32)
        if values.ndim != 2 or len(values) != horizon:
            raise ValueError("native action vectors must match the indexed horizon")
        valid = np.broadcast_to(np.asarray(row["action_valid"], dtype=bool)[:, None], values.shape).copy()
    else:
        width = max(stop for _, stop in layout.values())
        values = np.zeros((horizon, width), dtype=np.float32)
        valid = np.zeros_like(values, dtype=bool)
        for key, (start, stop) in layout.items():
            selected = np.asarray(read_vectors(row["action_indices_by_component"][key]), dtype=np.float32)
            if selected.shape != values.shape:
                raise ValueError("native action vectors disagree with component slices")
            values[:, start:stop] = selected[:, start:stop]
            valid[:, start:stop] = np.asarray(row["action_valid_by_component"][key])[:, None]
    return np.where(valid, values, 0.0).astype(np.float32, copy=False), valid
