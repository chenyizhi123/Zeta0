"""Production-facing row and collation boundary for predictive-world training."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from psi.data.predictive_world.alignment import unique_measurement_weights
from psi.data.predictive_world.contracts import (
    FutureSupervision,
    NativeActionBatch,
    ObservedHistory,
    PredictiveWorldBatch,
    RaggedPatchBatch,
    SampleMetadata,
)


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    result = torch.as_tensor(value)
    return result if dtype is None else result.to(dtype=dtype)


def _pad_2d(rows: Sequence[Tensor], *, value: float = 0) -> Tensor:
    if not rows or any(row.ndim != 2 for row in rows):
        raise ValueError("padding requires non-empty rank-2 tensors")
    length = max(row.shape[0] for row in rows)
    width = max(row.shape[1] for row in rows)
    output = rows[0].new_full((len(rows), length, width), value)
    for index, row in enumerate(rows):
        output[index, : row.shape[0], : row.shape[1]] = row
    return output


def _collate_qwen(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise ValueError("all qwen input rows must expose identical keys")
    result: dict[str, Any] = {}
    for key in sorted(keys):
        if key in {"image_counts", "primary_current_image_indices"}:
            continue
        rows = [_tensor(value[key]) for value in values]
        if key in {"input_ids", "attention_mask"}:
            flattened = [row.reshape(-1) for row in rows]
            width = max(row.numel() for row in flattened)
            fill = 0
            output = flattened[0].new_full((len(rows), width), fill)
            for index, row in enumerate(flattened):
                output[index, : row.numel()] = row
            result[key] = output
        elif key in {"pixel_values", "image_grid_thw", "image_content_boxes"}:
            result[key] = torch.cat(rows, dim=0)
        elif all(row.shape == rows[0].shape for row in rows):
            result[key] = torch.stack(rows)
        else:
            raise ValueError(f"unsupported ragged qwen field {key!r}")
    if "image_counts" in keys:
        counts = torch.tensor([int(_tensor(value["image_counts"]).item()) for value in values])
        if (counts <= 0).any():
            raise ValueError("each Qwen row needs at least one image")
        result["image_counts"] = counts
        starts = counts.cumsum(0) - counts
        local_primary = torch.tensor([
            int(_tensor(value["primary_current_image_indices"]).item()) for value in values
        ])
        if ((local_primary < 0) | (local_primary >= counts)).any():
            raise ValueError("Qwen primary current image is outside its row")
        result["primary_current_image_indices"] = starts + local_primary
    return result


class PredictiveWorldCollator:
    """Collate audited, materialized rows without inventing missing observations."""

    def __init__(self, *, groups: int = 3, state_count: int = 21, action_horizon: int = 30):
        self.groups = int(groups)
        self.state_count = int(state_count)
        self.action_horizon = int(action_horizon)
        if min(self.groups, self.state_count, self.action_horizon) <= 0:
            raise ValueError("collator dimensions must be positive")

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> PredictiveWorldBatch:
        if not rows:
            raise ValueError("cannot collate an empty predictive-world batch")
        batch = len(rows)
        qwen = _collate_qwen([row["qwen_inputs"] for row in rows])

        states = [_tensor(row["raw_states"], dtype=torch.float32) for row in rows]
        if any(value.ndim != 2 or value.shape[0] != self.state_count for value in states):
            raise ValueError("every row needs exactly state_count raw state samples")
        raw_states = _pad_2d(states)
        state_valid = torch.zeros_like(raw_states, dtype=torch.bool)
        state_times = torch.zeros(batch, self.state_count, dtype=torch.float32)
        for index, (row, value) in enumerate(zip(rows, states)):
            valid = _tensor(row["state_valid"], dtype=torch.bool)
            if valid.shape == (self.state_count,):
                valid = valid[:, None].expand_as(value)
            if valid.shape != value.shape:
                raise ValueError("state_valid must be per-time or per-element")
            state_valid[index, :, : value.shape[1]] = valid
            times = _tensor(row["state_times"], dtype=torch.float32)
            if times.shape != (self.state_count,):
                raise ValueError("state_times must have shape [state_count]")
            state_times[index] = times

        action_rows = [_tensor(row["actions"], dtype=torch.float32) for row in rows]
        if any(value.ndim != 2 or value.shape[0] != self.action_horizon for value in action_rows):
            raise ValueError("every row needs exactly action_horizon action samples")
        actions = _pad_2d(action_rows)
        action_valid = torch.zeros_like(actions, dtype=torch.bool)
        action_times = torch.zeros(batch, self.action_horizon, dtype=torch.float32)
        native_dims = torch.empty(batch, dtype=torch.long)
        for index, (row, value) in enumerate(zip(rows, action_rows)):
            valid = _tensor(row["action_valid"], dtype=torch.bool)
            if valid.shape == (self.action_horizon,):
                valid = valid[:, None].expand_as(value)
            if valid.shape != value.shape:
                raise ValueError("action_valid must be per-time or per-element")
            action_valid[index, :, : value.shape[1]] = valid
            times = _tensor(row["action_times"], dtype=torch.float32)
            if times.shape != (self.action_horizon,):
                raise ValueError("action_times must have shape [action_horizon]")
            action_times[index] = times
            native_dims[index] = value.shape[1]
        protocols = tuple(str(row["action_protocol_id"]) for row in rows)
        spaces = tuple(str(row["action_normalization_space"]) for row in rows)
        demonstrated = NativeActionBatch(actions, action_valid, action_times, protocols, native_dims, spaces)

        patch_values, offsets, grids = [], [0], []
        sample_indices, group_indices, cameras, actual_times = [], [], [], []
        latent_valid = torch.zeros(batch, self.groups, dtype=torch.bool)
        for sample_index, row in enumerate(rows):
            patches = row["future_patch_fields"]
            if len(patches) != self.groups:
                raise ValueError("future_patch_fields must contain one list per latent group")
            for group_index, fields in enumerate(patches):
                for field in fields:
                    values = _tensor(field["values"])
                    grid = tuple(int(value) for value in field["grid_hw"])
                    if values.ndim != 2 or values.shape[0] != grid[0] * grid[1]:
                        raise ValueError("future patch field does not match grid geometry")
                    patch_values.append(values)
                    offsets.append(offsets[-1] + values.shape[0])
                    grids.append(grid)
                    sample_indices.append(sample_index)
                    group_indices.append(group_index)
                    cameras.append(str(field["camera_id"]))
                    actual_times.append(float(field["actual_time"]))
                    latent_valid[sample_index, group_index] = True
        if not patch_values:
            raise ValueError("batch has no future visual supervision")
        visual = RaggedPatchBatch(
            values=torch.cat(patch_values, dim=0),
            offsets=torch.tensor(offsets, dtype=torch.long),
            grid_hw=torch.tensor(grids, dtype=torch.long),
            content_valid=torch.ones(offsets[-1], dtype=torch.bool),
            sample_indices=torch.tensor(sample_indices, dtype=torch.long),
            group_indices=torch.tensor(group_indices, dtype=torch.long),
            camera_ids=tuple(cameras),
            actual_times=torch.tensor(actual_times, dtype=torch.float32),
        )

        measurements = [_tensor(row["future_measurements"], dtype=torch.float32) for row in rows]
        if any(value.ndim != 2 for value in measurements):
            raise ValueError("future measurements must be [T,D]")
        measured = _pad_2d(measurements)
        measured_valid = torch.zeros_like(measured, dtype=torch.bool)
        measured_times = torch.zeros(measured.shape[:2], dtype=torch.float32)
        source_indices = torch.zeros(measured.shape[:2], dtype=torch.long)
        for index, (row, value) in enumerate(zip(rows, measurements)):
            valid = _tensor(row["future_measurement_valid"], dtype=torch.bool)
            if valid.shape != value.shape:
                raise ValueError("future measurement validity must match values")
            measured_valid[index, : value.shape[0], : value.shape[1]] = valid
            times = _tensor(row["future_measurement_times"], dtype=torch.float32)
            indices = _tensor(row["future_measurement_source_indices"], dtype=torch.long)
            if times.shape != (value.shape[0],) or indices.shape != times.shape:
                raise ValueError("future measurement time/source-index shape mismatch")
            measured_times[index, : value.shape[0]] = times
            source_indices[index, : value.shape[0]] = indices

        observed = ObservedHistory(
            qwen_inputs=qwen,
            raw_states=raw_states,
            state_valid=state_valid,
            state_times=state_times,
            state_schema_ids=tuple(str(row["state_schema_id"]) for row in rows),
            action_protocol_ids=protocols,
            action_native_dims=native_dims,
            action_normalization_spaces=spaces,
        )
        future = FutureSupervision(
            visual=visual,
            measured_states=measured,
            measured_valid=measured_valid,
            measured_times=measured_times,
            measured_source_indices=source_indices,
            latent_target_valid=latent_valid,
        )
        metadata = SampleMetadata(
            sample_ids=tuple(str(row["sample_id"]) for row in rows),
            source_ids=tuple(str(row["source_id"]) for row in rows),
            episode_ids=tuple(str(row["episode_id"]) for row in rows),
            occurrence_indices=torch.tensor([int(row["occurrence_index"]) for row in rows]),
            split=str(rows[0]["split"]),
        )
        if any(str(row["split"]) != metadata.split for row in rows):
            raise ValueError("one batch cannot mix dataset splits")
        result = PredictiveWorldBatch(observed, future, demonstrated, metadata)
        result.validate(state_count=self.state_count, groups=self.groups, action_horizon=self.action_horizon)
        return result


class PredictiveWorldTrainingCollator:
    """Produce the dense model boundary while retaining the typed source batch."""

    def __init__(
        self,
        *,
        future_measurement_offsets_s: Sequence[float],
        future_group_ends_s: Sequence[float] = (0.2, 0.4, 0.6),
        state_count: int = 21,
        action_horizon: int = 30,
        body_points_per_group: int = 10,
        action_condition_time_mode: str | None = None,
        action_hz: float = 50.0,
    ) -> None:
        if action_condition_time_mode not in (None, "fixed_action_grid_v1"):
            raise ValueError("unsupported action_condition_time_mode")
        if not math.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("action_hz must be finite and positive")
        self.action_condition_time_mode = action_condition_time_mode
        self.action_grid = (torch.arange(1, action_horizon + 1, dtype=torch.float64) / action_hz).float()
        self.group_ends = tuple(float(value) for value in future_group_ends_s)
        self.body_points_per_group = int(body_points_per_group)
        if self.body_points_per_group <= 0:
            raise ValueError("body_points_per_group must be positive")
        queries = torch.tensor(tuple(future_measurement_offsets_s), dtype=torch.float64)
        ends = torch.tensor(self.group_ends, dtype=torch.float64)
        if (
            queries.ndim != 1 or not queries.numel() or not ends.numel()
            or not torch.isfinite(queries).all() or not torch.isfinite(ends).all()
            or queries[0] <= 0 or ends[0] <= 0
            or (queries[1:] <= queries[:-1]).any() or (ends[1:] <= ends[:-1]).any()
            or queries[-1] > ends[-1] + 1e-7
        ):
            raise ValueError("future measurement queries and group ends must be finite, ordered and in range")
        self.measurement_count = queries.numel()
        self.measurement_indices = []
        slot_offsets = torch.full((len(ends), self.body_points_per_group), float("nan"))
        lower = 0.0
        for group, upper in enumerate(self.group_ends):
            lower_bound = lower + 1e-7 if group else 0.0
            selected = ((queries > lower_bound) & (queries <= upper + 1e-7)).nonzero().flatten()
            if len(selected) > self.body_points_per_group:
                raise ValueError(
                    f"future group has {len(selected)} body points, exceeding configured "
                    f"body_points_per_group={self.body_points_per_group}"
                )
            self.measurement_indices.append(selected)
            slot_offsets[group, :len(selected)] = (queries[selected] - lower).float()
            lower = upper
        self.measurement_pair_valid = torch.isclose(
            slot_offsets[1:], slot_offsets[:-1], atol=1e-6, rtol=0.0
        )
        self.base = PredictiveWorldCollator(
            groups=len(self.group_ends), state_count=state_count, action_horizon=action_horizon
        )

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if any(len(row["future_measurement_times"]) != self.measurement_count for row in rows):
            raise ValueError("future measurements must retain every nominal query slot from the index ABI")
        typed = self.base(rows)
        batch, groups = len(rows), len(self.group_ends)
        visual_width = typed.future.visual.values.shape[-1]
        grouped_fields: list[list[list[Tensor]]] = [[[] for _ in range(groups)] for _ in rows]
        grouped_times: list[list[list[float]]] = [[[] for _ in range(groups)] for _ in rows]
        visual = typed.future.visual
        for image_index in range(len(visual.camera_ids)):
            start, stop = int(visual.offsets[image_index]), int(visual.offsets[image_index + 1])
            sample, group = int(visual.sample_indices[image_index]), int(visual.group_indices[image_index])
            grouped_fields[sample][group].append(visual.values[start:stop])
            grouped_times[sample][group].append(float(visual.actual_times[image_index]))
        max_patches = max(
            sum(field.shape[0] for field in grouped_fields[sample][group])
            for sample in range(batch) for group in range(groups)
        )
        future_patches = visual.values.new_zeros(batch, groups, max_patches, visual_width)
        future_patch_valid = torch.zeros(batch, groups, max_patches, dtype=torch.bool)
        future_patch_positions = torch.zeros(batch, groups, max_patches, 2, dtype=torch.float32)
        future_patch_camera_indices = torch.zeros(batch, groups, max_patches, dtype=torch.long)
        future_scene_times = torch.zeros(batch, groups, dtype=torch.float32)
        for sample in range(batch):
            for group in range(groups):
                fields = grouped_fields[sample][group]
                if fields:
                    joined = torch.cat(fields, dim=0)
                    future_patches[sample, group, : joined.shape[0]] = joined
                    future_patch_valid[sample, group, : joined.shape[0]] = True
                    cursor = 0
                    image_rows = (visual.sample_indices == sample) & (visual.group_indices == group)
                    for image_index in image_rows.nonzero().flatten().tolist():
                        height, width = visual.grid_hw[image_index].tolist()
                        y, x = torch.meshgrid(
                            torch.linspace(-1, 1, height), torch.linspace(-1, 1, width),
                            indexing="ij",
                        )
                        count = height * width
                        future_patch_positions[sample, group, cursor:cursor + count] = torch.stack(
                            (x.flatten(), y.flatten()), dim=-1
                        )
                        camera = visual.camera_ids[image_index].encode("utf-8")
                        bucket = int.from_bytes(hashlib.sha256(camera).digest()[:8], "big") % 64
                        future_patch_camera_indices[sample, group, cursor:cursor + count] = bucket
                        cursor += count
                    times = grouped_times[sample][group]
                    if max(times) - min(times) > 1e-4:
                        raise ValueError("cameras in one future group must share a physical anchor time")
                    future_scene_times[sample, group] = sum(times) / len(times)

        measured = typed.future.measured_states
        measured_valid = typed.future.measured_valid
        measured_times = typed.future.measured_times
        measured_source_indices = typed.future.measured_source_indices
        measured_unique_weights = torch.zeros(measured.shape[:2], dtype=torch.float32)
        for row in range(batch):
            weights = unique_measurement_weights(
                measured_source_indices[row].tolist(),
                measured_valid[row].any(dim=-1).tolist(),
            )
            measured_unique_weights[row] = torch.from_numpy(weights).float()
        max_measurements = self.body_points_per_group
        future_measurements = measured.new_zeros(
            batch, groups, max_measurements, measured.shape[-1]
        )
        future_measurement_valid = torch.zeros_like(future_measurements, dtype=torch.bool)
        future_measurement_weights = torch.zeros_like(future_measurements, dtype=torch.float32)
        future_measurement_times = torch.zeros(batch, groups, max_measurements)
        for row in range(batch):
            for group in range(groups):
                # Query identity defines the slot, not its as-of timestamp or
                # validity. Missing values stay masked in place; later points
                # must never slide into an earlier readout query.
                selected = self.measurement_indices[group]
                count = len(selected)
                if count:
                    future_measurements[row, group, :count] = measured[row].index_select(0, selected)
                    future_measurement_valid[row, group, :count] = measured_valid[row].index_select(0, selected)
                    selected_weights = measured_unique_weights[row].index_select(0, selected)
                    future_measurement_weights[row, group, :count] = selected_weights[:, None]
                    future_measurement_times[row, group, :count] = measured_times[row].index_select(0, selected)

        current_rows = [_tensor(row["current_measurements"], dtype=torch.float32).reshape(-1) for row in rows]
        if len({value.shape[0] for value in current_rows}) != 1:
            raise ValueError("canonical current measurement width must be identical across sources")
        current_measurements = torch.stack(current_rows)
        current_valid_rows = [
            _tensor(row["current_measurement_valid"], dtype=torch.bool).reshape(-1) for row in rows
        ]
        if any(value.shape != current_rows[index].shape for index, value in enumerate(current_valid_rows)):
            raise ValueError("current measurement validity must match canonical values")
        world_action_valid_rows = [
            tuple(bool(value) for value in row.get("world_action_group_valid", ()))
            for row in rows
        ]
        if any(len(value) != groups for value in world_action_valid_rows):
            raise ValueError("world-action group eligibility must have exactly one value per group")
        model_actions = typed.demonstrated
        if self.action_condition_time_mode == "fixed_action_grid_v1":
            model_actions = replace(model_actions, times=self.action_grid[None].expand(batch, -1).clone())
        result = {
            "typed_batch": typed,
            "observed": typed.observed,
            "actions": model_actions,
            "current_measurements": current_measurements,
            "current_measurement_valid": torch.stack(current_valid_rows),
            "future_patches": future_patches,
            "future_patch_valid": future_patch_valid,
            "future_patch_positions": future_patch_positions,
            "future_patch_camera_indices": future_patch_camera_indices,
            "future_scene_times": future_scene_times,
            "future_measurements": future_measurements,
            "future_measurement_valid": future_measurement_valid,
            "future_measurement_weights": future_measurement_weights,
            "future_measurement_times": future_measurement_times,
            "future_measurement_pair_valid": self.measurement_pair_valid.clone(),
            "group_valid": typed.future.latent_target_valid,
            "world_action_group_valid": torch.tensor(world_action_valid_rows, dtype=torch.bool),
        }
        return result


__all__ = ["PredictiveWorldCollator", "PredictiveWorldTrainingCollator"]
