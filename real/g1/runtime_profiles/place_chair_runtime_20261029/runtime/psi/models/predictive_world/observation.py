"""Frozen Qwen observation extraction shared by training, cache and deployment."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from psi.data.predictive_world.contracts import ObservedHistory, assert_observed_only
from psi.models.action_aligned_vlm_features import (
    pool_qwen3vl_post_merger_image_tokens,
    resolve_qwen3vl_visual_feature_spec,
)
from psi.models.predictive_world.policy import ObservationFeatures


@torch.inference_mode()
def extract_qwen3vl_ragged_patch_fields(
    vlm_model: nn.Module,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
) -> tuple[list[Tensor], list[tuple[int, int]]]:
    """Return public post-merger patch fields without spatial pooling."""

    spec = resolve_qwen3vl_visual_feature_spec(vlm_model)
    result = vlm_model.get_image_features(pixel_values, image_grid_thw)
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], (list, tuple)):
        raise RuntimeError("Qwen get_image_features returned an unsupported result")
    fields, grids = [], []
    if len(result[0]) != image_grid_thw.shape[0]:
        raise ValueError("Qwen image output count disagrees with image_grid_thw")
    for tokens, geometry in zip(result[0], image_grid_thw.detach().cpu().tolist()):
        temporal, raw_h, raw_w = (int(value) for value in geometry)
        merge = spec.spatial_merge_size
        if temporal != 1 or raw_h % merge or raw_w % merge:
            raise ValueError("predictive-world cache requires static merge-aligned images")
        grid = (raw_h // merge, raw_w // merge)
        if tokens.shape != (grid[0] * grid[1], spec.output_hidden_size):
            raise ValueError("Qwen patch field shape disagrees with its grid")
        if not torch.isfinite(tokens).all():
            raise ValueError("Qwen produced non-finite visual patches")
        fields.append(tokens.detach())
        grids.append(grid)
    return fields, grids


class FrozenQwenObservationEncoder(nn.Module):
    """Run frozen Qwen once and expose causal language plus current visual tokens."""

    def __init__(self, vlm_model: nn.Module) -> None:
        super().__init__()
        self.vlm_model = vlm_model
        self.feature_spec = resolve_qwen3vl_visual_feature_spec(vlm_model)
        for parameter in self.vlm_model.parameters():
            parameter.requires_grad_(False)
        self.vlm_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vlm_model.eval()
        return self

    def forward(self, observed: ObservedHistory) -> ObservationFeatures:
        observed = assert_observed_only(observed)
        inputs = dict(observed.qwen_inputs)
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        missing = sorted(required - inputs.keys())
        if missing:
            raise ValueError(f"Qwen observed inputs are missing: {missing}")
        visual = getattr(getattr(self.vlm_model, "model", None), "visual", None)
        if not isinstance(visual, nn.Module):
            raise RuntimeError("Qwen model must expose model.visual")
        captured: list[Tensor] = []

        def capture(_module, _arguments, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, Tensor):
                raise RuntimeError("Qwen visual module returned no tensor")
            captured.append(value)

        handle = visual.register_forward_hook(capture)
        try:
            with torch.no_grad():
                backbone = self.vlm_model.model
                output = backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Qwen visual module must execute exactly once")
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, Tensor):
            raise RuntimeError("Qwen backbone returned no final hidden state")
        grid = inputs["image_grid_thw"]
        merge = self.feature_spec.spatial_merge_size
        fields, offset = [], 0
        for temporal, raw_h, raw_w in grid.detach().cpu().tolist():
            temporal, raw_h, raw_w = int(temporal), int(raw_h), int(raw_w)
            if temporal != 1 or raw_h % merge or raw_w % merge:
                raise ValueError("observed Qwen images must be static and merge-aligned")
            count = raw_h // merge * (raw_w // merge)
            fields.append(captured[0][offset : offset + count])
            offset += count
        if offset != captured[0].shape[0]:
            raise ValueError("captured Qwen patches disagree with image geometry")
        batch = hidden.shape[0]
        counts = inputs.get("image_counts", torch.ones(batch, dtype=torch.long))
        counts = torch.as_tensor(counts).detach().cpu().long()
        if counts.shape != (batch,) or int(counts.sum()) != len(fields) or (counts <= 0).any():
            raise ValueError("image_counts must partition all observed images")
        starts = counts.cumsum(0) - counts
        primary = inputs.get("primary_current_image_indices", starts)
        primary = torch.as_tensor(primary).detach().cpu().long()
        if primary.shape != (batch,) or ((primary < starts) | (primary >= starts + counts)).any():
            raise ValueError("primary current image index is outside its sample")
        boxes = inputs.get("image_content_boxes")
        current = pool_qwen3vl_post_merger_image_tokens(
            [fields[index] for index in primary.tolist()],
            grid.index_select(0, primary.to(grid.device)),
            spec=self.feature_spec,
            image_content_boxes=(
                None if boxes is None else boxes.index_select(0, primary.to(boxes.device))
            ),
        )
        return ObservationFeatures(
            qwen_context=hidden.detach(),
            context_valid=inputs["attention_mask"].bool(),
            current_visual=current.detach(),
        )


__all__ = ["FrozenQwenObservationEncoder", "extract_qwen3vl_ragged_patch_fields"]
