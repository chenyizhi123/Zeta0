"""Stable frozen Qwen3-VL visual-feature ABI for action-aligned prediction.

The action-aligned route predicts a future feature in the *same* VLM space
that Ψ0 already consumes.  Qwen3-VL's supported public visual interface is
``get_image_features``: it returns post-merger image embeddings with exactly
the width injected into the language model.  The pre-merger 1024-wide vision
states are intentionally not exposed here: reaching into individual visual
blocks would make a checkpoint/version-specific implementation detail part of
the training and deployment ABI.

For the pinned Qwen3-VL-2B Ψ0 backbone, a 240x320 policy image produces an
8x10 post-merger field of width 2048.  This module spatially pools that field
to a fixed row-major 4x4 field, ``[B, 16, 2048]``.  It contains no model hooks
and no prediction logic, so the same pure pooling operation can be used for:

* offline future-target cache construction after ``get_image_features``; and
* the current-image field captured from Ψ0's already-running VLM forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


VLM_TARGET_TOKENS = 16
VLM_TARGET_DIM = 2048
VLM_TARGET_GRID = (4, 4)
VLM_TOKEN_LAYOUT = "qwen3vl_post_merger_pool_4x4_row_major"


@dataclass(frozen=True)
class Qwen3VLVisualFeatureSpec:
    """Pinned properties of the public Qwen3-VL visual-output interface."""

    model_type: str
    vision_patch_size: int
    vision_temporal_patch_size: int
    spatial_merge_size: int
    vision_hidden_size: int
    output_hidden_size: int
    output_grid: tuple[int, int] = VLM_TARGET_GRID
    token_layout: str = VLM_TOKEN_LAYOUT

    def manifest_fields(self) -> dict[str, Any]:
        """Return canonical JSON-ready fields for cache provenance."""

        fields = asdict(self)
        fields["output_grid"] = list(self.output_grid)
        return fields


def resolve_qwen3vl_visual_feature_spec(
    model_or_config: Any,
    *,
    expected_output_hidden_size: int = VLM_TARGET_DIM,
) -> Qwen3VLVisualFeatureSpec:
    """Validate the supported Qwen3-VL post-merger feature ABI.

    ``model_or_config`` may be a conditional-generation model, its top-level
    config, or a config-like test double.  The canonical action-aligned
    checkpoint is intentionally strict: the cached target width must match the
    native embeddings supplied to Ψ0's language model, not an auxiliary
    learned projection.
    """

    config = getattr(model_or_config, "config", model_or_config)
    vision = getattr(config, "vision_config", None)
    if vision is None:
        raise ValueError("Qwen3-VL visual feature ABI requires config.vision_config")
    model_type = str(getattr(config, "model_type", ""))
    vision_model_type = str(getattr(vision, "model_type", ""))
    if model_type != "qwen3_vl" or vision_model_type != "qwen3_vl":
        raise ValueError(
            "action-aligned VLM targets require Qwen3-VL, got "
            f"top-level/vision model types {model_type!r}/{vision_model_type!r}"
        )
    patch_size = int(getattr(vision, "patch_size", 0))
    temporal_patch_size = int(getattr(vision, "temporal_patch_size", 0))
    merge_size = int(getattr(vision, "spatial_merge_size", 0))
    vision_hidden_size = int(getattr(vision, "hidden_size", 0))
    output_hidden_size = int(getattr(vision, "out_hidden_size", 0))
    expected = (
        patch_size,
        temporal_patch_size,
        merge_size,
        vision_hidden_size,
        output_hidden_size,
    )
    required = (16, 2, 2, 1024, int(expected_output_hidden_size))
    if expected != required:
        raise ValueError(
            "unsupported Qwen3-VL visual target ABI; expected "
            "(patch, temporal_patch, merge, native_width, output_width)="
            f"{required}, got {expected}"
        )
    return Qwen3VLVisualFeatureSpec(
        model_type=model_type,
        vision_patch_size=patch_size,
        vision_temporal_patch_size=temporal_patch_size,
        spatial_merge_size=merge_size,
        vision_hidden_size=vision_hidden_size,
        output_hidden_size=output_hidden_size,
    )


def endpoint_vlm_feature_shape(num_frames: int) -> tuple[int, int, int]:
    """Return the canonical frame-major cache shape ``[N,16,2048]``."""

    return int(num_frames), VLM_TARGET_TOKENS, VLM_TARGET_DIM


def pool_qwen3vl_post_merger_image_tokens(
    image_embeds: Sequence[torch.Tensor],
    image_grid_thw: torch.Tensor,
    *,
    spec: Qwen3VLVisualFeatureSpec,
    image_content_boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool public Qwen3-VL per-image outputs to ``[B,16,2048]``.

    ``Qwen3VLForConditionalGeneration.get_image_features`` returns one
    flattened post-merger token field per image.  The matching ``grid_thw``
    records its pre-merger ``[T,H,W]`` geometry.  The action-aligned route
    permits exactly one static image per sample, so this function treats each
    item as one batch row and rejects videos/multi-image ambiguity.
    """

    grid = torch.as_tensor(image_grid_thw)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape [num_images,3], got "
            f"{tuple(grid.shape)}"
        )
    if len(image_embeds) != int(grid.shape[0]):
        raise ValueError(
            "Qwen image embedding count must equal image_grid_thw rows, got "
            f"{len(image_embeds)} and {grid.shape[0]}"
        )
    if not image_embeds:
        raise ValueError("at least one Qwen image embedding is required")
    boxes = None if image_content_boxes is None else torch.as_tensor(image_content_boxes).detach().cpu()
    if boxes is not None and (boxes.shape != (len(image_embeds), 4) or not torch.isfinite(boxes).all()):
        raise ValueError("image_content_boxes must be finite [num_images,4] pixel bounds")

    pooled_rows: list[torch.Tensor] = []
    merge = int(spec.spatial_merge_size)
    for index, (tokens, geometry) in enumerate(zip(image_embeds, grid.detach().cpu())):
        geometry_values = [int(value) for value in geometry.tolist()]
        temporal, premerge_h, premerge_w = geometry_values
        if temporal != 1:
            raise ValueError(
                "action-aligned VLM target accepts one static image per row; "
                f"image {index} has temporal grid {temporal}"
            )
        if premerge_h <= 0 or premerge_w <= 0 or premerge_h % merge or premerge_w % merge:
            raise ValueError(
                f"invalid Qwen pre-merger grid for image {index}: {geometry_values}; "
                f"H/W must be positive multiples of merge={merge}"
            )
        if tokens.ndim != 2 or int(tokens.shape[1]) != int(spec.output_hidden_size):
            raise ValueError(
                "Qwen post-merger tokens must have shape [N,2048], got "
                f"{tuple(tokens.shape)} for image {index}"
            )
        merged_h, merged_w = premerge_h // merge, premerge_w // merge
        expected_tokens = merged_h * merged_w
        if int(tokens.shape[0]) != expected_tokens:
            raise ValueError(
                f"Qwen post-merger token count {tokens.shape[0]} does not match "
                f"grid {merged_h}x{merged_w} for image {index}"
            )
        # Qwen's public post-merger order follows its merged spatial grid.
        # Preserve row-major ordering after the 2D adaptive average pool.
        field = tokens.transpose(0, 1).reshape(
            1, int(spec.output_hidden_size), merged_h, merged_w
        )
        if boxes is None:
            pooled = F.adaptive_avg_pool2d(field, spec.output_grid)
        else:
            # Pool over content coordinates rather than the padded canvas.
            # Each axis weight is the physical overlap between a Qwen cell
            # and an output bin; fully padded cells have exactly zero weight.
            top, left, bottom, right = boxes[index].tolist()
            factor = spec.vision_patch_size * merge
            if not (0 <= top < bottom <= merged_h * factor
                    and 0 <= left < right <= merged_w * factor):
                raise ValueError("image content bounds lie outside the Qwen canvas")

            def overlap_weights(start, end, cells, outputs):
                edges = torch.linspace(start / factor, end / factor, outputs + 1,
                                       device=field.device, dtype=torch.float32)
                cells_start = torch.arange(cells, device=field.device, dtype=torch.float32)
                weights = (torch.minimum(edges[1:, None], cells_start[None, :] + 1)
                           - torch.maximum(edges[:-1, None], cells_start[None, :])).clamp_min(0)
                return weights / weights.sum(dim=1, keepdim=True)

            wh = overlap_weights(top, bottom, merged_h, spec.output_grid[0])
            ww = overlap_weights(left, right, merged_w, spec.output_grid[1])
            with torch.autocast(device_type=field.device.type, enabled=False):
                pooled = torch.einsum("oh,bchw,pw->bcop", wh, field.float(), ww).to(field.dtype)
        pooled_rows.append(pooled.flatten(2).transpose(1, 2).squeeze(0))

    result = torch.stack(pooled_rows, dim=0).contiguous()
    expected_shape = (len(image_embeds), VLM_TARGET_TOKENS, int(spec.output_hidden_size))
    if tuple(result.shape) != expected_shape:
        raise RuntimeError(
            "Qwen visual pooling produced an invalid endpoint field: "
            f"{tuple(result.shape)} != {expected_shape}"
        )
    return result


@torch.inference_mode()
def extract_qwen3vl_post_merger_image_features(
    vlm_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spec: Qwen3VLVisualFeatureSpec | None = None,
    image_content_boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one public Qwen visual forward and return ``[B,16,2048]``.

    This is intended for the offline cache builder.  Online Ψ0 instead
    captures the same post-merger field from its existing full VLM forward and
    calls :func:`pool_qwen3vl_post_merger_image_tokens`, avoiding a second
    visual pass.
    """

    if spec is None:
        spec = resolve_qwen3vl_visual_feature_spec(vlm_model)
    get_image_features = getattr(vlm_model, "get_image_features", None)
    if not callable(get_image_features):
        raise TypeError("VLM must expose Qwen3-VL get_image_features")
    result = get_image_features(pixel_values, image_grid_thw)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Qwen get_image_features must return (image_embeds, deepstack)")
    image_embeds, _deepstack = result
    if not isinstance(image_embeds, (list, tuple)):
        raise RuntimeError("Qwen get_image_features returned non-sequence image embeddings")
    return pool_qwen3vl_post_merger_image_tokens(
        image_embeds, image_grid_thw, spec=spec, image_content_boxes=image_content_boxes
    )


__all__ = [
    "VLM_TARGET_DIM",
    "VLM_TARGET_GRID",
    "VLM_TARGET_TOKENS",
    "VLM_TOKEN_LAYOUT",
    "Qwen3VLVisualFeatureSpec",
    "endpoint_vlm_feature_shape",
    "extract_qwen3vl_post_merger_image_features",
    "pool_qwen3vl_post_merger_image_tokens",
    "resolve_qwen3vl_visual_feature_spec",
]
