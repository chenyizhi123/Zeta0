"""Versioned RGB geometry shared by cache, training and serving.

Profiles describe model inputs, not machine resources. Existing checkpoints
default to legacy_v1; new runs explicitly select aspect_v1 in their provenance.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_PREPROCESSING_VERSIONS = ("legacy_v1", "aspect_v1")


def image_preprocessing_version(provenance: dict) -> str:
    value = provenance.get("image_preprocessing")
    if value is None:
        return "legacy_v1"
    if not isinstance(value, dict) or value != image_preprocessing_spec(value.get("version", "")):
        raise ValueError("image preprocessing provenance is unsupported or inconsistent")
    return value["version"]


def image_preprocessing_spec(version: str = "legacy_v1") -> dict:
    if version == "legacy_v1":
        return {"version": version, "resize_hw": [240, 320], "interpolation": "pil_nearest",
                "pool": "adaptive_average_4x4"}
    if version != "aspect_v1":
        raise ValueError(f"unsupported image preprocessing: {version!r}")
    return {
        "version": version, "short_edge": 240, "max_visual_tokens_per_image": 160,
        "spatial_factor": 32, "allow_upscale": False,
        "interpolation": "torch_cpu_bilinear_antialias_round_uint8_v1",
        "padding": "symmetric_rgb_127", "processor_do_resize": False,
        "pool": "content_area_overlap_4x4_v1",
    }


@lru_cache(maxsize=512)
def image_geometry(height: int, width: int, version: str = "aspect_v1") -> dict:
    spec = image_preprocessing_spec(version)
    if height <= 0 or width <= 0:
        raise ValueError("RGB dimensions must be positive")
    if version == "legacy_v1":
        return {"source_hw": [height, width], "content_box": [0, 0, 240, 320],
                "canvas_hw": [240, 320]}
    factor, cap = spec["spatial_factor"], spec["max_visual_tokens_per_image"]
    # Maximize retained resolution at the requested short edge, accounting
    # for alignment padding in the token ceiling. Never crop or split a view.
    scale = min(1.0, spec["short_edge"] / min(height, width))

    def fit(value):
        h, w = max(1, round(height * value)), max(1, round(width * value))
        ch, cw = math.ceil(h / factor) * factor, math.ceil(w / factor) * factor
        return h, w, ch, cw

    if fit(scale)[2] * fit(scale)[3] > cap * factor**2:
        low, high = 0.0, scale
        for _ in range(48):
            middle = (low + high) / 2
            _, _, ch, cw = fit(middle)
            if ch * cw <= cap * factor**2:
                low = middle
            else:
                high = middle
        scale = low
    h, w, ch, cw = fit(scale)
    top, left = (ch - h) // 2, (cw - w) // 2
    return {"source_hw": [height, width], "content_box": [top, left, top + h, left + w],
            "canvas_hw": [ch, cw]}


def aspect_preprocess_batch(images: torch.Tensor, *, version: str = "aspect_v1") -> torch.Tensor:
    if version != "aspect_v1":
        raise ValueError("aspect_preprocess_batch requires aspect_v1")
    images = torch.as_tensor(images)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("source images must be [B,3,H,W]")
    if images.dtype == torch.uint8:
        pixels = images.cpu()
    elif images.is_floating_point():
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("source RGB images must be finite in [0,1]")
        pixels = torch.floor(images.cpu().float() * 255).to(torch.uint8)
    else:
        raise ValueError("source RGB images must be uint8 or floating point [0,1]")
    geometry = image_geometry(*pixels.shape[-2:], version)
    top, left, bottom, right = geometry["content_box"]
    size = (bottom - top, right - left)
    if tuple(pixels.shape[-2:]) != size:
        # One CPU implementation makes PIL, batched cache and serving bytes
        # identical, including rounding at interpolation boundaries.
        pixels = F.interpolate(pixels.float(), size=size, mode="bilinear",
                               align_corners=False, antialias=True).round().clamp(0, 255).to(torch.uint8)
    ch, cw = geometry["canvas_hw"]
    return F.pad(pixels, (left, cw - right, top, ch - bottom), value=127).contiguous()


def aspect_preprocess_image(image, *, version: str = "aspect_v1") -> Image.Image:
    if isinstance(image, Image.Image):
        pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    else:
        pixels = torch.as_tensor(image)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("source RGB image must be [3,H,W]")
    geometry = image_geometry(*pixels.shape[-2:], version)
    result = aspect_preprocess_batch(pixels.unsqueeze(0), version=version)[0]
    pil = Image.fromarray(result.permute(1, 2, 0).numpy())
    pil.info["action_aligned_geometry"] = geometry
    return pil


def content_boxes(images: list[Image.Image]) -> torch.Tensor:
    return torch.tensor([image.info["action_aligned_geometry"]["content_box"]
                         for image in images], dtype=torch.float32)


def process_aspect_qwen(processor, *, images, texts=None):
    """One image-processing pass; aligned canvases must not be resized again."""
    if texts is None:
        result = processor.image_processor(images=images, do_resize=False, return_tensors="pt")
    else:
        result = processor(text=texts, images=images, padding=True, return_tensors="pt",
                           images_kwargs={"do_resize": False})
    shapes = ([tuple(images.shape[-2:])] * len(images) if isinstance(images, torch.Tensor)
              else [(image.height, image.width) for image in images])
    expected = torch.tensor([[1, h // 16, w // 16] for h, w in shapes])
    if not torch.equal(torch.as_tensor(result["image_grid_thw"]).cpu(), expected):
        raise ValueError("Qwen changed the aspect_v1 image grid; refusing incompatible preprocessing")
    return result
