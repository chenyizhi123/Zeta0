"""Frame-major frozen-Qwen3-VL endpoint target cache.

The cache contains one post-merger Ψ0-VLM spatial field per dense LeRobot
row.  Both current and future visual representations are therefore in the
same frozen Qwen3-VL coordinate system; action-aligned training never starts
or depends on a separate auxiliary vision process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
import torch
from PIL import Image
from psi.models.action_aligned_image import (
    image_preprocessing_spec, image_geometry, aspect_preprocess_image,
    aspect_preprocess_batch, process_aspect_qwen,
)

from psi.models.action_aligned_vlm_features import (
    VLM_TARGET_DIM,
    VLM_TARGET_GRID,
    VLM_TARGET_TOKENS,
    VLM_TOKEN_LAYOUT,
    Qwen3VLVisualFeatureSpec,
    endpoint_vlm_feature_shape,
    extract_qwen3vl_post_merger_image_features,
    resolve_qwen3vl_visual_feature_spec,
)


CACHE_FORMAT_VERSION = 2
CACHE_FORMAT = "action_aligned_vlm_endpoint_v2"
CACHE_LAYOUT = "frame_major_npy"
CACHE_INDEX_LAYOUT = "lerobot_dataset_row"
CACHE_MANIFEST_FILENAME = "action_aligned_vlm_endpoint_manifest.json"
FEATURE_PREFIX = "action_aligned_vlm_endpoint_features."
INTENT_TARGET_MODE = "future_endpoint_spatial"

# The action-aligned policy transform is explicitly pinned to this geometry.
# Qwen's own image processor then deterministically creates its patch grid.
POLICY_RESIZE = (240, 320)
POLICY_CENTER_CROP = (240, 320)
CANONICAL_VLM_FEATURE_SPEC = {
    "model_type": "qwen3_vl",
    "vision_patch_size": 16,
    "vision_temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "vision_hidden_size": 1024,
    "output_hidden_size": VLM_TARGET_DIM,
    "output_grid": list(VLM_TARGET_GRID),
    "token_layout": VLM_TOKEN_LAYOUT,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_action_aligned_vlm_checkpoint_fingerprint(
    model_name_or_path: str | os.PathLike[str],
    config: Any,
    *,
    include_content_hash: bool = True,
) -> dict[str, Any]:
    """Describe the frozen Qwen3-VL checkpoint used for endpoint targets.

    Cache construction records a content digest for every weight shard.  A
    distributed training job, however, opens both train and validation
    datasets on every rank; re-hashing the same multi-gigabyte frozen VLM four
    times is pure startup I/O and can make the data path look broken.  The
    lightweight form retains the exact file set, byte size and nanosecond
    mtime, then :func:`vlm_checkpoint_fingerprints_match` compares that fast
    identity against the content-addressed manifest.  A changed file therefore
    still fails closed without repeatedly scanning its contents.
    """

    root = Path(model_name_or_path).expanduser()
    # ``PretrainedConfig.to_dict()`` includes volatile runtime metadata such
    # as ``transformers_version`` and ``_name_or_path``.  The cache builder
    # and the training venv can legitimately use different patch versions of
    # Transformers, while loading the same pinned checkpoint.  Hash the
    # checkpoint's canonical config.json instead; this is the actual on-disk
    # model ABI and is stable across those environments.  Hub-only callers
    # retain a scrubbed config fallback until a local snapshot is resolved.
    config_path = root / "config.json" if root.is_dir() else root.parent / "config.json"
    config_payload: Any | None = None
    if config_path.is_file():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid VLM config.json: {config_path}") from error
    if config_payload is None:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise ValueError("VLM config must provide to_dict() for cache provenance")

        def _scrub_runtime_metadata(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: _scrub_runtime_metadata(item)
                    for key, item in value.items()
                    if key not in {"_name_or_path", "transformers_version"}
                }
            if isinstance(value, list):
                return [_scrub_runtime_metadata(item) for item in value]
            return value

        config_payload = _scrub_runtime_metadata(to_dict())
    config_sha = sha256_bytes(canonical_json_bytes(config_payload))
    shards: dict[str, dict[str, Any]] = {}
    if root.is_file():
        candidates = [root]
        base = root.parent
    elif root.is_dir():
        base = root
        candidates = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
        )
    else:
        # A Hub ID may not have a local snapshot when this function is called.
        # Keep that fact explicit rather than constructing a path-dependent ID.
        base = root
        candidates = []
    for path in candidates:
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            relative = path.name
        stat = path.stat()
        shard: dict[str, Any] = {
            "byte_size": int(path.stat().st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if include_content_hash:
            shard["sha256"] = sha256_file(path)
        shards[relative] = shard
    return {
        "method": "resolved_commit+config_file_sha256+weight_shards_content_stat_v3",
        "resolved_commit": getattr(config, "_commit_hash", None),
        "config_sha256": config_sha,
        "weight_shards": dict(sorted(shards.items())),
    }


def vlm_checkpoint_fingerprints_match(
    stored: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    """Compare a content-addressed manifest to a fast local checkpoint view.

    ``stored`` is always required to contain SHA-256 digests because it was
    emitted by the cache builder.  ``expected`` may omit them during training
    so opening a cache never causes one content scan per rank/dataset split.
    If a caller does provide expected SHA-256 values, they are checked too.
    """

    if not isinstance(stored, Mapping):
        return False
    for key in ("method", "resolved_commit", "config_sha256"):
        if stored.get(key) != expected.get(key):
            return False
    stored_shards = stored.get("weight_shards")
    expected_shards = expected.get("weight_shards")
    if not isinstance(stored_shards, Mapping) or not isinstance(expected_shards, Mapping):
        return False
    if set(stored_shards) != set(expected_shards):
        return False
    for name, expected_shard in expected_shards.items():
        stored_shard = stored_shards.get(name)
        if not isinstance(stored_shard, Mapping) or not isinstance(expected_shard, Mapping):
            return False
        if not isinstance(stored_shard.get("sha256"), str) or not stored_shard["sha256"]:
            return False
        for key in ("byte_size", "mtime_ns"):
            if stored_shard.get(key) != expected_shard.get(key):
                return False
        if "sha256" in expected_shard and stored_shard.get("sha256") != expected_shard["sha256"]:
            return False
    return True


def build_action_aligned_vlm_processor_fingerprint(processor: Any) -> dict[str, Any]:
    """Bind Qwen image configuration to the loaded preprocessing runtime."""

    image_processor = getattr(processor, "image_processor", None)
    to_json_string = getattr(image_processor, "to_json_string", None)
    if image_processor is None or not callable(to_json_string):
        raise ValueError("Qwen VLM processor must expose image_processor.to_json_string()")
    try:
        processor_config = json.loads(to_json_string())
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Qwen VLM image processor did not produce JSON") from error
    return {
        "method": "qwen_image_processor_json+image_runtime_v2",
        "image_runtime": {
            "pillow": PIL.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "transformers_version": importlib.metadata.version("transformers"),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "image_processor_config_sha256": sha256_bytes(
            canonical_json_bytes(processor_config)
        ),
    }


def _int_vector(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    return arr.astype(np.int64, copy=False)


def normalize_episode_data_index(
    episode_data_index: Mapping[str, Any], *, num_frames: int
) -> dict[str, list[int]]:
    """Validate contiguous end-exclusive episode ranges."""

    starts = _int_vector(episode_data_index.get("from"), "episode_data_index.from")
    ends = _int_vector(episode_data_index.get("to"), "episode_data_index.to")
    if starts.size == 0 or starts.shape != ends.shape:
        raise ValueError("episode_data_index must contain non-empty from/to vectors")
    if starts[0] != 0 or ends[-1] != int(num_frames):
        raise ValueError("episode_data_index must cover every dataset row")
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("episode ranges must be positive and ordered")
    if starts.size > 1 and np.any(starts[1:] != ends[:-1]):
        raise ValueError("episode ranges must be contiguous and non-overlapping")
    return {"from": starts.tolist(), "to": ends.tolist()}


def endpoint_row_index(
    row_index: int, horizon: int, episode_data_index: Mapping[str, Any]
) -> int | None:
    """Return exact ``row+horizon`` if it stays in the source episode."""

    starts = _int_vector(episode_data_index["from"], "episode_data_index.from")
    ends = _int_vector(episode_data_index["to"], "episode_data_index.to")
    row_index, horizon = int(row_index), int(horizon)
    if horizon < 0 or starts.size != ends.size:
        raise ValueError("horizon must be non-negative and episode index valid")
    if row_index < 0 or row_index >= int(ends[-1]):
        raise IndexError(f"row_index {row_index} outside dataset")
    slot = int(np.searchsorted(ends, row_index, side="right"))
    target = row_index + horizon
    return target if target < int(ends[slot]) else None


def preprocess_endpoint_vlm_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    *,
    policy_resize: Sequence[int] = POLICY_RESIZE,
    policy_center_crop: Sequence[int] = POLICY_CENTER_CROP,
    image_preprocessing: str = "legacy_v1",
) -> Image.Image:
    """Apply the same deterministic policy image resize/crop as Ψ0 training.

    The returned PIL image is intentionally passed directly to the frozen
    Qwen image processor.  Its own resize/normalization/patching configuration
    is recorded in the cache manifest instead of being reimplemented here.
    """

    image_preprocessing_spec(image_preprocessing)
    if image_preprocessing == "aspect_v1":
        return aspect_preprocess_image(image)
    resize = tuple(int(value) for value in policy_resize)
    crop = tuple(int(value) for value in policy_center_crop)
    if resize != POLICY_RESIZE or crop != POLICY_CENTER_CROP:
        raise ValueError(
            "action-aligned VLM target preprocessing is fixed to policy "
            "resize/crop [240,320]"
        )
    if isinstance(image, Image.Image):
        # LeRobot repack and deployment already expose PIL images.  Keep this
        # branch byte-exact with the tensor branch below; in particular, do
        # not route it through torchvision Resize, whose nearest-neighbour
        # pixel-center convention differs from PIL for 480x640 -> 240x320.
        pil = image.convert("RGB")
    else:
        image = torch.as_tensor(image)
        if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
            raise ValueError("source RGB image must be floating-point [3,H,W]")
        image = image.float()
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("source RGB image must be finite in [0,1]")
        quantized = torch.floor(image * 255.0).to(torch.uint8)
        pil = Image.fromarray(quantized.permute(1, 2, 0).cpu().numpy())
    resize_h, resize_w = resize
    crop_h, crop_w = crop
    pil = pil.resize((resize_w, resize_h), resample=Image.Resampling.NEAREST)
    left = max((resize_w - crop_w) // 2, 0)
    top = max((resize_h - crop_h) // 2, 0)
    return pil.crop((left, top, left + crop_w, top + crop_h))


def preprocess_endpoint_vlm_images(
    images: torch.Tensor | np.ndarray,
    *,
    device: torch.device | str | None = None,
    policy_resize: Sequence[int] = POLICY_RESIZE,
    policy_center_crop: Sequence[int] = POLICY_CENTER_CROP,
    image_preprocessing: str = "legacy_v1",
) -> torch.Tensor:
    """Vectorized exact counterpart of :func:`preprocess_endpoint_vlm_image`.

    The original PIL helper is retained as the human-readable reference and
    for single-image tests.  Cache construction uses this batched tensor path:
    it performs Qwen's policy resize on the GPU with ``nearest-exact`` (which
    is pixel-identical to PIL's nearest-neighbour result for our source
    frames), then passes one `[B,3,240,320]` uint8 batch to Qwen's fast image
    processor.  This removes a Python/PIL loop from every video batch.
    """

    image_preprocessing_spec(image_preprocessing)
    if image_preprocessing == "aspect_v1":
        result = aspect_preprocess_batch(torch.as_tensor(images))
        return result if device is None else result.to(device, non_blocking=True)
    resize = tuple(int(value) for value in policy_resize)
    crop = tuple(int(value) for value in policy_center_crop)
    if resize != POLICY_RESIZE or crop != POLICY_CENTER_CROP:
        raise ValueError(
            "action-aligned VLM target preprocessing is fixed to policy "
            "resize/crop [240,320]"
        )
    tensor = torch.as_tensor(images)
    if tensor.ndim != 4 or tensor.shape[1] != 3 or not tensor.is_floating_point():
        raise ValueError("source RGB images must be floating-point [B,3,H,W]")
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("source RGB images must be finite in [0,1]")
    if device is not None:
        tensor = tensor.to(device=device, non_blocking=True)
    quantized = torch.floor(tensor.float() * 255.0).to(torch.uint8)
    resize_h, resize_w = resize
    if tuple(quantized.shape[-2:]) != resize:
        # ``nearest-exact`` is deliberately not plain ``nearest``: its pixel
        # centers match PIL.Image.Resampling.NEAREST exactly.
        quantized = torch.nn.functional.interpolate(
            quantized.float(), size=resize, mode="nearest-exact"
        ).to(torch.uint8)
    crop_h, crop_w = crop
    left = max((resize_w - crop_w) // 2, 0)
    top = max((resize_h - crop_h) // 2, 0)
    result = quantized[..., top : top + crop_h, left : left + crop_w].contiguous()
    if tuple(result.shape[1:]) != (3, crop_h, crop_w):
        raise RuntimeError(
            "batched VLM policy preprocessing produced invalid shape "
            f"{tuple(result.shape)}"
        )
    return result


@torch.inference_mode()
def extract_endpoint_vlm_features(
    vlm_model: torch.nn.Module,
    processor: Any,
    images: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    spec: Qwen3VLVisualFeatureSpec | None = None,
    policy_resize: Sequence[int] = POLICY_RESIZE,
    policy_center_crop: Sequence[int] = POLICY_CENTER_CROP,
    image_preprocessing: str = "legacy_v1",
) -> torch.Tensor:
    """Extract raw fp16 ``[B,16,2048]`` fields with one Qwen visual pass."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if spec is None:
        spec = resolve_qwen3vl_visual_feature_spec(vlm_model)
    policy_images = preprocess_endpoint_vlm_images(
        images,
        device=device,
        policy_resize=policy_resize,
        policy_center_crop=policy_center_crop,
        image_preprocessing=image_preprocessing,
    )
    # The visual processor is sufficient here: cache extraction has no text
    # condition, and bypassing Qwen's tokenizer avoids per-row Python work.
    # It accepts a batched uint8 tensor and preserves the exact output of
    # ``processor(text=[""], images=[PIL...])`` (covered by regression test).
    image_processor = getattr(processor, "image_processor", None)
    if not callable(image_processor):
        raise TypeError("Qwen VLM processor must expose a callable image_processor")
    boxes = None
    if image_preprocessing == "aspect_v1":
        processor_inputs = process_aspect_qwen(processor, images=policy_images)
        boxes = torch.tensor([image_geometry(*images.shape[-2:])["content_box"]] * len(images))
    else:
        processor_inputs = image_processor(images=policy_images, return_tensors="pt")
    pixel_values = torch.as_tensor(processor_inputs["pixel_values"])
    image_grid_thw = torch.as_tensor(processor_inputs["image_grid_thw"])
    if device is not None and pixel_values.device.type != torch.device(device).type:
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)
    pooled = extract_qwen3vl_post_merger_image_features(
        vlm_model,
        pixel_values,
        image_grid_thw,
        spec=spec,
        image_content_boxes=boxes,
    )
    expected = (len(policy_images), VLM_TARGET_TOKENS, VLM_TARGET_DIM)
    if tuple(pooled.shape) != expected:
        raise RuntimeError(
            "Qwen endpoint extractor must return [B,16,2048], got "
            f"{tuple(pooled.shape)}"
        )
    if not torch.isfinite(pooled).all():
        raise RuntimeError("Qwen endpoint extractor returned NaN or infinity")
    return pooled.to(dtype=torch.float16).cpu()


def build_action_aligned_endpoint_manifest(
    *,
    num_frames: int,
    episode_data_index: Mapping[str, Any],
    primary_image_key: str,
    vlm_model: str,
    vlm_feature_spec: Qwen3VLVisualFeatureSpec | Mapping[str, Any],
    dataset_fingerprint: Mapping[str, Any] | None = None,
    vlm_checkpoint_fingerprint: Mapping[str, Any] | None = None,
    vlm_processor_fingerprint: Mapping[str, Any] | None = None,
    feature_file: str | None = None,
    feature_nbytes: int | None = None,
    feature_sha256: str | None = None,
    feature_mtime_ns: int | None = None,
    image_preprocessing: str = "legacy_v1",
) -> dict[str, Any]:
    """Construct the canonical VLM cache manifest; integrity fields are required."""

    num_frames = int(num_frames)
    episode_index = normalize_episode_data_index(episode_data_index, num_frames=num_frames)
    if not primary_image_key:
        raise ValueError("primary_image_key is required")
    if not vlm_model:
        raise ValueError("vlm_model is required")
    if isinstance(vlm_feature_spec, Qwen3VLVisualFeatureSpec):
        feature_spec = vlm_feature_spec.manifest_fields()
    else:
        feature_spec = dict(vlm_feature_spec)
    # Keep this comparison explicit so a cache cannot claim to be VLM-based
    # while quietly substituting an auxiliary visual coordinate system.
    if feature_spec != CANONICAL_VLM_FEATURE_SPEC:
        raise ValueError(
            "unsupported action-aligned VLM feature spec: "
            f"expected {CANONICAL_VLM_FEATURE_SPEC}, got {feature_spec}"
        )
    if feature_file is None:
        feature_file = f"{FEATURE_PREFIX}{uuid.uuid4().hex}.npy"
    path = Path(feature_file)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.name.startswith(FEATURE_PREFIX)
        or path.suffix != ".npy"
    ):
        raise ValueError("feature_file must be a unique local VLM feature name")
    if feature_nbytes is None or feature_sha256 is None:
        raise ValueError("feature_nbytes and feature_sha256 are required manifest fields")
    if not vlm_checkpoint_fingerprint:
        raise ValueError("vlm_checkpoint_fingerprint is required manifest provenance")
    if not vlm_processor_fingerprint:
        raise ValueError("vlm_processor_fingerprint is required manifest provenance")
    geometry_spec = image_preprocessing_spec(image_preprocessing)
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "format": CACHE_FORMAT,
        "layout": CACHE_LAYOUT,
        "index_layout": CACHE_INDEX_LAYOUT,
        "feature_file": path.name,
        "feature_nbytes": int(feature_nbytes),
        "feature_sha256": str(feature_sha256),
        "feature_mtime_ns": (
            int(feature_mtime_ns) if feature_mtime_ns is not None else None
        ),
        "token_layout": VLM_TOKEN_LAYOUT,
        "feature_shape": list(endpoint_vlm_feature_shape(num_frames)),
        "dtype": "float16",
        "num_frames": num_frames,
        "episode_data_index": episode_index,
        "episode_boundary_sha256": sha256_bytes(canonical_json_bytes(episode_index)),
        "primary_image_key": str(primary_image_key),
        "vlm_model": str(vlm_model),
        "vlm_checkpoint_fingerprint": dict(vlm_checkpoint_fingerprint or {}),
        "vlm_processor_fingerprint": dict(vlm_processor_fingerprint or {}),
        "vlm_visual_feature_spec": feature_spec,
        "source_rgb": "float_[0,1]",
        "source_quantization": "uint8_floor",
        "policy_resize": list(POLICY_RESIZE),
        "policy_resize_interpolation": "pil_nearest",
        "policy_center_crop": list(POLICY_CENTER_CROP),
        "pool": "adaptive_average_4x4",
        **({"image_preprocessing": geometry_spec, "policy_resize": None,
            "policy_resize_interpolation": geometry_spec["interpolation"],
            "policy_center_crop": None, "pool": geometry_spec["pool"]}
           if image_preprocessing != "legacy_v1" else {}),
    }


def _manifest_expected_mismatches(
    stored: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, value in expected.items():
        actual = stored.get(key)
        if key == "vlm_checkpoint_fingerprint":
            if not vlm_checkpoint_fingerprints_match(actual, value):
                mismatches[key] = (actual, value)
        elif actual != value:
            mismatches[key] = (actual, value)
    return mismatches


class ActionAlignedEndpointCache:
    """Read-only mmap cache with exact same-episode endpoint indexing."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        manifest: Mapping[str, Any] | None = None,
        *,
        expected_manifest: Mapping[str, Any] | None = None,
        episode_data_index: Mapping[str, Any] | None = None,
        target_mode: str = INTENT_TARGET_MODE,
        verify_feature_sha256: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / CACHE_MANIFEST_FILENAME
        if manifest is None:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid action-aligned cache manifest: {manifest_path}") from error
        self.manifest = dict(manifest)
        if (
            self.manifest.get("format_version") != CACHE_FORMAT_VERSION
            or self.manifest.get("format") != CACHE_FORMAT
        ):
            raise ValueError("unsupported endpoint cache; rebuild as action_aligned_vlm_endpoint_v2")
        if (
            self.manifest.get("layout") != CACHE_LAYOUT
            or self.manifest.get("index_layout") != CACHE_INDEX_LAYOUT
        ):
            raise ValueError("endpoint cache layout/index_layout mismatch")
        static_fields = {
            "token_layout": VLM_TOKEN_LAYOUT,
            "dtype": "float16",
            "source_rgb": "float_[0,1]",
            "source_quantization": "uint8_floor",
            "policy_resize": list(POLICY_RESIZE),
            "policy_resize_interpolation": "pil_nearest",
            "policy_center_crop": list(POLICY_CENTER_CROP),
            "pool": "adaptive_average_4x4",
            "vlm_visual_feature_spec": CANONICAL_VLM_FEATURE_SPEC,
        }
        if "image_preprocessing" in self.manifest:
            geometry_spec = image_preprocessing_spec(self.manifest["image_preprocessing"]["version"])
            static_fields.update(image_preprocessing=geometry_spec, policy_resize=None,
                                 policy_resize_interpolation=geometry_spec["interpolation"],
                                 policy_center_crop=None, pool=geometry_spec["pool"])
        static_mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in static_fields.items()
            if self.manifest.get(key) != value
        }
        if static_mismatches:
            raise ValueError(f"endpoint cache preprocessing ABI mismatch: {static_mismatches}")
        if not self.manifest.get("vlm_model"):
            raise ValueError("endpoint cache is missing its frozen VLM model identity")
        if not self.manifest.get("vlm_checkpoint_fingerprint"):
            raise ValueError("endpoint cache is missing frozen VLM checkpoint provenance")
        if not self.manifest.get("vlm_processor_fingerprint"):
            raise ValueError("endpoint cache is missing VLM processor provenance")
        shape = tuple(int(x) for x in self.manifest.get("feature_shape", ()))
        num_frames = int(self.manifest.get("num_frames", -1))
        if shape != endpoint_vlm_feature_shape(num_frames):
            raise ValueError("endpoint cache must have dtype float16 and shape [N,16,2048]")
        if self.manifest.get("token_layout") != VLM_TOKEN_LAYOUT:
            raise ValueError("endpoint cache token layout mismatch")
        stored_index = normalize_episode_data_index(
            self.manifest.get("episode_data_index", {}), num_frames=num_frames
        )
        if episode_data_index is not None:
            expected_index = normalize_episode_data_index(episode_data_index, num_frames=num_frames)
            if stored_index != expected_index:
                raise ValueError("endpoint cache episode boundaries do not match dataset")
        if sha256_bytes(canonical_json_bytes(stored_index)) != self.manifest.get(
            "episode_boundary_sha256"
        ):
            raise ValueError("endpoint cache episode_boundary_sha256 mismatch")
        if expected_manifest:
            mismatches = _manifest_expected_mismatches(self.manifest, expected_manifest)
            if mismatches:
                raise ValueError(f"endpoint cache manifest mismatch: {mismatches}")
        feature_name = Path(str(self.manifest.get("feature_file", "")))
        if (
            feature_name.is_absolute()
            or ".." in feature_name.parts
            or not feature_name.name.startswith(FEATURE_PREFIX)
            or feature_name.suffix != ".npy"
        ):
            raise ValueError("invalid endpoint feature_file")
        self.feature_path = self.cache_dir / feature_name
        if not self.feature_path.is_file():
            raise FileNotFoundError(f"missing endpoint feature array: {self.feature_path}")
        self.feature_shape = shape
        self._verify_feature_sha256 = bool(verify_feature_sha256)
        self._verify_feature()
        self._feature_stat = self.feature_path.stat()
        self._array: np.memmap | None = None
        self._pid: int | None = None
        self._episode_index = stored_index
        if target_mode != INTENT_TARGET_MODE:
            raise ValueError(
                "action-aligned VLM endpoint fixes target_mode to "
                f"{INTENT_TARGET_MODE!r}"
            )
        self.target_mode = str(target_mode)

    def _verify_feature(self) -> None:
        expected_size = int(self.manifest.get("feature_nbytes", -1))
        actual_size = self.feature_path.stat().st_size
        if expected_size != actual_size:
            raise ValueError(f"endpoint feature size mismatch: {actual_size} != {expected_size}")
        expected_mtime = self.manifest.get("feature_mtime_ns")
        if expected_mtime is not None and int(expected_mtime) != int(
            self.feature_path.stat().st_mtime_ns
        ):
            raise ValueError("endpoint feature mtime mismatch; rebuild or restore cache")
        if self._verify_feature_sha256:
            expected_hash = str(self.manifest.get("feature_sha256", ""))
            if sha256_file(self.feature_path) != expected_hash:
                raise ValueError("endpoint feature SHA-256 mismatch; rebuild or restore cache")
        try:
            array = np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise ValueError("cannot open endpoint feature array") from error
        try:
            if tuple(array.shape) != self.feature_shape or array.dtype != np.float16:
                raise ValueError("endpoint feature array shape/dtype mismatch")
        finally:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def _array_view(self) -> np.memmap:
        pid = os.getpid()
        if self._array is None or self._pid != pid:
            if self._array is not None and getattr(self._array, "_mmap", None) is not None:
                self._array._mmap.close()
            self._array = np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
            self._pid = pid
        return self._array

    def get_endpoint(self, row_index: int, horizon: int) -> tuple[torch.Tensor, bool]:
        """Return the exact VLM field at ``t+horizon`` or invalid zero target."""

        current_stat = self.feature_path.stat()
        if (
            current_stat.st_size != self._feature_stat.st_size
            or current_stat.st_mtime_ns != self._feature_stat.st_mtime_ns
            or current_stat.st_ctime_ns != self._feature_stat.st_ctime_ns
        ):
            self._verify_feature()
            self._feature_stat = current_stat
        target = endpoint_row_index(row_index, horizon, self._episode_index)
        if target is None:
            return torch.zeros((VLM_TARGET_TOKENS, VLM_TARGET_DIM), dtype=torch.float16), False
        value = np.asarray(self._array_view()[int(target)], dtype=np.float16).copy()
        if value.shape != (VLM_TARGET_TOKENS, VLM_TARGET_DIM):
            raise RuntimeError(
                "endpoint cache returned an invalid VLM spatial field: "
                f"{value.shape}"
            )
        return torch.from_numpy(value), True

    def close(self) -> None:
        if self._array is not None and getattr(self._array, "_mmap", None) is not None:
            self._array._mmap.close()
        self._array = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_pid"] = None
        return state


EndpointFeatureCache = ActionAlignedEndpointCache


__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_VERSION",
    "CACHE_MANIFEST_FILENAME",
    "CANONICAL_VLM_FEATURE_SPEC",
    "FEATURE_PREFIX",
    "INTENT_TARGET_MODE",
    "ActionAlignedEndpointCache",
    "EndpointFeatureCache",
    "build_action_aligned_endpoint_manifest",
    "build_action_aligned_vlm_checkpoint_fingerprint",
    "build_action_aligned_vlm_processor_fingerprint",
    "canonical_json_bytes",
    "endpoint_row_index",
    "extract_endpoint_vlm_features",
    "normalize_episode_data_index",
    "preprocess_endpoint_vlm_image",
    "preprocess_endpoint_vlm_images",
    "sha256_file",
    "vlm_checkpoint_fingerprints_match",
]
