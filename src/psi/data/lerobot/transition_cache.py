"""Frame-major frozen-DINO cache utilities for INSTEP.

The cache stores one DINO state per local LeRobot dataset row.  A training
sample assembles its ``current + H future`` state sequence lazily and clamps
the suffix to the final frame of the same episode, matching LeRobot's delta
query semantics without duplicating overlapping windows on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


CACHE_FORMAT_VERSION = 2
CACHE_LAYOUT = "frame_major_npy"
CACHE_INDEX_LAYOUT = "lerobot_dataset_row"
CACHE_FEATURE_FILENAME = "dino_frame_states.npy"
CACHE_MANIFEST_FILENAME = "latentroute_cache_manifest.json"
CACHE_PREPROCESS_VERSION = 2


class IndexedSingleImageDataset(torch.utils.data.Dataset):
    """Expose the true local row index and one current RGB frame."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        image_key: str,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        row_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.image_key = image_key
        self.image_transform = image_transform
        if row_indices is None:
            self.row_indices = np.arange(len(dataset), dtype=np.int64)
        else:
            indices = np.asarray(row_indices)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("row_indices must be a one-dimensional integer sequence.")
            self.row_indices = indices.astype(np.int64, copy=False)
        if np.any(self.row_indices < 0) or np.any(self.row_indices >= len(dataset)):
            raise IndexError(
                "row_indices must contain only dataset rows in "
                f"[0, {len(dataset)})."
            )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, local_index: int) -> tuple[int, torch.Tensor]:
        row_index = int(self.row_indices[local_index])
        item = self.dataset[row_index]
        image = item[self.image_key]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected one current RGB frame [3,H,W] at row {row_index}, "
                f"got {tuple(image.shape)}."
            )
        # Deliberately do not use item['index']: Sonic datasets reset that
        # field inside every episode.  row_index is the dense LeRobot row.
        if self.image_transform is not None:
            image = self.image_transform(image)
        return int(row_index), image


class DinoCacheImageTransform:
    """Pickle-safe deterministic PIL preprocessing run inside loader workers."""

    def __init__(
        self,
        *,
        policy_resize: Sequence[int],
        policy_center_crop: Sequence[int],
        image_size: int,
    ) -> None:
        from torchvision.transforms import v2

        resize = tuple(int(value) for value in policy_resize)
        crop = tuple(int(value) for value in policy_center_crop)
        if len(resize) != 2 or len(crop) != 2:
            raise ValueError("policy_resize and policy_center_crop must be [H,W].")
        if min(*resize, *crop, int(image_size)) <= 0:
            raise ValueError("All preprocessing sizes must be positive.")
        if crop[0] > resize[0] or crop[1] > resize[1]:
            raise ValueError("policy_center_crop cannot exceed policy_resize.")
        self.transform = v2.Compose(
            [
                v2.Resize(resize, interpolation=v2.InterpolationMode.NEAREST),
                v2.CenterCrop(crop),
                v2.Resize((int(image_size), int(image_size)), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"image must be a [3,H,W] tensor, got {type(image)} "
                f"with shape {getattr(image, 'shape', None)}."
            )
        if not image.is_floating_point():
            raise ValueError(
                "LeRobot DINO cache input must be floating-point RGB in [0,1]; "
                f"got {image.dtype}."
            )
        float_image = image.float()
        if not torch.isfinite(float_image).all():
            raise ValueError("DINO cache input contains NaN or infinity.")
        minimum = float_image.min().item()
        maximum = float_image.max().item()
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "LeRobot DINO cache input must be in [0,1], "
                f"got range [{minimum},{maximum}]."
            )
        quantized = float_image.mul(255.0).to(torch.uint8)
        # HWC uint8 with three channels determines RGB without a mode override.
        pil_image = Image.fromarray(quantized.permute(1, 2, 0).numpy())
        return self.transform(pil_image)


def _integer_vector(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers, got dtype {array.dtype}.")
    return array.astype(np.int64, copy=False)


def build_lerobot_dataset_fingerprint(dataset: Any) -> dict[str, Any]:
    """Fingerprint the loaded rows and all episode source files without reading videos.

    The source-file digest is the cache identity.  It covers the parquet and
    video episode files returned by LeRobot and is stable across symlinked
    dataset roots and different ``delta_timestamps`` constructions.  The
    Hugging Face fingerprint is retained as a diagnostic value only: LeRobot's
    fingerprint can change when the same dataset is opened through a different
    root path, even though the decoded frames are identical.
    """
    root = Path(dataset.root).resolve()
    get_paths = getattr(dataset, "get_episodes_file_paths", None)
    if callable(get_paths):
        source_paths = [Path(path) for path in get_paths()]
    else:
        source_paths = []
        for directory_name in ("data", "videos"):
            directory = root / directory_name
            if directory.is_dir():
                source_paths.extend(path for path in directory.rglob("*") if path.is_file())
    if not source_paths:
        raise ValueError(f"No LeRobot episode source files found under {root}.")

    source_digest = hashlib.sha256()
    for source_path in sorted(source_paths, key=lambda value: value.as_posix()):
        full_path = source_path if source_path.is_absolute() else root / source_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot source file: {full_path}")
        try:
            relative_path = full_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"LeRobot source file lies outside its dataset root: {full_path}"
            ) from error
        stat = full_path.stat()
        source_digest.update(relative_path.as_posix().encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(str(stat.st_size).encode("ascii"))
        source_digest.update(b"\0")
        source_digest.update(str(stat.st_mtime_ns).encode("ascii"))
        source_digest.update(b"\n")

    hf_dataset = getattr(dataset, "hf_dataset", None)
    hf_fingerprint = getattr(hf_dataset, "_fingerprint", None)
    return {
        "method": "hf_fingerprint+source_path_size_mtime_ns_v1",
        "hf_dataset_fingerprint": (
            str(hf_fingerprint) if hf_fingerprint is not None else None
        ),
        "source_files_digest": source_digest.hexdigest(),
        "source_file_count": len(source_paths),
        "dataset_revision": str(getattr(dataset, "revision", "unknown")),
    }


def dataset_fingerprints_match(
    stored: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    """Compare cache dataset provenance without using root-dependent HF state.

    ``hf_dataset_fingerprint`` is intentionally excluded.  It is generated by
    the LeRobot/Hugging Face dataset builder and can differ for a symlinked
    root or another harmless dataset-view construction.  The episode source
    digest, file count, revision, and fingerprint method remain strict so a
    changed dataset cannot silently reuse a stale DINO cache.
    """
    if not isinstance(stored, Mapping):
        return False
    for key in (
        "method",
        "source_files_digest",
        "source_file_count",
        "dataset_revision",
    ):
        if stored.get(key) != expected.get(key):
            return False
    return True


def transition_cache_manifest_mismatches(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return manifest differences using the stable dataset comparison rule."""
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, expected_value in expected.items():
        stored_value = stored.get(key)
        if key == "dataset_fingerprint":
            if not dataset_fingerprints_match(stored_value, expected_value):
                mismatches[key] = (stored_value, expected_value)
        elif stored_value != expected_value:
            mismatches[key] = (stored_value, expected_value)
    return mismatches


def build_dino_checkpoint_fingerprint(
    model_name_or_path: str,
    config: Any,
) -> dict[str, Any]:
    """Bind cached features to the resolved DINO config/commit or local files."""
    if not callable(getattr(config, "to_dict", None)):
        raise ValueError("DINO config must provide to_dict() for cache provenance.")
    config_payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    result: dict[str, Any] = {
        "method": "resolved_commit+config_sha256+local_file_stats_v1",
        "resolved_commit": getattr(config, "_commit_hash", None),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
    }

    local_path = Path(model_name_or_path).expanduser()
    if local_path.exists():
        checkpoint_files = (
            [local_path]
            if local_path.is_file()
            else sorted(path for path in local_path.rglob("*") if path.is_file())
        )
        local_digest = hashlib.sha256()
        for checkpoint_file in checkpoint_files:
            relative_path = (
                Path(checkpoint_file.name)
                if local_path.is_file()
                else checkpoint_file.relative_to(local_path)
            )
            stat = checkpoint_file.stat()
            local_digest.update(relative_path.as_posix().encode("utf-8"))
            local_digest.update(b"\0")
            local_digest.update(str(stat.st_size).encode("ascii"))
            local_digest.update(b"\0")
            local_digest.update(str(stat.st_mtime_ns).encode("ascii"))
            local_digest.update(b"\n")
        result["local_files_digest"] = local_digest.hexdigest()
        result["local_file_count"] = len(checkpoint_files)
    else:
        result["local_files_digest"] = None
        result["local_file_count"] = 0
    return result


def normalize_episode_data_index(
    episode_data_index: Mapping[str, Any],
    *,
    num_frames: int,
) -> dict[str, list[int]]:
    """Validate and serialize contiguous, end-exclusive episode boundaries."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    try:
        episode_from = _integer_vector(
            episode_data_index["from"], name="episode_data_index['from']"
        )
        episode_to = _integer_vector(
            episode_data_index["to"], name="episode_data_index['to']"
        )
    except KeyError as error:
        raise ValueError("episode_data_index must contain 'from' and 'to'.") from error

    if episode_from.size == 0 or episode_from.shape != episode_to.shape:
        raise ValueError(
            "episode_data_index must contain equally sized, non-empty 'from'/'to' vectors."
        )
    if episode_from[0] != 0 or episode_to[-1] != num_frames:
        raise ValueError(
            "episode_data_index must cover every local dataset row exactly: "
            f"expected [0,{num_frames}), got [{episode_from[0]},{episode_to[-1]})."
        )
    if np.any(episode_to <= episode_from):
        raise ValueError("Every episode must contain at least one frame.")
    if episode_from.size > 1 and np.any(episode_from[1:] != episode_to[:-1]):
        raise ValueError("episode_data_index episodes must be contiguous and non-overlapping.")

    return {
        "from": episode_from.tolist(),
        "to": episode_to.tolist(),
    }


def transition_window_indices(
    row_index: int,
    horizon: int,
    episode_data_index: Mapping[str, Any],
) -> np.ndarray:
    """Return ``H+1`` local row indices without crossing an episode boundary."""
    episode_from = _integer_vector(
        episode_data_index["from"], name="episode_data_index['from']"
    )
    episode_to = _integer_vector(
        episode_data_index["to"], name="episode_data_index['to']"
    )
    if episode_from.size == 0 or episode_from.shape != episode_to.shape:
        raise ValueError("Invalid episode_data_index.")
    return _transition_window_indices(row_index, horizon, episode_from, episode_to)


def _transition_window_indices(
    row_index: int,
    horizon: int,
    episode_from: np.ndarray,
    episode_to: np.ndarray,
) -> np.ndarray:
    row_index = int(row_index)
    horizon = int(horizon)
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}.")
    num_frames = int(episode_to[-1])
    if row_index < 0 or row_index >= num_frames:
        raise IndexError(f"Dataset row {row_index} is outside [0,{num_frames}).")

    episode_slot = int(np.searchsorted(episode_to, row_index, side="right"))
    if episode_slot >= episode_to.size or row_index < int(episode_from[episode_slot]):
        raise ValueError(
            f"Dataset row {row_index} is not covered by episode_data_index."
        )
    episode_last = int(episode_to[episode_slot]) - 1
    requested = row_index + np.arange(horizon + 1, dtype=np.int64)
    return np.minimum(requested, episode_last)


def build_transition_cache_manifest(
    *,
    repo_id: str,
    image_key: str,
    fps: float,
    num_frames: int,
    episode_data_index: Mapping[str, Any],
    image_size: int,
    policy_resize: Sequence[int],
    policy_center_crop: Sequence[int],
    dino_model: str,
    dataset_fingerprint: Mapping[str, Any],
    dino_checkpoint_fingerprint: Mapping[str, Any],
    feature_file: str = CACHE_FEATURE_FILENAME,
    state_tokens: int = 5,
    state_dim: int = 384,
) -> dict[str, Any]:
    """Build the complete v2 manifest shared by the writer and reader."""
    episode_layout = normalize_episode_data_index(
        episode_data_index, num_frames=num_frames
    )
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "layout": CACHE_LAYOUT,
        "index_layout": CACHE_INDEX_LAYOUT,
        "feature_file": str(feature_file),
        "feature_shape": [num_frames, state_tokens, state_dim],
        "repo_id": str(repo_id),
        "image_key": str(image_key),
        "fps": float(fps),
        "num_frames": int(num_frames),
        "episode_data_index": episode_layout,
        "image_size": int(image_size),
        "policy_resize": [int(value) for value in policy_resize],
        "policy_center_crop": [int(value) for value in policy_center_crop],
        "dino_model": str(dino_model),
        "dataset_fingerprint": dict(dataset_fingerprint),
        "dino_checkpoint_fingerprint": dict(dino_checkpoint_fingerprint),
        "state_tokens": int(state_tokens),
        "state_dim": int(state_dim),
        "dtype": "float16",
        "preprocess_version": CACHE_PREPROCESS_VERSION,
        "source_quantization": "uint8_floor",
        "policy_resize_interpolation": "pil_nearest",
        "dino_resize_interpolation": "pil_bilinear_antialias",
    }


def preprocess_dino_cache_images(
    images: torch.Tensor,
    *,
    policy_resize: Sequence[int],
    policy_center_crop: Sequence[int],
    image_size: int,
) -> torch.Tensor:
    """Reproduce INSTEP's deterministic PIL preprocessing for a frame batch.

    LeRobot video frames normally arrive as floats in ``[0,1]``.  The online
    training path first converts them to uint8 PIL images, applies Psi0's
    nearest-neighbour resize and centre crop, and then performs DINO's bilinear
    square resize.  Keeping the same PIL path here prevents cache/online drift.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"images must be a [B,3,H,W] tensor, got {type(images)} "
            f"with shape {getattr(images, 'shape', None)}."
        )
    transform = DinoCacheImageTransform(
        policy_resize=policy_resize,
        policy_center_crop=policy_center_crop,
        image_size=image_size,
    )
    return torch.stack([transform(image) for image in images.detach().cpu()])


class PerFrameDinoCache:
    """PID-aware, read-only mmap view over a frame-major DINO cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: Mapping[str, Any],
        *,
        episode_data_index: Mapping[str, Any],
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.manifest = dict(manifest)
        if self.manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                "INSTEP cache format is unsupported. Expected frame-major "
                f"format v{CACHE_FORMAT_VERSION}; rebuild the old cache."
            )
        if self.manifest.get("layout") != CACHE_LAYOUT:
            raise ValueError(
                f"INSTEP cache layout must be {CACHE_LAYOUT!r}, "
                f"got {self.manifest.get('layout')!r}."
            )
        if self.manifest.get("index_layout") != CACHE_INDEX_LAYOUT:
            raise ValueError(
                f"INSTEP cache index_layout must be {CACHE_INDEX_LAYOUT!r}."
            )

        feature_file = Path(str(self.manifest.get("feature_file", "")))
        if (
            not feature_file.name
            or feature_file.is_absolute()
            or ".." in feature_file.parts
        ):
            raise ValueError(f"Invalid cache feature_file: {feature_file!s}.")
        self.feature_path = self.cache_dir / feature_file
        if not self.feature_path.is_file():
            raise FileNotFoundError(f"Missing INSTEP feature array: {self.feature_path}")

        try:
            feature_shape = tuple(int(value) for value in self.manifest["feature_shape"])
            expected_dtype = np.dtype(str(self.manifest["dtype"]))
            num_frames = int(self.manifest["num_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("INSTEP cache manifest has invalid array metadata.") from error
        expected_shape = (
            num_frames,
            int(self.manifest.get("state_tokens", -1)),
            int(self.manifest.get("state_dim", -1)),
        )
        if feature_shape != expected_shape or len(feature_shape) != 3:
            raise ValueError(
                f"Manifest feature_shape must be {expected_shape}, got {feature_shape}."
            )
        if expected_dtype != np.dtype(np.float16):
            raise ValueError(
                f"INSTEP cache dtype must be float16, got {expected_dtype}."
            )

        self.episode_data_index = normalize_episode_data_index(
            episode_data_index, num_frames=num_frames
        )
        if self.manifest.get("episode_data_index") != self.episode_data_index:
            raise ValueError(
                "INSTEP cache episode boundaries do not match the loaded dataset."
            )
        self.feature_shape = feature_shape
        self.dtype = expected_dtype
        self._episode_from = np.asarray(
            self.episode_data_index["from"], dtype=np.int64
        )
        self._episode_to = np.asarray(
            self.episode_data_index["to"], dtype=np.int64
        )
        self._array: np.memmap | None = None
        self._array_pid: int | None = None
        self._validate_backing_array()

    def _open_array(self) -> np.memmap:
        try:
            array = np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot open INSTEP feature array {self.feature_path}."
            ) from error
        if not isinstance(array, np.memmap):
            raise ValueError(
                f"INSTEP feature array {self.feature_path} is not mmap-compatible."
            )
        return array

    @staticmethod
    def _close_array(array: np.memmap | None) -> None:
        if array is not None:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def _validate_backing_array(self) -> None:
        array = self._open_array()
        try:
            if tuple(array.shape) != self.feature_shape:
                raise ValueError(
                    f"INSTEP feature array has shape {tuple(array.shape)}, "
                    f"expected {self.feature_shape}."
                )
            if array.dtype != self.dtype:
                raise ValueError(
                    f"INSTEP feature array has dtype {array.dtype}, expected {self.dtype}."
                )
        finally:
            self._close_array(array)

    def _get_array(self) -> np.memmap:
        process_id = os.getpid()
        if self._array is None or self._array_pid != process_id:
            self._close_array(self._array)
            self._array = self._open_array()
            self._array_pid = process_id
        return self._array

    def get_window(self, row_index: int, horizon: int) -> torch.Tensor:
        indices = _transition_window_indices(
            row_index, horizon, self._episode_from, self._episode_to
        )
        # Advanced indexing a read-only mmap can still expose a read-only NumPy
        # view on some versions.  An explicit copy gives torch an ordinary,
        # writable CPU buffer and avoids worker/collator warnings.
        window = np.array(self._get_array()[indices], copy=True)
        return torch.from_numpy(window)

    def close(self) -> None:
        self._close_array(self._array)
        self._array = None
        self._array_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        state["_array_pid"] = None
        return state
