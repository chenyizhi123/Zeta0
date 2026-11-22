"""Exercise versioned preprocessing provenance and actual cache rejection."""

from types import SimpleNamespace

import numpy as np
import PIL
import pytest
import torch

from psi.data.lerobot.action_aligned_cache import (
    ActionAlignedEndpointCache, build_action_aligned_vlm_processor_fingerprint,
)
from scripts.data.cache_action_aligned_endpoint_features import publish_feature_array
from test_action_aligned_cache import _spec


def processor():
    return SimpleNamespace(image_processor=SimpleNamespace(to_json_string=lambda: '{"resize":32}'))


def test_runtime_fingerprint_has_explicit_schema_and_loaded_versions():
    first = build_action_aligned_vlm_processor_fingerprint(processor())
    assert first == build_action_aligned_vlm_processor_fingerprint(processor())
    assert first["method"] == "qwen_image_processor_json+image_runtime_v2"
    assert first["image_runtime"] == {
        "pillow": PIL.__version__, "numpy": np.__version__, "torch": torch.__version__,
    }


@pytest.mark.parametrize("library", [PIL, np, torch])
def test_runtime_change_changes_fingerprint(monkeypatch, library):
    original = build_action_aligned_vlm_processor_fingerprint(processor())
    monkeypatch.setattr(library, "__version__", "different-runtime")
    assert build_action_aligned_vlm_processor_fingerprint(processor()) != original


@pytest.mark.parametrize("change", ["legacy", "pillow", "none"])
def test_cache_rejects_unbound_or_changed_runtime(tmp_path, monkeypatch, change):
    stored = build_action_aligned_vlm_processor_fingerprint(processor())
    if change == "legacy":
        stored.pop("image_runtime", None)
        stored["method"] = "qwen_image_processor_json_sha256_v1"
    publish_feature_array(
        np.zeros((2, 16, 2048), dtype=np.float16), tmp_path,
        episode_data_index={"from": [0], "to": [2]},
        primary_image_key="observation.images.head", vlm_model="synthetic-teacher",
        vlm_feature_spec=_spec(), vlm_checkpoint_fingerprint={"sha256": "fixture"},
        vlm_processor_fingerprint=stored,
    )
    if change == "pillow":
        monkeypatch.setattr(PIL, "__version__", "different-runtime")
    current = build_action_aligned_vlm_processor_fingerprint(processor())
    if change == "none":
        cache = ActionAlignedEndpointCache(
            tmp_path, expected_manifest={"vlm_processor_fingerprint": current}
        )
        assert cache.manifest["vlm_processor_fingerprint"] == current
        return
    with pytest.raises(ValueError, match="vlm_processor_fingerprint"):
        ActionAlignedEndpointCache(tmp_path, expected_manifest={"vlm_processor_fingerprint": current})
