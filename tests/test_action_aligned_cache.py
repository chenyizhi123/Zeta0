import json

import numpy as np
import pytest
import torch

from psi.data.lerobot.action_aligned_cache import (
    ActionAlignedEndpointCache,
    build_action_aligned_endpoint_manifest,
    build_action_aligned_vlm_checkpoint_fingerprint,
    endpoint_row_index,
    sha256_file,
)
from psi.models.action_aligned_vlm_features import (
    Qwen3VLVisualFeatureSpec,
    VLM_TARGET_DIM,
    VLM_TARGET_TOKENS,
    pool_qwen3vl_post_merger_image_tokens,
)
from scripts.data.cache_action_aligned_endpoint_features import publish_feature_array


def _spec() -> Qwen3VLVisualFeatureSpec:
    return Qwen3VLVisualFeatureSpec(
        model_type="qwen3_vl",
        vision_patch_size=16,
        vision_temporal_patch_size=2,
        spatial_merge_size=2,
        vision_hidden_size=1024,
        output_hidden_size=VLM_TARGET_DIM,
    )


def _provenance() -> dict[str, dict[str, str]]:
    return {
        "vlm_checkpoint_fingerprint": {"sha256": "checkpoint"},
        "vlm_processor_fingerprint": {"sha256": "processor"},
    }


def test_qwen_pool_is_row_major_and_fixed_4x4():
    # A Sofa policy image has a pre-merger 16x20 grid and therefore an 8x10
    # post-merger field.  Compare the helper to explicit adaptive pooling.
    tokens = torch.arange(8 * 10 * VLM_TARGET_DIM, dtype=torch.float32).reshape(
        8 * 10, VLM_TARGET_DIM
    )
    pooled = pool_qwen3vl_post_merger_image_tokens(
        [tokens], torch.tensor([[1, 16, 20]]), spec=_spec()
    )
    expected = torch.nn.functional.adaptive_avg_pool2d(
        tokens.T.reshape(1, VLM_TARGET_DIM, 8, 10), (4, 4)
    ).flatten(2).transpose(1, 2)
    assert pooled.shape == (1, VLM_TARGET_TOKENS, VLM_TARGET_DIM)
    assert torch.equal(pooled, expected)


def test_endpoint_index_never_crosses_episode():
    index = {"from": [0, 5], "to": [5, 9]}
    assert endpoint_row_index(1, 3, index) == 4
    assert endpoint_row_index(2, 3, index) is None
    assert endpoint_row_index(5, 3, index) == 8
    assert endpoint_row_index(7, 3, index) is None


def test_manifest_and_feature_hash_are_fail_closed(tmp_path):
    features = np.zeros((9, VLM_TARGET_TOKENS, VLM_TARGET_DIM), dtype=np.float16)
    for row in range(len(features)):
        for token in range(VLM_TARGET_TOKENS):
            features[row, token] = row * 32 + token
    feature_path = tmp_path / "action_aligned_vlm_endpoint_features.test.npy"
    np.save(feature_path, features, allow_pickle=False)
    manifest = build_action_aligned_endpoint_manifest(
        num_frames=9,
        episode_data_index={"from": [0, 5], "to": [5, 9]},
        primary_image_key="observation.images.head",
        vlm_model="/models/qwen3vl",
        vlm_feature_spec=_spec(),
        **_provenance(),
        feature_file=feature_path.name,
        feature_nbytes=feature_path.stat().st_size,
        feature_sha256=sha256_file(feature_path),
    )
    (tmp_path / "action_aligned_vlm_endpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    cache = ActionAlignedEndpointCache(tmp_path)
    target, valid = cache.get_endpoint(1, 3)
    assert valid
    assert target.shape == (VLM_TARGET_TOKENS, VLM_TARGET_DIM)
    assert torch.equal(target, torch.from_numpy(features[4]))
    assert not torch.equal(target[0], target[1])
    assert not cache.get_endpoint(2, 3)[1]
    feature_path.write_bytes(feature_path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        ActionAlignedEndpointCache(tmp_path)


def test_cache_publisher_uses_the_vlm_generation_pointer(tmp_path):
    features = np.zeros((3, VLM_TARGET_TOKENS, VLM_TARGET_DIM), dtype=np.float16)
    manifest_path = publish_feature_array(
        features,
        tmp_path,
        episode_data_index={"from": [0], "to": [3]},
        primary_image_key="observation.images.front",
        vlm_model="/models/qwen3vl",
        vlm_feature_spec=_spec(),
        **_provenance(),
    )
    assert manifest_path.name == "action_aligned_vlm_endpoint_manifest.json"
    assert ActionAlignedEndpointCache(tmp_path).get_endpoint(0, 1)[1]


def test_vlm_checkpoint_fingerprint_uses_stable_config_file(tmp_path):
    # Runtime Config metadata differs between Transformers patch versions;
    # the pinned on-disk config must remain the provenance identity.
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_vl","vision_config":{"out_hidden_size":2048}}',
        encoding="utf-8",
    )

    class ConfigA:
        def to_dict(self):
            return {
                "_name_or_path": "/runtime/a",
                "transformers_version": "4.57.0.dev0",
                "model_type": "qwen3_vl",
            }

    class ConfigB:
        def to_dict(self):
            return {
                "_name_or_path": "/runtime/b",
                "transformers_version": "4.57.3",
                "model_type": "qwen3_vl",
            }

    first = build_action_aligned_vlm_checkpoint_fingerprint(
        tmp_path, ConfigA(), include_content_hash=False
    )
    second = build_action_aligned_vlm_checkpoint_fingerprint(
        tmp_path, ConfigB(), include_content_hash=False
    )
    assert first == second
    assert first["method"].endswith("_v3")
