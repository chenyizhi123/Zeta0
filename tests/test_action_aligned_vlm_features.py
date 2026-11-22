from types import SimpleNamespace

import pytest
import torch

from psi.models.action_aligned_vlm_features import (
    VLM_TARGET_DIM,
    VLM_TARGET_TOKENS,
    extract_qwen3vl_post_merger_image_features,
    resolve_qwen3vl_visual_feature_spec,
)


def _config(*, output_hidden_size=VLM_TARGET_DIM):
    return SimpleNamespace(
        model_type="qwen3_vl",
        vision_config=SimpleNamespace(
            model_type="qwen3_vl",
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            hidden_size=1024,
            out_hidden_size=output_hidden_size,
        ),
    )


class _FakeVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _config()
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw):
        self.calls += 1
        # Two policy images have post-merger grids 8x10 and 10x8.
        fields = [
            torch.arange(80 * VLM_TARGET_DIM, dtype=torch.float32).reshape(80, VLM_TARGET_DIM),
            torch.arange(80 * VLM_TARGET_DIM, dtype=torch.float32).reshape(80, VLM_TARGET_DIM) + 1,
        ]
        return fields, []


def test_extract_uses_one_public_qwen_visual_call_and_returns_fixed_field():
    model = _FakeVLM()
    output = extract_qwen3vl_post_merger_image_features(
        model,
        torch.empty(640, 1536),
        torch.tensor([[1, 16, 20], [1, 20, 16]]),
    )
    assert model.calls == 1
    assert output.shape == (2, VLM_TARGET_TOKENS, VLM_TARGET_DIM)
    assert torch.isfinite(output).all()


def test_feature_spec_rejects_auxiliary_projection_width():
    with pytest.raises(ValueError, match="unsupported Qwen3-VL visual target ABI"):
        resolve_qwen3vl_visual_feature_spec(_config(output_hidden_size=768))
