from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from psi.models.action_aligned_image import (
    aspect_preprocess_batch, aspect_preprocess_image, content_boxes,
    image_geometry, process_aspect_qwen,
)
from psi.models.action_aligned_vlm_features import (
    pool_qwen3vl_post_merger_image_tokens, resolve_qwen3vl_visual_feature_spec,
)


@pytest.mark.parametrize("hw,canvas,tokens", [
    ((480, 640), (256, 320), 80), ((480, 1280), (256, 640), 160),
    ((640, 480), (320, 256), 80), ((1080, 1920), (256, 448), 112),
    ((8, 16), (32, 32), 1),
])
def test_geometry_retains_whole_image_at_bounded_resolution(hw, canvas, tokens):
    g = image_geometry(*hw)
    assert tuple(g["canvas_hw"]) == canvas
    top, left, bottom, right = g["content_box"]
    h, w = bottom - top, right - left
    assert abs(w - hw[1] * h / hw[0]) <= 1
    assert h <= hw[0] and w <= hw[1]
    assert canvas[0] * canvas[1] // 32**2 == tokens


@pytest.mark.parametrize("hw", [(20, 12000), (12000, 20), (1001, 2001)])
def test_unusual_aspects_obey_token_cap(hw):
    g = image_geometry(*hw)
    ch, cw = g["canvas_hw"]
    assert ch % 32 == cw % 32 == 0
    assert ch * cw // 1024 <= 160


def test_pil_and_tensor_cache_bytes_are_identical_and_antialiased():
    y, x = torch.meshgrid(torch.arange(480), torch.arange(1280), indexing="ij")
    raw = ((y + x) % 2 * 255).to(torch.uint8).expand(3, -1, -1)
    pil = Image.fromarray(raw.permute(1, 2, 0).numpy())
    one = aspect_preprocess_image(pil)
    batch = aspect_preprocess_batch(raw.unsqueeze(0).float() / 255)
    assert np.array_equal(np.asarray(one), batch[0].permute(1, 2, 0).numpy())
    assert float(batch[0, :, 12:-12, 4:-4].float().std()) < 1
    assert content_boxes([one]).tolist() == [[8, 0, 248, 640]]


def test_pool_excludes_fully_padded_tokens_and_keeps_gradients():
    spec = resolve_qwen3vl_visual_feature_spec(SimpleNamespace(
        model_type="qwen3_vl", vision_config=SimpleNamespace(
            model_type="qwen3_vl", patch_size=16, temporal_patch_size=2,
            spatial_merge_size=2, hidden_size=1024, out_hidden_size=2048)))
    values = torch.full((8, 10, 2048), 99.0)
    values[2:6] = 3
    tokens = values.reshape(80, 2048).requires_grad_()
    output = pool_qwen3vl_post_merger_image_tokens(
        [tokens], torch.tensor([[1, 16, 20]]), spec=spec,
        image_content_boxes=torch.tensor([[64, 0, 192, 320]]))
    torch.testing.assert_close(output, torch.full((1, 16, 2048), 3.0))
    output.sum().backward()
    assert tokens.grad.reshape(8, 10, 2048)[:2].count_nonzero() == 0


def test_processor_is_called_once_and_must_keep_grid():
    class Processor:
        calls = 0
        def __call__(self, *, text, images, padding, return_tensors, images_kwargs):
            self.calls += 1
            assert images_kwargs == {"do_resize": False}
            return {"image_grid_thw": torch.tensor([[1, im.height // 16, im.width // 16] for im in images])}
    processor = Processor()
    images = [aspect_preprocess_image(Image.new("RGB", (w, 480))) for w in (640, 1280)]
    assert process_aspect_qwen(processor, images=images, texts=["a", "b"])["image_grid_thw"].tolist() == [[1, 16, 20], [1, 16, 40]]
    assert processor.calls == 1
