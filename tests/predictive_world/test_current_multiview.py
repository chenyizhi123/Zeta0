from dataclasses import replace
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.dataset import PredictiveWorldRowReader
from psi.data.predictive_world.index import temporal_abi_from_data_config
from psi.deploy.predictive_world_history import PredictiveWorldHistoryBuffer, TimedImage
from psi.deploy.predictive_world_service import QwenRequestAdapter
from test_deploy_history import _request

class Processor:
    def __init__(self):
        self.image_processor = self.encode_images
    def apply_chat_template(self, conversation, **kwargs):
        self.content = conversation[0]['content']
        return 'prompt'
    def encode_images(self, *, images, **kwargs):
        return {'image_grid_thw': torch.tensor([[1, 2, 2]] * len(images)),
                'pixel_values': torch.zeros(len(images) * 4, 3)}
    def __call__(self, *, images, **kwargs):
        return {**self.encode_images(images=images), 'input_ids': torch.ones(1, 8, dtype=torch.long),
                'attention_mask': torch.ones(1, 8, dtype=torch.long)}


def test_current_training_and_serving_propagate_the_future_cache_content_geometry():
    from psi.models.action_aligned_image import content_boxes, aspect_preprocess_image
    from psi.data.predictive_world.runtime import _collate_qwen
    from test_deploy_service import _sonic_request_and_window

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    expected = content_boxes([aspect_preprocess_image(Image.fromarray(image))])
    reader = PredictiveWorldRowReader.__new__(PredictiveWorldRowReader)
    reader.processor = Processor()
    reader.image_preprocessing = 'aspect_v1'
    reader.observed_visual_mode = 'current_multiview'
    reader.current_visual_pooling = 'content_area_v1'
    reader._decode_frames = lambda path, indices: [Image.fromarray(image) for _ in indices]
    row = dict(history_visual_indices=[0], history_visual_valid=[True], history_visual_actual_times=[0.])
    output = reader._qwen(row, dict(video_paths={'primary': 'p'}, primary_camera_id='primary', instruction='move'))
    torch.testing.assert_close(output['image_content_boxes'], expected, rtol=0, atol=0)
    collated = _collate_qwen([output, output])
    torch.testing.assert_close(collated['image_content_boxes'], expected.repeat(2, 1), rtol=0, atol=0)
    request, _ = _sonic_request_and_window(prefix_count=0)
    request = replace(request, images=(replace(request.images[0], raw_rgb=image),))
    window = PredictiveWorldHistoryBuffer(observed_visual_mode='current_multiview').build(
        request, receive_time=request.observation_anchor_time)
    adapter = QwenRequestAdapter(Processor(), image_preprocessing='aspect_v1', action_horizon=30,
        normalization_spaces={'sonic_semantic40_v31': 'native'}, current_visual_pooling='content_area_v1')
    serving = adapter(request, window)
    torch.testing.assert_close(serving.qwen_inputs['image_content_boxes'], expected, rtol=0, atol=0)
    reader.current_visual_pooling = 'canvas_v1'
    legacy = reader._qwen(row, dict(video_paths={'primary': 'p'}, primary_camera_id='primary', instruction='move'))
    assert 'image_content_boxes' not in legacy

@pytest.mark.parametrize('mode,expected', [('history', [1, 2, 3]), ('current_multiview', [3])])
def test_training_filters_before_decode_and_keeps_all_current_cameras(mode, expected):
    # Test image construction independently of the unrelated future feature store.
    reader = PredictiveWorldRowReader.__new__(PredictiveWorldRowReader)
    reader.processor = Processor()
    reader.image_preprocessing = 'aspect_v1'
    reader.current_visual_pooling = 'canvas_v1'
    reader.observed_visual_mode = mode
    decoded = {}
    def decode(path, indices):
        decoded[path] = list(indices)
        return [Image.new('RGB', (32, 32)) for _ in decoded[path]]
    reader._decode_frames = decode
    row = {'history_visual_indices': [1, 2, 3], 'history_visual_valid': [True] * 3,
           'history_visual_actual_times': [-.4, -.2, -.01]}
    row.update({
        'history_visual_indices_by_camera': {'primary': [1, 2, 3], 'wrist': [10, 20, 30]},
        'history_visual_valid_by_camera': {'primary': [True] * 3, 'wrist': [False, True, True]},
        'history_visual_actual_times_by_camera': {'primary': [-.4, -.2, -.01], 'wrist': [float('nan'), -.21, -.02]},
    })
    output = reader._qwen(row, {'video_paths': {'primary': 'p', 'wrist': 'w'},
                              'primary_camera_id': 'primary', 'instruction': 'move'})
    assert decoded == {'p': expected, 'w': [20, 30] if mode == 'history' else [30]}
    assert output['image_counts'].item() == (5 if mode == 'history' else 2)
    assert output['primary_current_image_indices'].item() == len(expected) - 1
    labels = [item['text'] for item in reader.processor.content if item['type'] == 'text']
    assert any('Camera: wrist;' in item for item in labels)
    if mode == 'current_multiview':
        assert not any('-0.400' in item or '-0.200' in item for item in labels)
        assert any('-0.010' in item for item in labels)

def test_current_multiview_serving_keeps_states_and_causal_views():
    stream = io.BytesIO()
    Image.new('RGB', (32, 32)).save(stream, format='PNG')
    request = _request()
    request = replace(request, images=tuple(replace(x, encoded_image=stream.getvalue()) for x in request.images) + (
        TimedImage('wrist', 9.6, 'old', stream.getvalue()),
        TimedImage('wrist', 9.99, 'now', stream.getvalue()),
        TimedImage('wrist', 10.01, 'future', stream.getvalue()),
        TimedImage('stale', 9.0, 'stale', stream.getvalue()),
    ))
    legacy = PredictiveWorldHistoryBuffer().build(request, receive_time=10.05)
    window = PredictiveWorldHistoryBuffer(observed_visual_mode='current_multiview').build(request, receive_time=10.05)
    assert [(x.camera_id, x.frame_id) for x in window.images] == [('primary', '2'), ('wrist', 'now')]
    assert window.image_requested_times == (10.0, 10.0)
    for name in ('state_values', 'state_dimension_valid', 'state_actual_times', 'state_source_sequences'):
        np.testing.assert_array_equal(getattr(window, name), getattr(legacy, name))
    processor = Processor()
    observed = QwenRequestAdapter(processor, image_preprocessing='aspect_v1', action_horizon=30,
                                 normalization_spaces={'toy': 'test'})(request, window)
    assert observed.qwen_inputs['image_counts'].item() == 2
    assert observed.qwen_inputs['primary_current_image_indices'].item() == 0
    assert observed.raw_states.shape == (1, 21, 2)
    assert any('Camera: wrist;' in x.get('text', '') for x in processor.content)
    without_primary = replace(request, images=tuple(x for x in request.images if x.camera_id != 'primary'))
    with pytest.raises(ValueError, match='current primary'):
        PredictiveWorldHistoryBuffer(observed_visual_mode='current_multiview').build(without_primary, receive_time=10.05)


def test_serving_adapter_accepts_simulator_raw_rgb_without_an_encoded_container(monkeypatch):
    import psi.deploy.predictive_world_service as service

    request = _request()
    raw = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(32, 24, 3)
    request = replace(
        request,
        images=tuple(
            replace(image, encoded_image=b"", raw_rgb=raw)
            for image in request.images
        ),
    )
    window = PredictiveWorldHistoryBuffer().build(request, receive_time=10.05)
    seen = []

    def identity_preprocess(image, *, image_preprocessing):
        assert image_preprocessing == "aspect_v1"
        seen.append(np.asarray(image).copy())
        return image

    monkeypatch.setattr(service, "preprocess_endpoint_vlm_image", identity_preprocess)
    QwenRequestAdapter(
        Processor(),
        image_preprocessing="aspect_v1",
        action_horizon=30,
        normalization_spaces={"toy": "test"},
    )(request, window)
    assert len(seen) == len(window.images)
    for image in seen:
        np.testing.assert_array_equal(image, raw)

def test_visual_selection_preserves_index_cache_abi_but_changes_bundle_contract():
    original = PredictiveWorldDataConfig()
    current = replace(original, observed_visual_mode='current_multiview')
    assert temporal_abi_from_data_config(current) == temporal_abi_from_data_config(original)
    assert current.observed_visual_offsets_s == (0.0,)
    assert current.observation_contract != original.observation_contract


@pytest.mark.parametrize('cameras', [1, 3])
def test_local_qwen_single_preprocessing_matches_previous_reader(cameras, monkeypatch):
    """Use the installed real processor/tokenizer, including image-token expansion."""
    from transformers import AutoProcessor
    model_path = Path(__file__).resolve().parents[2] / 'posttrain_assets/public/ckpts/qwen3vl'
    if not (model_path / 'preprocessor_config.json').is_file():
        pytest.skip('local Qwen processor assets are required for the deployment-specific equivalence check')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    calls = []
    original = type(processor.image_processor).preprocess

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(processor.image_processor), 'preprocess', counted)
    reader = PredictiveWorldRowReader.__new__(PredictiveWorldRowReader)
    reader.processor = processor
    reader.image_preprocessing = 'aspect_v1'
    reader.current_visual_pooling = 'canvas_v1'
    reader.observed_visual_mode = 'current_multiview'
    rng = np.random.default_rng(23)
    images = {str(i): Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
              for i, shape in enumerate([(480, 640, 3), (240, 320, 3), (640, 480, 3)][:cameras])}
    reader._decode_frames = lambda path, indices: [images[path] for _ in indices]
    row = dict(history_visual_indices=[0], history_visual_valid=[True], history_visual_actual_times=[0.0])
    materialization = dict(video_paths={str(i): str(i) for i in range(cameras)},
                           primary_camera_id='0', instruction='Move the box.')
    output = reader._qwen(row, materialization)
    assert len(calls) == 1

    from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
    policy_images = [preprocess_endpoint_vlm_image(image, image_preprocessing='aspect_v1')
                     for image in images.values()]
    content = []
    for i, image in enumerate(policy_images):
        content.extend([{'type': 'text', 'text': f'Camera: {i}; observation time: +0.000 seconds.'},
                        {'type': 'image', 'image': image}])
    content.append({'type': 'text', 'text': materialization['instruction']})
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                        tokenize=False, add_generation_prompt=True)
    previous = processor(text=[text], images=policy_images, padding=True, return_tensors='pt')
    exact = processor.image_processor(images=policy_images, return_tensors='pt')
    previous.update(exact)
    for name, value in previous.items():
        expected = value[0] if name in {'input_ids', 'attention_mask'} else value
        torch.testing.assert_close(output[name], expected, rtol=0, atol=0)
