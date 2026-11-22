"""Bounded processor diagnostics; evidence here never authorizes cache reuse."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from psi.data.lerobot.action_aligned_cache import (
    preprocess_endpoint_vlm_image, preprocess_endpoint_vlm_images,
)
from psi.data.predictive_world.cache import canonical_sha256
from psi.models.action_aligned_image import process_aspect_qwen


def legacy_processor_diagnostic(source, actual):
    """Compare recorded v1 fields, without treating missing runtime fields as equal."""
    fields = {
        'method', 'transformers_version', 'processor_class',
        'image_processor_class', 'image_processor_config_sha256',
    }
    if set(source) != fields or source['method'] != 'qwen_image_processor_json_sha256_v1':
        raise ValueError('expected the complete, unmodified source v1 fingerprint')
    if actual.get('method') != 'qwen_image_processor_json+image_runtime_v2':
        raise ValueError('expected an actual v2 runtime fingerprint')
    projected = {key: actual[key] for key in fields - {'method'}}
    projected['method'] = source['method']
    return {
        'source_processor_id': 'qwen3vl-processor:' + canonical_sha256(source),
        'actual_processor_id': 'qwen3vl-processor:' + canonical_sha256(actual),
        'recorded_v1_fields_match': projected == source,
        'differences': {key: {'source': source[key], 'actual': projected[key]}
                        for key in sorted(fields) if source[key] != projected[key]},
        'source_runtime_unrecorded': ['pillow', 'numpy', 'torch'],
        'authorizes_reuse': False,
    }


def tensor_record(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError('non-finite processor diagnostic tensor')
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}


def audit_processor_batch(processor, frames, instruction):
    """Exercise production geometry and Qwen image/chat interfaces on identical RGB."""
    batch = torch.stack(frames)
    canvases = preprocess_endpoint_vlm_images(batch, image_preprocessing='aspect_v1')
    images = [preprocess_endpoint_vlm_image(frame, image_preprocessing='aspect_v1')
              for frame in frames]
    pil_pixels = torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
                              for image in images])
    if not torch.equal(canvases, pil_pixels):
        raise ValueError('cache tensor/current PIL geometry differs')
    cache = process_aspect_qwen(processor, images=canvases)
    # Mirror the current-observation chat/image-only interfaces, including the
    # image-only pixel replacement, not a separately implemented normalizer.
    content = []
    for index, image in enumerate(images):
        content.extend((
            {'type': 'text', 'text': 'Camera: observation.images.front; observation time: '
             f'{(index-len(images)+1)/50:+.3f} seconds.'},
            {'type': 'image', 'image': image},
        ))
    content.append({'type': 'text', 'text': instruction})
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)
    chat = processor(text=[text], images=images, padding=True, return_tensors='pt')
    current = processor.image_processor(images=images, return_tensors='pt')
    if not torch.equal(chat['image_grid_thw'], current['image_grid_thw']):
        raise ValueError('chat and image-only grids differ')
    for key in ('pixel_values', 'image_grid_thw'):
        if not torch.equal(cache[key], current[key]):
            raise ValueError(f'cache/current {key} differ')
        chat[key] = current[key]
    return {
        'input': tensor_record(batch), 'canvas': tensor_record(canvases),
        'chat_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'cache': {key: tensor_record(value) for key, value in cache.items()},
        'current': {key: tensor_record(value) for key, value in chat.items()},
        'cache_current_pixels_equal': True,
    }


def compare_processor_reports(baseline, candidate):
    """Fail on different inputs, coverage or implementation as well as output bytes."""
    for key in ('format', 'source_files', 'model_sha256', 'code_sha256', 'fixture_spec'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'processor audit comparison differs in {key}')
    if not baseline['fixtures'] or baseline['fixtures'] != candidate['fixtures']:
        raise ValueError('processor audit fixture inputs or outputs differ')
    return {'byte_equal': True, 'fixture_batches': len(candidate['fixtures']),
            'authorizes_reuse': False, 'full_teacher_equivalence': False}
