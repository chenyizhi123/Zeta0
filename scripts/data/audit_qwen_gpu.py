#!/usr/bin/env python3
"""Read-only GPU diagnostic: real RGB teacher features and source-checkpoint E/N.

This is not cache migration, a training entry point, or historical equivalence
acceptance. Body evidence is explicitly masked; no optimizer is constructed.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from psi.config.predictive_world import load_predictive_world_config
from psi.data.lerobot.action_aligned_cache import extract_endpoint_vlm_features
from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.processor_audit import legacy_processor_diagnostic, tensor_record
from psi.data.predictive_world.provenance import verify_qwen_provenance
from psi.data.predictive_world.video import decode_torchcodec_frames
from psi.models.predictive_world.action_expert import clone_independent_expert
from psi.models.predictive_world.checkpoint import load_predictive_world_bundle
from psi.models.predictive_world.factory import _forecast_modules, _reference_modules, _state_registry
from psi.models.predictive_world.initialization import read_initialization_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--asset-path-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(101)
    asset_map = load_asset_path_map(args.asset_path_map)
    if not torch.cuda.is_available():
        raise RuntimeError('this diagnostic requires CUDA')
    baseline = json.loads(args.cpu_report.read_text())
    root = Path(__file__).resolve().parents[2]
    for relative, digest in baseline['code_sha256'].items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f'CPU report implementation changed: {relative}')
    source_dir = Path(baseline['source_directory'])
    for name, digest in baseline['source_files'].items():
        if file_sha256(source_dir / name) != digest:
            raise ValueError(f'CPU report source changed: {name}')
    processor = AutoProcessor.from_pretrained(baseline['model_path'], local_files_only=True)
    print('Verifying actual GPU-run teacher identity and full model hash', flush=True)
    # Fresh diagnostic expectation, explicitly the NEW runtime from the CPU
    # report. Never relabel the original cache or checkpoint as this identity.
    identity = verify_qwen_provenance(baseline['model_path'], {
        'encoder_id': 'qwen3vl:' + baseline['model_sha256'],
        'processor_id': baseline['source_diagnostic']['actual_processor_id'],
        'image_preprocessing': 'aspect_v1',
        'feature_layer': 'post_merger_content_area_overlap_4x4_v1',
        'dtype': 'float16', 'feature_shape': [16, 2048],
    }, processor=processor)
    source_fp = json.loads((source_dir / 'teacher_processor_fingerprint.json').read_text())
    comparison = legacy_processor_diagnostic(source_fp, identity['processor_fingerprint'])
    if not comparison['recorded_v1_fields_match']:
        raise ValueError('recorded source processor fields differ')
    print(json.dumps({'event': 'teacher_verified', **identity}), flush=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        baseline['model_path'], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa',
    ).to('cuda').eval().requires_grad_(False)
    results = {}
    retained = None
    retained_single = None
    decoded = {}
    for task in ('HSI_boxing', 'HOI_double_desk'):
        fixture = baseline['fixtures'][task]['source']
        if file_sha256(Path(fixture['video'])) != fixture['video_sha256']:
            raise ValueError('fixture video hash changed')
        # Anchor from the recorded fixture, then real +0.2/+0.4/+0.6 s frames.
        start = fixture['frame_indices'][0]
        indices = [start + offset for offset in (0, 10, 20, 30)]
        frames = torch.stack(decode_torchcodec_frames(fixture['video'], indices))
        decoded[task] = frames
        batched = extract_endpoint_vlm_features(
            qwen, processor, frames, device='cuda', image_preprocessing='aspect_v1')
        single = torch.cat([extract_endpoint_vlm_features(
            qwen, processor, frame[None], device='cuda', image_preprocessing='aspect_v1')
            for frame in frames])
        delta = (batched.float() - single.float()).abs()
        # Batch-vs-single is a measured numerical comparison, not a historical
        # reference or a pass/fail tolerance chosen after observing the result.
        results[task] = {
            'source': fixture, 'frame_indices': indices, 'rgb': tensor_record(frames),
            'features': tensor_record(batched), 'single_features': tensor_record(single),
            'feature_abs_max': batched.float().abs().max().item(),
            'batch_single_max_abs': delta.max().item(),
            'batch_single_mean_abs': delta.mean().item(),
            'batch_single_relative_l2': (delta.norm() / single.float().norm().clamp_min(1e-12)).item(),
        }
        if task == 'HSI_boxing':
            retained = batched
            retained_single = single
        print(json.dumps({'event': 'features', 'task': task, **results[task]}), flush=True)
    # Finish every BF16 sample before conversion: converting back with .bfloat16()
    # would also round buffers originally retained in FP32 by from_pretrained.
    # Same loaded weights, FP32 arithmetic, not original full-precision weights.
    qwen.float()
    for task, frames in decoded.items():
        fp32_batch = extract_endpoint_vlm_features(
            qwen, processor, frames, device='cuda', image_preprocessing='aspect_v1')
        fp32_single = torch.cat([extract_endpoint_vlm_features(
            qwen, processor, frame[None], device='cuda', image_preprocessing='aspect_v1')
            for frame in frames])
        fp32_delta = (fp32_batch.float() - fp32_single.float()).abs()
        results[task]['fp32_compute'] = {
            'features': tensor_record(fp32_batch), 'single_features': tensor_record(fp32_single),
            'batch_single_max_abs': fp32_delta.max().item(),
            'batch_single_relative_l2': (fp32_delta.norm() / fp32_single.float().norm().clamp_min(1e-12)).item(),
        }
        print(json.dumps({'event': 'fp32_features', 'task': task,
                          **results[task]['fp32_compute']}), flush=True)
    del qwen
    torch.cuda.empty_cache()
    print('Verifying source bundle/config and strictly loading every joint module', flush=True)
    sidecar, payload, registries = read_initialization_source(
        args.checkpoint, args.source_config, asset_path_map=asset_map)
    if sidecar['metadata']['processor_id'] != comparison['source_processor_id']:
        raise ValueError('joint checkpoint and source teacher processor disagree')
    if sidecar['metadata']['encoder_id'] != identity['encoder_id']:
        raise ValueError('joint checkpoint and actual teacher weights disagree')
    config, loaded_payload = load_predictive_world_config(args.source_config)
    if loaded_payload != payload:
        raise ValueError('source config changed while loading')
    dims, stats = _state_registry(str(resolve_asset_path(
        config.data.state_schema_registry, asset_map)))
    reference = _reference_modules(config, registries['action_protocol_registry'], dims, stats, qwen_dim=2048)
    forecast = _forecast_modules(config, registries['action_protocol_registry'], dims, stats, qwen_dim=2048)
    modules = {**reference, **forecast, 'student': clone_independent_expert(reference['teacher'])}
    load_predictive_world_bundle(args.checkpoint, phase='joint', modules=modules,
                                 expected_metadata=sidecar['metadata'])
    encoder, normalizer = reference['encoder'].cuda().eval(), reference['normalizer'].cuda().eval()
    from psi.models.predictive_world.training import module_state_sha256
    if module_state_sha256(normalizer) != sidecar['metadata']['normalization_id']:
        raise ValueError('loaded normalizer identity differs from checkpoint')
    if not bool(normalizer.fitted) or not bool(encoder.normalizer_frozen):
        raise ValueError('source latent normalizer is not fitted/frozen')
    before = {key: value.clone() for key, value in normalizer.state_dict().items()}
    patches = retained.cuda().float()
    dm = config.model.measurement_dim
    arguments = {
        'anchor_patches': patches[0:1],
        'anchor_patch_valid': torch.ones(1, 16, device='cuda', dtype=torch.bool),
        'future_patches': patches[1:][None],
        'future_patch_valid': torch.ones(1, 3, 16, device='cuda', dtype=torch.bool),
        'future_scene_times': torch.tensor([[0.2, 0.4, 0.6]], device='cuda'),
        'current_measurements': torch.zeros(1, dm, device='cuda'),
        'current_measurement_valid': torch.zeros(1, dm, device='cuda', dtype=torch.bool),
        'future_measurements': torch.zeros(1, 3, 10, dm, device='cuda'),
        'future_measurement_valid': torch.zeros(1, 3, 10, dm, device='cuda', dtype=torch.bool),
        'future_measurement_times': torch.zeros(1, 3, 10, device='cuda'),
        'group_valid': torch.ones(1, 3, device='cuda', dtype=torch.bool),
    }
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        target = encoder(**arguments)
        online = encoder.encode_online(**arguments)
        white = normalizer.whiten(target)
        arguments['anchor_patches'] = retained_single[0:1].cuda().float()
        arguments['future_patches'] = retained_single[1:][None].cuda().float()
        white_single = normalizer.whiten(encoder(**arguments))
    for key, value in normalizer.state_dict().items():
        if not torch.equal(before[key], value):
            raise ValueError('source normalizer mutated')
    latent = {name: {**tensor_record(value.float()), 'abs_max': value.float().abs().max().item()}
              for name, value in {'target': target, 'online': online, 'whitened': white}.items()}
    latent.update(normalizer_unchanged=True, body_evidence='all masked (visual-only diagnostic)',
                  batch_single_white_relative_l2=((white-white_single).norm() / white_single.norm().clamp_min(1e-12)).item(),
                  last_target_update=encoder.last_target_update.item(),
                  last_normalizer_update=encoder.last_normalizer_update.item())
    report = {
        'format': 'zeta0.qwen_gpu_diagnostic.v1', 'seed': 101,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
        'cpu_report_sha256': file_sha256(args.cpu_report), 'actual_teacher': identity,
        'source_processor_comparison': comparison,
        'source_checkpoint_metadata': sidecar, 'source_config_sha256': file_sha256(args.source_config),
        'diagnostic_code_sha256': file_sha256(Path(__file__)),
        'runtime_code_sha256': {str(path.relative_to(root)): file_sha256(path)
                               for path in sorted((root/'src/psi').rglob('*.py'))},
        'features': results, 'latent': latent,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'optimizer_steps': 0, 'authorizes_cache_reuse': False,
        'historical_teacher_equivalence': False,
        'limitations': ['No historical feature reference available.',
                        'E/N visual-only diagnostic, not full boxing batch or policy training.',
                        'No checkpoint/cache migration and no closed-loop evaluation.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'latent': latent}), flush=True)


if __name__ == '__main__':
    main()
