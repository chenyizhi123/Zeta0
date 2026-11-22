#!/usr/bin/env python3
"""CPU-only processor A/B diagnostic. Writes a new report, never a production cache.

Budget: three synthetic aspect ratios and first/middle/last frames of episode 0
for each selected task. No model inference, training, or automatic migration.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from packaging.requirements import Requirement
from transformers import AutoProcessor

from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.processor_audit import (
    audit_processor_batch, compare_processor_reports, legacy_processor_diagnostic,
)
from psi.data.predictive_world.sonic_post_training import TASKS, inspect_dataset
from psi.data.predictive_world.video import decode_torchcodec_frames
from psi.data.predictor_pretrain.runtime import fingerprint_model_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source_files = {name: file_sha256(args.source_directory/name) for name in (
        'stage_a_config.json', 'teacher_processor_fingerprint.json', 'provenance.json',
    )}
    stage = json.loads((args.source_directory/'stage_a_config.json').read_text())
    source = json.loads((args.source_directory/'teacher_processor_fingerprint.json').read_text())
    if source != stage['teacher_processor_fingerprint']:
        raise ValueError('source standalone fingerprint disagrees with Stage A')
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    actual = build_action_aligned_vlm_processor_fingerprint(processor)
    print('Hashing frozen model files (no model allocation)', flush=True)
    model_sha = fingerprint_model_directory(args.model)
    if model_sha != stage['teacher_sha256']:
        raise ValueError('actual frozen model hash differs from source Stage A')
    code_paths = [Path(__file__), *[root/path for path in (
        'src/psi/data/predictive_world/processor_audit.py',
        'src/psi/data/predictive_world/dataset.py',
        'src/psi/data/predictive_world/video.py',
        'src/psi/data/predictive_world/sonic_post_training.py',
        'src/psi/data/lerobot/action_aligned_cache.py',
        'src/psi/models/action_aligned_image.py',
    )]]
    report = {
        'format': 'qwen_processor_cpu_ab_v1', 'source_files': source_files,
        'source_directory': str(args.source_directory.resolve()),
        'model_path': str(args.model.resolve()), 'model_sha256': model_sha,
        'code_sha256': {str(path.relative_to(root)): file_sha256(path) for path in code_paths},
        'base_uv_lock_sha256': file_sha256(root/'uv.lock'),
        'overlay_lock_sha256': file_sha256(root/'scripts/data/processor4573.requirements.txt'),
        'processor_fingerprint': actual,
        'source_diagnostic': legacy_processor_diagnostic(source, actual),
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(), 'packages': {}},
        'fixture_spec': {'synthetic_hw': [[481, 641], [641, 481], [240, 1280]],
                         'tasks': args.tasks, 'episode': 0,
                         'positions': ['first', 'middle', 'last'], 'device': 'cpu'},
        'fixtures': {}, 'production_identity_guard_changed': False,
        'limitations': ['Source v1 omits Pillow/NumPy/Torch versions.',
                        'Same local decoder/runtime on both sides, not original source outputs.',
                        'No Qwen feature inference, GPU/BF16, training or closed-loop validation.',
                        'Diagnostic evidence only; not a migration approval.'],
    }
    core_requirements = {}
    for text in importlib.metadata.requires('transformers'):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        installed = importlib.metadata.version(requirement.name)
        if not requirement.specifier.contains(installed, prereleases=True):
            raise ValueError(f'unsatisfied Transformers core dependency: {text}: {installed}')
        core_requirements[text] = installed
    report['environment']['transformers_core_requirements'] = core_requirements
    for package, module_name in (('transformers', 'transformers'), ('torch', 'torch'),
                                 ('numpy', 'numpy'), ('pillow', 'PIL'),
                                 ('torchvision', 'torchvision'), ('torchcodec', 'torchcodec'),
                                 ('tokenizers', 'tokenizers')):
        module = importlib.import_module(module_name)
        report['environment']['packages'][package] = {
            'version': importlib.metadata.version(package), 'module_path': module.__file__,
        }
    for height, width in report['fixture_spec']['synthetic_hw']:
        y, x = np.indices((height, width), dtype=np.int32)
        pixels = np.stack(((x*17+y*3) % 256, (x ^ y) % 256,
                           (x*7+y*29) % 256), axis=0).astype(np.uint8)
        frame = torch.from_numpy(pixels).float()/255
        report['fixtures'][f'synthetic_{height}x{width}'] = audit_processor_batch(
            processor, [frame], 'Move the box. 把箱子放到桌上。'
        )
    for task in args.tasks:
        print(f'Auditing {task}: episode 0, three real frames', flush=True)
        _, instruction, episodes, _, _ = inspect_dataset(
            args.data_root/task/'sonic_refpose_v3_1', task=task
        )
        episode = next(value for value in episodes if value['episode'] == 0)
        length = len(episode['times'])
        indices = [episode['frame_offset']+index for index in (0, length//2, length-1)]
        path = Path(episode['video'])
        video_sha = file_sha256(path)
        frames = decode_torchcodec_frames(str(path), indices)
        result = audit_processor_batch(processor, frames, instruction)
        result['source'] = {'video': str(path.resolve()), 'video_sha256': video_sha,
                            'frame_indices': indices, 'parquet_sha256': episode['parquet_sha256']}
        if video_sha != file_sha256(path):
            raise ValueError('source video changed during audit')
        report['fixtures'][task] = result
    comparison_error = None
    if args.compare_report:
        try:
            report['comparison'] = compare_processor_reports(
                json.loads(args.compare_report.read_text()), report
            )
        except ValueError as error:
            comparison_error = error
            report['comparison'] = {'byte_equal': False, 'error': str(error),
                                    'authorizes_reuse': False}
        report['comparison']['baseline_report_sha256'] = file_sha256(args.compare_report)
    # Exclusive publication: reports are evidence, never overwritten or used as manifests.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'source': report['source_diagnostic'],
                      'comparison': report.get('comparison')}, ensure_ascii=False), flush=True)
    if comparison_error:
        raise comparison_error


if __name__ == '__main__':
    main()
