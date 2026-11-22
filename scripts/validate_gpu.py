"""Require real CUDA and run bounded predictor/native-head GPU contract checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from psi.models.predictive_world.gr00t import native_types
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU acceptance requires CUDA; CPU/skipped runs are not accepted')
    native_types()  # Missing/mismatched upstream source must fail before pytest can skip it.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, 'PW_CUDA_SMOKE': '1', 'OMP_NUM_THREADS': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'NO_ALBUMENTATIONS_UPDATE': '1', 'PYTHONPATH': str(ROOT / 'src')}
    subprocess.run([sys.executable, str(ROOT / 'scripts/examples/quickstart.py'),
                    '--device', 'cuda', '--output', str(output / 'quickstart')],
                   cwd=ROOT, env=environment, check=True, timeout=120)
    xml_path = output / 'gpu-tests.xml'
    command = [sys.executable, '-m', 'pytest', '-v',
               'tests/predictive_world/test_phases.py::test_cuda_bf16_world_and_joint_optimizer_steps',
               'tests/predictive_world/test_gr00t.py', '-k', 'cuda',
               '--junitxml=' + str(xml_path)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True, timeout=300)
    cases = ET.parse(xml_path).findall('.//testcase')
    if len(cases) < 3 or any(case.find('skipped') is not None for case in cases):
        raise RuntimeError('Required CUDA contracts were skipped or not selected')
    report = {'format': 'zeta0.gpu-acceptance.v1', 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'tests_passed': len(cases),
              'tests_skipped': 0, 'full_model_or_robot_acceptance': False}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
