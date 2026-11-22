"""Train, save and resume the actual latent predictor on synthetic CPU/CUDA data.

This is a small installation demonstration, not a robot policy or benchmark.
No model downloads, camera input, network access or hardware control is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from psi.models.predictive_world.engine import (
    EngineConfig, PhaseLoss, load_engine_state, run_phase_training, save_engine_state,
)
from psi.models.predictive_world.latent_transition import TemporalLatentTransition


def build():
    return TemporalLatentTransition(
        {'synthetic': {'dim': 2, 'mean': [0., 0.], 'std': [1., 1.]}},
        context_dim=12, latent_shape=(3, 2, 4), hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32,
    )


def run(output: Path, device='cpu', steps=8, seed=7):
    if steps < 2:
        raise ValueError('At least two updates are needed to demonstrate resume')
    target_device = torch.device(device)
    if target_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build().to(target_device)
    context = torch.randn(4, 3, 12, device=target_device)
    target = context[..., :8].reshape(4, 3, 2, 4).tanh()
    valid = torch.ones(4, dtype=torch.bool, device=target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    identity = f'zeta0.synthetic-prior.v1/seed={seed}/steps={steps}'
    metrics = []

    def forward(module, batch):
        x, y, mask = batch
        loss = (module(x, mode='prior') - y).square().mean(dim=(1, 2, 3))
        return PhaseLoss(loss, mask, {'synthetic_mse': loss.mean()})

    def train(module, opt, start, stop):
        return run_phase_training(
            model=module, batches=iter([(context, target, valid)] * steps),
            optimizer=opt, scheduler=None,
            config=EngineConfig(phase='world_model', max_updates=stop, accumulation_steps=1,
                                max_grad_norm=1., log_every=1, checkpoint_every=steps),
            validity_fn=lambda batch: (batch[2], None), forward_fn=forward,
            device=target_device, start_update=start,
            on_update=lambda update, value: metrics.append({'update': update, 'mse': value['synthetic_mse']}),
        )

    initial = float(forward(model, (context, target, valid)).per_sample.mean().detach())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split = steps // 2
    train(model, optimizer, 0, split)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(output / 'predictor.safetensors'))
    save_engine_state(output, optimizer=optimizer, scheduler=None, completed_updates=split,
                      next_occurrence=split, config_identity=identity)
    restored = build().to(target_device)
    restored.load_state_dict(load_file(str(output / 'predictor.safetensors'), device=device), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(context, mode='prior'), restored(context, mode='prior'), rtol=0, atol=0)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    start, cursor = load_engine_state(output, optimizer=restored_optimizer, scheduler=None,
                                     config_identity=identity, device=target_device)
    if (start, cursor) != (split, split):
        raise RuntimeError('Unexpected resume identity')
    completed = train(restored, restored_optimizer, start, steps)
    with torch.no_grad():
        prediction = restored(context, mode='prior')
        final = float((prediction - target).square().mean())
    if not torch.isfinite(prediction).all() or not final < initial:
        raise RuntimeError('Synthetic learning check failed')
    digest = hashlib.sha256((output / 'predictor.safetensors').read_bytes()).hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True).stdout.strip()
    report = {'format': 'zeta0.quickstart.v1', 'data': 'synthetic', 'device': device,
              'source_commit': revision or None, 'torch': torch.__version__, 'seed': seed,
              'completed_updates': completed, 'resumed_at': start, 'checkpoint_sha256': digest,
              'initial_mse': initial, 'final_mse': final, 'prediction_shape': list(prediction.shape),
              'robot_commands_sent': False, 'output': str(output), 'metrics': metrics}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('runs') / ('quickstart-' + uuid.uuid4().hex[:8]))
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(run(args.output, args.device, args.steps), indent=2))
