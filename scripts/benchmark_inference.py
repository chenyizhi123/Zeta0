#!/usr/bin/env python3
"""Benchmark deploy-only Zeta-0 foresight + action expert latency (eager vs torch.compile).

Measures the causal System 1.5 + System 1 chain (``PredictiveHistoryEncoder`` +
``TemporalLatentTransition`` + ``TemporalNativeActionExpert``) on CPU or CUDA
without requiring external model weights or robot hardware.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from psi.data.predictive_world.contracts import ObservedHistory
from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import (
    LatentNormalizer,
    TemporalLatentTransition,
)
from psi.models.predictive_world.policy import ObservationFeatures, PredictiveWorldPolicy

PRESETS = {
    "tiny-cpu": dict(
        qwen_dim=64, state_dim=41, action_dim=76, horizon=30, solver_steps=4,
        history_dim=64, history_blocks=2, history_heads=4,
        transition_dim=64, transition_blocks=2, transition_heads=4,
        expert_dim=64, expert_blocks=2, expert_heads=4,
        latent_shape=(3, 8, 32), context_len=16,
    ),
    "paper-psi0": dict(
        qwen_dim=2048, state_dim=41, action_dim=76, horizon=50, solver_steps=10,
        history_dim=1024, history_blocks=16, history_heads=16,
        transition_dim=768, transition_blocks=8, transition_heads=12,
        expert_dim=768, expert_blocks=8, expert_heads=12,
        latent_shape=(3, 8, 256), context_len=64,
    ),
}


class _SyntheticObservationEncoder(torch.nn.Module):
    def __init__(self, features: ObservationFeatures) -> None:
        super().__init__()
        self._features = features

    def forward(self, _observed: ObservedHistory) -> ObservationFeatures:
        return self._features


def benchmark(
    *,
    preset: str = "tiny-cpu",
    device: str = "cpu",
    warmup: int = 2,
    iterations: int = 8,
    use_compile: bool = False,
) -> dict:
    spec = PRESETS[preset]
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocols = {
        "bench": {
            "dim": spec["action_dim"],
            "mean": [0.0] * spec["action_dim"],
            "std": [1.0] * spec["action_dim"],
        }
    }
    state_dims = {"bench_state": spec["state_dim"]}
    state_stats = {
        "bench_state": (torch.zeros(spec["state_dim"]), torch.ones(spec["state_dim"]))
    }
    features = ObservationFeatures(
        qwen_context=torch.randn(1, spec["context_len"], spec["qwen_dim"], device=target_device),
        context_valid=torch.ones(1, spec["context_len"], dtype=torch.bool, device=target_device),
        current_visual=torch.randn(1, 16, spec["qwen_dim"], device=target_device),
    )
    adapters = StateAdapterBank(state_dims, statistics=state_stats)
    history = PredictiveHistoryEncoder(
        qwen_dim=spec["qwen_dim"],
        visual_dim=spec["qwen_dim"],
        state_history_count=1,
        output_tokens=16,
        context_tokens=16,
        hidden_dim=spec["history_dim"],
        num_blocks=spec["history_blocks"],
        num_heads=spec["history_heads"],
        ff_dim=spec["history_dim"] * 4,
    )
    transition = TemporalLatentTransition(
        protocols,
        context_dim=spec["history_dim"],
        latent_shape=spec["latent_shape"],
        group_ends_s=(0.2, 0.4, 0.6),
        hidden_dim=spec["transition_dim"],
        num_blocks=spec["transition_blocks"],
        num_heads=spec["transition_heads"],
        ff_dim=spec["transition_dim"] * 4,
    )
    student = TemporalNativeActionExpert(
        protocols,
        state_dims,
        state_statistics=state_stats,
        qwen_dim=spec["qwen_dim"],
        context_tokens=16,
        latent_dim=spec["latent_shape"][2],
        latent_groups=spec["latent_shape"][0],
        latent_slots=spec["latent_shape"][1],
        state_history_count=1,
        action_steps=spec["horizon"],
        hidden_dim=spec["expert_dim"],
        num_blocks=spec["expert_blocks"],
        num_heads=spec["expert_heads"],
        ff_dim=spec["expert_dim"] * 4,
    )
    if use_compile:
        history = torch.compile(history)
        transition = torch.compile(transition)

    policy = PredictiveWorldPolicy(
        observation_encoder=_SyntheticObservationEncoder(features),
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=transition,
        latent_normalizer=LatentNormalizer(spec["latent_shape"]),
        student=student,
        action_solver_steps=spec["solver_steps"],
        latent_prediction_space="raw",
    ).to(target_device).eval()

    observed = ObservedHistory(
        qwen_inputs={},
        raw_states=torch.zeros(1, 1, spec["state_dim"], device=target_device),
        state_valid=torch.ones(1, 1, spec["state_dim"], dtype=torch.bool, device=target_device),
        state_times=torch.zeros(1, 1, device=target_device),
        state_schema_ids=("bench_state",),
        action_protocol_ids=("bench",),
        action_native_dims=torch.tensor([spec["action_dim"]], device=target_device),
        action_normalization_spaces=("native",),
    )
    noise = torch.randn(1, spec["horizon"], spec["action_dim"], device=target_device)

    for _ in range(warmup):
        policy.sample(observed, action_noise=noise)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)

    totals_ms = []
    stage_samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        timings: dict[str, float] = {}
        started = time.perf_counter()
        policy.sample(observed, action_noise=noise, stage_timings=timings)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        totals_ms.append((time.perf_counter() - started) * 1000.0)
        for key, value in timings.items():
            stage_samples.setdefault(key, []).append(value * 1000.0)

    arr = np.asarray(totals_ms, dtype=np.float64)
    return {
        "format": "zeta0.inference_benchmark.v1",
        "preset": preset,
        "device": device,
        "compiled": use_compile,
        "iterations": iterations,
        "latency_ms": {
            "mean": round(float(arr.mean()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
        },
        "stages_mean_ms": {k: round(float(np.mean(v)), 3) for k, v in sorted(stage_samples.items())},
        "peak_cuda_mib": (
            round(torch.cuda.max_memory_allocated(target_device) / (1024 ** 2), 2)
            if target_device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny-cpu")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                preset=args.preset,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
                use_compile=args.compile,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
