"""Closed-loop policy bridge for evaluating Zeta-0 in HumanoidArena (7 tasks).

Connects an exported or loaded ``PredictiveWorldPolicy`` (``zeta_native`` or
``gr00t_n1d7``) to HumanoidArena's 50 Hz SONIC semantic-40 / state-64 control
contract, with deterministic seeding, latency profiling, optional causal
``WorldReadout`` body-forecast logging, and JSONL episode reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.data.predictive_world.sonic_post_training import PROTOCOL, SPACE, STATE, TASKS

HUMANOIDARENA_ACTION_DIM = 40
HUMANOIDARENA_STATE_DIM = 64
HUMANOIDARENA_HZ = 50.0


@dataclass(frozen=True)
class EpisodeMetrics:
    task: str
    episode_index: int
    seed: int
    steps: int
    chunks_inferred: int
    success: bool
    mean_chunk_latency_ms: float
    p95_chunk_latency_ms: float
    host_architecture: str


class HumanoidArenaPolicyBridge:
    """Chunk-executing closed-loop bridge over a causal ``PredictiveWorldPolicy``."""

    def __init__(
        self,
        *,
        policy: Any,
        prepare_qwen_inputs: Callable[[Mapping[str, Any], str], dict[str, torch.Tensor]],
        action_horizon: int,
        execute_steps: int = 20,
        state_history_count: int = 1,
        host_architecture: str = "zeta_native",
        device: str | torch.device = "cpu",
    ) -> None:
        if action_horizon <= 0 or not 1 <= execute_steps <= action_horizon:
            raise ValueError("execute_steps must lie in [1, action_horizon]")
        if state_history_count <= 0:
            raise ValueError("state_history_count must be positive")
        self.policy = policy
        self.prepare_qwen_inputs = prepare_qwen_inputs
        self.action_horizon = int(action_horizon)
        self.execute_steps = int(execute_steps)
        self.state_history_count = int(state_history_count)
        self.host_architecture = str(host_architecture)
        self.device = torch.device(device)
        self._state_buffer: list[np.ndarray] = []
        self._chunk: np.ndarray | None = None
        self._chunk_cursor = 0
        self._latencies_ms: list[float] = []
        self._generator = torch.Generator(device=self.device)

    def reset(self, *, seed: int = 0) -> None:
        self._state_buffer.clear()
        self._chunk = None
        self._chunk_cursor = 0
        self._latencies_ms.clear()
        self._generator.manual_seed(int(seed))

    def _build_observed(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        committed_prefix: np.ndarray | None = None,
    ) -> ObservedHistory:
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (HUMANOIDARENA_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state [{HUMANOIDARENA_STATE_DIM}]")
        self._state_buffer.append(state.copy())
        if len(self._state_buffer) > self.state_history_count:
            self._state_buffer = self._state_buffer[-self.state_history_count :]
        padded = np.zeros((self.state_history_count, HUMANOIDARENA_STATE_DIM), dtype=np.float32)
        valid = np.zeros_like(padded, dtype=bool)
        times = np.zeros(self.state_history_count, dtype=np.float32)
        count = len(self._state_buffer)
        padded[-count:] = np.stack(self._state_buffer, axis=0)
        valid[-count:] = True
        times[-count:] = (np.arange(1 - count, 1, dtype=np.float32)) / HUMANOIDARENA_HZ

        committed = None
        if committed_prefix is not None and len(committed_prefix):
            prefix_arr = np.asarray(committed_prefix, dtype=np.float32)
            if prefix_arr.ndim != 2 or prefix_arr.shape[1] != HUMANOIDARENA_ACTION_DIM:
                raise ValueError("committed_prefix must have shape [P, 40]")
            pv = torch.zeros((1, self.action_horizon, HUMANOIDARENA_ACTION_DIM), device=self.device)
            pm = torch.zeros_like(pv, dtype=torch.bool)
            pt = torch.zeros((1, self.action_horizon), device=self.device)
            n = min(len(prefix_arr), self.action_horizon)
            pv[0, :n] = torch.as_tensor(prefix_arr[:n], device=self.device)
            pm[0, :n] = True
            pt[0, :n] = torch.arange(1, n + 1, device=self.device, dtype=torch.float32) / HUMANOIDARENA_HZ
            committed = NativeActionBatch(
                values=pv,
                valid=pm,
                times=pt,
                protocol_ids=(PROTOCOL,),
                native_dims=torch.tensor([HUMANOIDARENA_ACTION_DIM], device=self.device),
                normalization_spaces=(SPACE,),
            )

        qwen_inputs = self.prepare_qwen_inputs(observation, instruction)
        return ObservedHistory(
            qwen_inputs=qwen_inputs,
            raw_states=torch.from_numpy(padded).unsqueeze(0).to(self.device),
            state_valid=torch.from_numpy(valid).unsqueeze(0).to(self.device),
            state_times=torch.from_numpy(times).unsqueeze(0).to(self.device),
            state_schema_ids=(STATE,),
            action_protocol_ids=(PROTOCOL,),
            action_native_dims=torch.tensor([HUMANOIDARENA_ACTION_DIM], device=self.device),
            action_normalization_spaces=(SPACE,),
            committed_actions=committed,
        )

    def act(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        committed_prefix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one 40-D SONIC semantic action step, inferring a new chunk as needed."""
        if self._chunk is None or self._chunk_cursor >= self.execute_steps:
            observed = self._build_observed(observation, instruction, committed_prefix=committed_prefix)
            noise = torch.randn(
                1, self.action_horizon, HUMANOIDARENA_ACTION_DIM,
                device=self.device, generator=self._generator,
            )
            started = time.perf_counter()
            actions = self.policy.sample(observed, action_noise=noise)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._latencies_ms.append(elapsed_ms)
            chunk = actions[0, :, :HUMANOIDARENA_ACTION_DIM].detach().float().cpu().numpy()
            if chunk.shape != (self.action_horizon, HUMANOIDARENA_ACTION_DIM) or not np.isfinite(chunk).all():
                raise ValueError("Policy returned invalid HumanoidArena action chunk")
            self._chunk = chunk
            self._chunk_cursor = 0
        step_action = self._chunk[self._chunk_cursor]
        self._chunk_cursor += 1
        return step_action

    def run_episode(
        self,
        env: Any,
        *,
        task: str,
        instruction: str,
        episode_index: int = 0,
        seed: int = 0,
        max_steps: int = 500,
    ) -> EpisodeMetrics:
        """Run one closed-loop episode against any gym-like HumanoidArena environment."""
        if task not in TASKS:
            raise ValueError(f"Unknown HumanoidArena task {task!r}; expected one of {TASKS}")
        self.reset(seed=seed)
        obs = env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        success = False
        steps = 0
        for steps in range(1, max_steps + 1):
            action = self.act(obs, instruction)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, _reward, done, info = step_out
            if isinstance(info, Mapping) and info.get("success"):
                success = True
            if done:
                break
        arr = np.asarray(self._latencies_ms or [0.0], dtype=np.float64)
        return EpisodeMetrics(
            task=task,
            episode_index=int(episode_index),
            seed=int(seed),
            steps=steps,
            chunks_inferred=len(self._latencies_ms),
            success=success,
            mean_chunk_latency_ms=float(arr.mean()),
            p95_chunk_latency_ms=float(np.percentile(arr, 95)),
            host_architecture=self.host_architecture,
        )


def write_eval_summary(episodes: list[EpisodeMetrics], output_path: str | Path) -> dict[str, Any]:
    """Write per-episode JSONL and aggregate per-task & seven-task mean success rates."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[EpisodeMetrics]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task, []).append(ep)
    per_task = {
        task: {
            "episodes": len(items),
            "success_rate": sum(1 for item in items if item.success) / len(items),
            "mean_chunk_latency_ms": float(np.mean([item.mean_chunk_latency_ms for item in items])),
        }
        for task, items in sorted(by_task.items())
    }
    summary = {
        "format": "zeta0.humanoidarena_eval.v1",
        "tasks_evaluated": len(per_task),
        "total_episodes": len(episodes),
        "mean_success_rate": (
            float(np.mean([stats["success_rate"] for stats in per_task.values()]))
            if per_task
            else 0.0
        ),
        "per_task": per_task,
        "episodes": [asdict(ep) for ep in episodes],
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
