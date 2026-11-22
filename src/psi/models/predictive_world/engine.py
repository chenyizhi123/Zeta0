"""Shared update-exact engine for all predictive-world training phases."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import random
import tempfile
from time import perf_counter
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from psi.data.predictive_world.cache import canonical_sha256
from psi.data.predictive_world.objective import UpdateLossNormalizer


@dataclass(frozen=True)
class PhaseLoss:
    per_sample: Tensor
    valid: Tensor
    metrics: Mapping[str, Tensor]
    sample_weight: Tensor | None = None
    metric_weights: Mapping[str, Tensor] | None = None

    def validate(self) -> None:
        if self.per_sample.ndim != 1 or self.valid.shape != self.per_sample.shape:
            raise ValueError("phase loss must expose aligned [B] values/validity")
        if self.sample_weight is not None and self.sample_weight.shape != self.per_sample.shape:
            raise ValueError("phase sample weights must have shape [B]")
        if self.metric_weights is not None and self.metric_weights.keys() != self.metrics.keys():
            raise ValueError("phase metric weights must have exactly the metric names")
        if not torch.isfinite(self.per_sample[self.valid.bool()]).all():
            raise FloatingPointError("phase produced non-finite valid per-sample loss")


@dataclass(frozen=True)
class EngineConfig:
    phase: str
    max_updates: int
    accumulation_steps: int
    max_grad_norm: float
    log_every: int = 10
    checkpoint_every: int = 3000
    first_checkpoint_update: int | None = None
    autocast_dtype: torch.dtype = torch.bfloat16

    def validate(self) -> None:
        if self.first_checkpoint_update is not None and self.first_checkpoint_update <= 0:
            raise ValueError("first_checkpoint_update must be positive")
        if self.phase not in {"world_model", "joint", "policy"}:
            raise ValueError("unsupported predictive-world phase")
        if min(self.max_updates, self.accumulation_steps, self.log_every, self.checkpoint_every) <= 0:
            raise ValueError("engine update/cadence values must be positive")
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_ranks_true(value: bool, *, device: torch.device) -> bool:
    """Return one synchronized boolean without hiding rank-local exceptions."""

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def publish_run_contract(path: Path, contract: Mapping[str, Any], *, device: torch.device) -> None:
    """Publish on rank zero, then propagate failure before any rank continues."""
    rank, _ = _world()
    error = None
    if rank == 0:
        temporary = None
        try:
            if not path.exists():
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.", delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(contract, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # Atomic visibility without overwriting another run's contract.
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            if json.loads(path.read_text(encoding="utf-8")) != contract:
                raise ValueError("output directory belongs to a different run contract")
        except Exception as caught:
            error = caught
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if not _all_ranks_true(error is None, device=device):
        raise RuntimeError("run contract publication failed on rank 0") from error
    if error is not None:
        raise error


def _validate_update_validity(
    validity: list[tuple[Tensor, Tensor | None]],
) -> None:
    """Validate all local denominator inputs before its first collective."""

    if not validity:
        raise ValueError("an optimizer update must contain at least one micro-batch")
    for valid, weight in validity:
        if not isinstance(valid, Tensor) or valid.ndim != 1:
            raise ValueError("update validity must be a tensor with shape [B]")
        if weight is not None:
            if not isinstance(weight, Tensor) or weight.shape != valid.shape:
                raise ValueError("update sample weights must have shape [B]")
            if not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("update sample weights must be finite and non-negative")


def _read_only_fp64_grad_norm(parameters: list[nn.Parameter], *, device: torch.device) -> Tensor:
    """Measure the local L2 gradient norm in FP64 without changing gradients."""

    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        component = torch.linalg.vector_norm(values, ord=2, dtype=torch.float64)
        squared_norm.add_(component.square())
    return squared_norm.sqrt()


def _clip_gradients_(parameters: list[nn.Parameter], *, norm: Tensor, maximum: float) -> None:
    """Clip only after the synchronized read-only finite check has succeeded."""

    coefficient = min(1.0, float(maximum / (norm.item() + 1e-12)))
    if coefficient >= 1.0:
        return
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient._values().mul_(coefficient)
        else:
            gradient.mul_(coefficient)


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move nested typed batches without converting metadata strings/tuples."""

    if isinstance(value, Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            field.name: move_to_device(getattr(value, field.name), device)
            for field in dataclasses.fields(value)
        })
    if isinstance(value, Mapping):
        return type(value)((key, move_to_device(item, device)) for key, item in value.items())
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng(value: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("checkpoint contains CUDA RNG but resume device is CPU")
        torch.cuda.set_rng_state(value["cuda"], device=device)


def save_engine_state(
    directory: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    completed_updates: int,
    next_occurrence: int,
    config_identity: str,
) -> Path:
    rank, world_size = _world()
    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"trainer_state_rank{rank:04d}.pt"
    temporary = output / f".{path.name}.{uuid.uuid4().hex}.tmp"
    torch.save({
        "format": "zeta0.predictive_world.trainer.v1",
        "rank": rank,
        "world_size": world_size,
        "completed_updates": int(completed_updates),
        "next_occurrence": int(next_occurrence),
        "config_identity": str(config_identity),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng": _rng_state(),
    }, temporary)
    temporary.replace(path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        marker_tmp = output / f".trainer_complete.{uuid.uuid4().hex}.tmp"
        marker_tmp.write_text(f"{completed_updates}\n", encoding="utf-8")
        marker_tmp.replace(output / ".trainer_complete")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path


def load_engine_state(
    directory: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config_identity: str,
    device: torch.device,
) -> tuple[int, int]:
    rank, world_size = _world()
    root = Path(directory).expanduser().resolve()
    if not (root / ".trainer_complete").is_file():
        raise ValueError("predictive-world trainer checkpoint is incomplete")
    state = torch.load(root / f"trainer_state_rank{rank:04d}.pt", map_location="cpu", weights_only=False)
    if (
        state.get("format") != "zeta0.predictive_world.trainer.v1"
        or state.get("rank") != rank
        or state.get("world_size") != world_size
        or state.get("config_identity") != config_identity
    ):
        raise ValueError("trainer resume rank/world/config identity mismatch")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("resume checkpoint lacks scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"], device)
    return int(state["completed_updates"]), int(state["next_occurrence"])


def run_phase_training(
    *,
    model: nn.Module,
    batches: Iterator[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: EngineConfig,
    validity_fn: Callable[[Any], tuple[Tensor, Tensor | None]],
    forward_fn: Callable[[nn.Module, Any], PhaseLoss],
    device: torch.device,
    start_update: int = 0,
    on_update: Callable[[int, Mapping[str, float]], None] | None = None,
    on_checkpoint: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    prepare_update: Callable[[int, list[Any]], None] | None = None,
    on_step_end: Callable[[int], None] | None = None,
    log_due: Callable[[int], bool] | None = None,
) -> int:
    """Run updates with one global denominator across every rank and microbatch."""

    config.validate()
    rank, world_size = _world()
    normalizer = UpdateLossNormalizer(distributed=world_size > 1)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    for update in range(start_update, config.max_updates):
        step_started = perf_counter()
        input_wait_seconds = 0.0
        input_error: Exception | None = None
        micros: list[Any] = []
        validity: list[tuple[Tensor, Tensor | None]] = []
        try:
            for _ in range(config.accumulation_steps):
                waiting = perf_counter()
                batch = next(batches)
                input_wait_seconds += perf_counter() - waiting
                micros.append(move_to_device(batch, device))
            if prepare_update is not None:
                prepare_update(update, micros)
            validity = [validity_fn(batch) for batch in micros]
            _validate_update_validity(validity)
        except Exception as error:
            input_error = error
        if not _all_ranks_true(input_error is None, device=device):
            raise RuntimeError(
                f"input preparation failed on at least one rank at update {update + 1}"
            ) from input_error
        if input_error is not None:
            raise input_error
        denominator = normalizer.denominator(
            [item[0] for item in validity],
            None if all(item[1] is None for item in validity) else [
                torch.ones_like(item[0], dtype=torch.float32) if item[1] is None else item[1]
                for item in validity
            ],
        )
        metric_numerators: dict[str, Tensor] = {}
        metric_denominators: dict[str, Tensor] = {}
        for micro_index, batch in enumerate(micros):
            synchronize = world_size == 1 or micro_index == len(micros) - 1
            sync_context = contextlib.nullcontext()
            if not synchronize and hasattr(model, "no_sync"):
                sync_context = model.no_sync()
            forward_error: Exception | None = None
            loss: Tensor | None = None
            local_metric_numerators: dict[str, Tensor] = {}
            local_metric_denominators: dict[str, Tensor] = {}
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=config.autocast_dtype,
                    enabled=device.type == "cuda",
                ):
                    # Unexpected exceptions inside a DDP forward cannot safely be
                    # followed by a new collective: an internal collective may
                    # already be incomplete. Launcher teardown and the PG timeout
                    # own that failure mode. Completed forwards are synchronized
                    # below before any rank is allowed to enter backward.
                    result = forward_fn(model, batch)
                    try:
                        result.validate()
                        expected_valid, expected_weight = validity[micro_index]
                        if not torch.equal(result.valid.bool(), expected_valid.bool()):
                            raise RuntimeError(
                                "forward validity differs from precomputed update denominator"
                            )
                        loss = normalizer.scale_micro(
                            result.per_sample,
                            result.valid,
                            denominator,
                            sample_weight=expected_weight,
                            graph_tensor=result.per_sample,
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"non-finite {config.phase} loss at update {update + 1}"
                            )
                        default_metric_weight = result.valid.detach().double().sum()
                        for name, value in result.metrics.items():
                            if value.numel() != 1 or not torch.isfinite(value):
                                raise FloatingPointError(f"invalid scalar metric {name!r}")
                            weight = (
                                default_metric_weight
                                if result.metric_weights is None
                                else result.metric_weights[name].detach().to(
                                    device=value.device, dtype=torch.float64
                                )
                            )
                            if weight.numel() != 1 or not torch.isfinite(weight) or weight < 0:
                                raise FloatingPointError(f"invalid metric weight {name!r}")
                            local_metric_numerators[name] = value.detach().double() * weight
                            local_metric_denominators[name] = weight
                    except Exception as error:
                        forward_error = error
                # Keep backward inside DDP.no_sync(); restoring the flag before
                # backward would silently synchronize every accumulation micro.
                if not _all_ranks_true(forward_error is None, device=device):
                    raise FloatingPointError(
                        f"invalid {config.phase} forward result on at least one rank "
                        f"at update {update + 1}"
                    ) from forward_error
                if forward_error is not None:
                    raise forward_error
                assert loss is not None
                loss.backward()
            for name in local_metric_numerators:
                metric_numerators[name] = (
                    metric_numerators.get(name, torch.zeros((), device=device, dtype=torch.float64))
                    + local_metric_numerators[name]
                )
                metric_denominators[name] = (
                    metric_denominators.get(name, torch.zeros((), device=device, dtype=torch.float64))
                    + local_metric_denominators[name]
                )
        gradient_error: Exception | None = None
        try:
            norm = _read_only_fp64_grad_norm(trainable, device=device)
            gradients_finite = bool(torch.isfinite(norm).item())
        except Exception as error:
            gradient_error = error
            gradients_finite = False
            norm = torch.full((), float("nan"), dtype=torch.float64, device=device)
        if not _all_ranks_true(gradients_finite, device=device):
            raise FloatingPointError(
                f"non-finite gradient on at least one rank at update {update + 1}"
            ) from gradient_error
        if gradient_error is not None:
            raise gradient_error
        if world_size > 1:
            dist.all_reduce(norm, op=dist.ReduceOp.MAX)
        _clip_gradients_(trainable, norm=norm, maximum=config.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed = update + 1
        if on_update is not None and (
            log_due(completed) if log_due is not None
            else completed == 1 or completed % config.log_every == 0
        ):
            names = sorted(metric_numerators)
            global_valid_weight = torch.tensor(
                [float(denominator.global_weight)], device=device, dtype=torch.float64,
            )
            metric_values = torch.stack(
                [metric_numerators[name] for name in names]
                + [metric_denominators[name] for name in names]
            ) if names else torch.empty(0, device=device, dtype=torch.float64)
            if world_size > 1:
                # This denominator is already global. MAX checks/retains its
                # identical value without multiplying it by world size.
                dist.all_reduce(global_valid_weight, op=dist.ReduceOp.MAX)
                if metric_values.numel():
                    dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
            values = {}
            for index, name in enumerate(names):
                metric_denominator = metric_values[index + len(names)]
                values[name] = (
                    float(metric_values[index] / metric_denominator)
                    if metric_denominator.item() > 0 else 0.0
                )
            values["grad_norm"] = float(norm)
            values["global_valid_weight"] = float(global_valid_weight[0])
            if rank == 0:
                # Only synchronize at the existing log cadence. This sample
                # includes completed GPU work, but excludes logging/checkpoints.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = perf_counter() - step_started
                values["perf/step_seconds"] = seconds
                values["perf/input_wait_seconds"] = input_wait_seconds
                values["perf/input_wait_fraction"] = input_wait_seconds / max(seconds, 1e-9)
                values["perf/rank_samples_per_second"] = sum(
                    item[0].numel() for item in validity
                ) / max(seconds, 1e-9)
                on_update(completed, values)
        if on_step_end is not None:
            on_step_end(completed)
        checkpoint_due = (
            completed == config.first_checkpoint_update
            or completed % config.checkpoint_every == 0
            or completed == config.max_updates
        )
        if on_checkpoint is not None and checkpoint_due:
            on_checkpoint(completed)
        stop = bool(should_stop is not None and should_stop())
        if world_size > 1:
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            stop = bool(stop_tensor.item())
        if stop:
            if on_checkpoint is not None and not checkpoint_due:
                on_checkpoint(completed)
            return completed
    return config.max_updates


def engine_config_identity(config: Mapping[str, Any]) -> str:
    """Bind resume to resolved semantics, optimizer and topology."""

    return canonical_sha256(dict(config))


__all__ = [
    "EngineConfig",
    "PhaseLoss",
    "engine_config_identity",
    "load_engine_state",
    "move_to_device",
    "run_phase_training",
    "save_engine_state",
]
