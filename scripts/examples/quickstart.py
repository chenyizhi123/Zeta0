"""End-to-end Zeta-0 quickstart: joint scene-and-body foresight + dual-response transfer.

Runs the full Zeta-0 architecture on synthetic kinematic trajectories (CPU or CUDA):
- Online/EMA future encoder E_phi / E_phi_bar + scene/body readout D
- Causal conditioner H + shared prior/world predictor F_theta
- Independent flow-matching Teacher T_psi and Executing Policy G_omega
- Dual-response transfer L_resp (raw latent space, v6 objective)
- Exact midpoint checkpoint save/reload/resume via ``run_phase_training``
- Deploy-only causal ``PredictiveWorldPolicy`` rollout (no E/D/T at inference)
- Automatic ``foresight_preview.png`` generation for immediate visual inspection

No external model downloads, datasets, network access, or robot hardware are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.framework.visualize import render_foresight_preview
from psi.models.predictive_world.action_expert import (
    TemporalNativeActionExpert,
    clone_independent_expert,
)
from psi.models.predictive_world.engine import (
    EngineConfig,
    load_engine_state,
    run_phase_training,
    save_engine_state,
)
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import (
    LatentNormalizer,
    TemporalLatentTransition,
)
from psi.models.predictive_world.phases import (
    ForecastObjective,
    JointObjective,
    PolicyObjective,
    RepresentationObjective,
    WorldModelObjective,
)
from psi.models.predictive_world.policy import (
    ObservationFeatures,
    PredictiveWorldPolicy,
)
from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
    WorldReadout,
)

VISUAL_DIM = 16
MEASUREMENT_DIM = 6
ACTION_DIM = 6
ACTION_STEPS = 6
GROUPS = 3
SLOTS = 8
LATENT_DIM = 8
BODY_POINTS = 2
SCENE_PATCHES = 4
PROTOCOLS = {
    "synthetic": {
        "dim": ACTION_DIM,
        "mean": [0.0] * ACTION_DIM,
        "std": [1.0] * ACTION_DIM,
    }
}
STATE_DIMS = {"synthetic_state": MEASUREMENT_DIM}


def build_components():
    """Construct a compact full-algorithm Zeta-0 joint training stack."""
    state_stats = {
        "synthetic_state": (torch.zeros(MEASUREMENT_DIM), torch.ones(MEASUREMENT_DIM))
    }
    online = FutureTrajectoryEncoder(
        visual_dim=VISUAL_DIM,
        measurement_dim=MEASUREMENT_DIM,
        latent_shape=(GROUPS, SLOTS, LATENT_DIM),
        scene_slots=4,
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    encoder = EMAFutureTrajectoryEncoder(online, decay=0.95)
    readout = WorldReadout(
        visual_dim=VISUAL_DIM,
        measurement_dim=MEASUREMENT_DIM,
        latent_shape=(GROUPS, SLOTS, LATENT_DIM),
        max_scene_patches=SCENE_PATCHES,
        body_points_per_group=BODY_POINTS,
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    teacher = TemporalNativeActionExpert(
        PROTOCOLS,
        STATE_DIMS,
        state_statistics=state_stats,
        qwen_dim=VISUAL_DIM,
        context_tokens=4,
        latent_dim=LATENT_DIM,
        latent_groups=GROUPS,
        latent_slots=SLOTS,
        state_history_count=1,
        action_steps=ACTION_STEPS,
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    student = clone_independent_expert(teacher)
    adapters = StateAdapterBank(STATE_DIMS, statistics=state_stats)
    history = PredictiveHistoryEncoder(
        qwen_dim=VISUAL_DIM,
        visual_dim=VISUAL_DIM,
        state_history_count=1,
        output_tokens=SCENE_PATCHES,
        context_tokens=4,
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    transition = TemporalLatentTransition(
        PROTOCOLS,
        context_dim=history.hidden_dim,
        latent_shape=(GROUPS, SLOTS, LATENT_DIM),
        group_ends_s=(0.2, 0.4, 0.6),
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    normalizer = LatentNormalizer((GROUPS, SLOTS, LATENT_DIM))
    representation = RepresentationObjective(
        encoder,
        readout,
        teacher,
        scene_weight=1.0,
        body_weight=1.0,
        action_weight=1.0,
        time_weight=0.25,
        regularizer_weight=1e-3,
    )
    forecast = ForecastObjective(
        reference_encoder=encoder,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=transition,
        control_teacher=teacher,
        control_reader="both",
        control_weight=1.0,
        readout=readout,
        readout_weight=0.25,
        prior_weight=1.0,
        world_weight=1.0,
        freeze_reference=False,
        latent_prediction_space="raw",
    )
    world = WorldModelObjective(representation, forecast, normalizer)
    policy_obj = PolicyObjective(
        student=student,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=transition,
        freeze_world=False,
        latent_prediction_space="raw",
    )
    joint = JointObjective(world, policy_obj)
    return {
        "joint": joint,
        "encoder": encoder,
        "readout": readout,
        "teacher": teacher,
        "student": student,
        "adapters": adapters,
        "history": history,
        "transition": transition,
        "normalizer": normalizer,
    }


def build_synthetic_batch(device: torch.device, batch_size: int = 4) -> dict:
    """Create smooth synthetic kinematic trajectories across +0.2 / +0.4 / +0.6 s."""
    phases = torch.linspace(0.0, math.pi * 0.75, batch_size, device=device)
    freqs = torch.linspace(1.0, 2.2, MEASUREMENT_DIM, device=device)
    t0 = torch.zeros(batch_size, 1, device=device)
    current_meas = torch.sin(phases[:, None] + t0 * freqs[None]) * 0.6

    horizon_t = torch.linspace(0.1, 0.6, GROUPS * BODY_POINTS, device=device)
    future_meas_flat = torch.sin(
        phases[:, None, None] + horizon_t[None, :, None] * freqs[None, None, :] * 3.0
    ) * 0.6
    future_meas = future_meas_flat.reshape(batch_size, GROUPS, BODY_POINTS, MEASUREMENT_DIM)
    future_meas_times = horizon_t.reshape(1, GROUPS, BODY_POINTS).expand(batch_size, -1, -1)

    anchor_patches = torch.zeros(batch_size, SCENE_PATCHES, VISUAL_DIM, device=device)
    anchor_patches[:, :, :MEASUREMENT_DIM] = current_meas[:, None, :]
    anchor_patches[:, :, MEASUREMENT_DIM : 2 * MEASUREMENT_DIM] = torch.cos(phases[:, None, None])

    future_patches = torch.zeros(batch_size, GROUPS, SCENE_PATCHES, VISUAL_DIM, device=device)
    group_end_meas = future_meas[:, :, -1, :]
    future_patches[:, :, :, :MEASUREMENT_DIM] = group_end_meas[:, :, None, :]
    future_patches[:, :, :, MEASUREMENT_DIM : 2 * MEASUREMENT_DIM] = (
        group_end_meas[:, :, None, :] - current_meas[:, None, None, :]
    )

    action_times = torch.linspace(0.1, 0.6, ACTION_STEPS, device=device)[None].expand(batch_size, -1)
    actions = NativeActionBatch(
        values=future_meas_flat,
        valid=torch.ones(batch_size, ACTION_STEPS, ACTION_DIM, dtype=torch.bool, device=device),
        times=action_times,
        protocol_ids=("synthetic",) * batch_size,
        native_dims=torch.full((batch_size,), ACTION_DIM, dtype=torch.long, device=device),
        normalization_spaces=("native",) * batch_size,
    )
    observed = ObservedHistory(
        qwen_inputs={},
        raw_states=current_meas[:, None, :],
        state_valid=torch.ones(batch_size, 1, MEASUREMENT_DIM, dtype=torch.bool, device=device),
        state_times=torch.zeros(batch_size, 1, device=device),
        state_schema_ids=("synthetic_state",) * batch_size,
        action_protocol_ids=("synthetic",) * batch_size,
        action_native_dims=torch.full((batch_size,), ACTION_DIM, dtype=torch.long, device=device),
        action_normalization_spaces=("native",) * batch_size,
    )
    features = ObservationFeatures(
        qwen_context=anchor_patches,
        context_valid=torch.ones(batch_size, SCENE_PATCHES, dtype=torch.bool, device=device),
        current_visual=anchor_patches,
    )
    return {
        "observed": observed,
        "features": features,
        "actions": actions,
        "anchor_patches": anchor_patches,
        "anchor_patch_valid": torch.ones(batch_size, SCENE_PATCHES, dtype=torch.bool, device=device),
        "current_measurements": current_meas,
        "current_measurement_valid": torch.ones(batch_size, MEASUREMENT_DIM, dtype=torch.bool, device=device),
        "future_patches": future_patches,
        "future_patch_valid": torch.ones(batch_size, GROUPS, SCENE_PATCHES, dtype=torch.bool, device=device),
        "future_scene_times": torch.tensor([[0.2, 0.4, 0.6]], device=device).expand(batch_size, -1),
        "future_measurements": future_meas,
        "future_measurement_valid": torch.ones(batch_size, GROUPS, BODY_POINTS, MEASUREMENT_DIM, dtype=torch.bool, device=device),
        "future_measurement_times": future_meas_times,
        "group_valid": torch.ones(batch_size, GROUPS, dtype=torch.bool, device=device),
        "action_noise": torch.zeros(batch_size, ACTION_STEPS, ACTION_DIM, device=device),
        "action_sigma": torch.full((batch_size,), 0.5, device=device),
        "_action_prefix_lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
        "_rtc_training_max_delay": 2,
    }


def _evaluate_losses(joint: JointObjective, batch: dict, *, update: int) -> tuple[float, float]:
    with torch.no_grad():
        sample = {**batch, "_optimizer_update": update}
        out = joint(sample)
        total = float(out.per_sample.mean().item())
        dual = float(out.metrics.get("prior_control_response_mse", torch.tensor(0.0)).item())
        return total, dual


def _predict_body_preview(parts: dict, batch: dict) -> tuple[list[list[float]], list[list[float]]]:
    with torch.no_grad():
        obs = batch["observed"]
        feat = batch["features"]
        time_valid = obs.state_valid.any(dim=-1)
        adapted = parts["adapters"](
            obs.raw_states, obs.state_valid, list(obs.state_schema_ids),
            state_present=time_valid.any(dim=1),
        )
        condition = parts["history"](
            qwen_context=feat.qwen_context,
            context_valid=feat.context_valid,
            current_visual=feat.current_visual,
            adapted_states=adapted,
            state_valid=time_valid,
            state_times=obs.state_times,
        )
        z_pred = parts["transition"](
            condition,
            context_valid=torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device),
            mode="prior",
        )
        _, pred_body = parts["readout"](
            anchor_patches=batch["anchor_patches"],
            anchor_patch_valid=batch["anchor_patch_valid"],
            z_raw=z_pred,
            group_valid=batch["group_valid"],
            scene_patch_count=SCENE_PATCHES,
        )
        predicted = pred_body[0].reshape(GROUPS * BODY_POINTS, MEASUREMENT_DIM).cpu().tolist()
        target = batch["future_measurements"][0].reshape(GROUPS * BODY_POINTS, MEASUREMENT_DIM).cpu().tolist()
        return predicted, target


def _verify_causal_deploy_policy(parts: dict, batch: dict, device: torch.device) -> list[int]:
    class _FixedEncoder(torch.nn.Module):
        def __init__(self, features: ObservationFeatures) -> None:
            super().__init__()
            self._features = features

        def forward(self, _observed: ObservedHistory) -> ObservationFeatures:
            return self._features

    deploy_policy = PredictiveWorldPolicy(
        observation_encoder=_FixedEncoder(batch["features"]),
        history_state_adapters=parts["adapters"],
        history_encoder=parts["history"],
        latent_transition=parts["transition"],
        latent_normalizer=parts["normalizer"],
        student=parts["student"],
        action_solver_steps=4,
        latent_prediction_space="raw",
    ).eval()
    for forbidden in ("encoder", "readout", "teacher"):
        if hasattr(deploy_policy, forbidden):
            raise RuntimeError(f"Deploy policy must not retain training-only module {forbidden}")
    noise = torch.zeros(batch["observed"].raw_states.shape[0], ACTION_STEPS, ACTION_DIM, device=device)
    actions = deploy_policy.sample(batch["observed"], action_noise=noise)
    if not torch.isfinite(actions).all():
        raise RuntimeError("Causal deploy policy produced non-finite actions")
    return list(actions.shape)


def run(output: Path, device: str = "cpu", steps: int = 8, seed: int = 7) -> dict:
    if steps < 2:
        raise ValueError("At least two updates are needed to demonstrate resume")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parts = build_components()
    joint = parts["joint"].to(target_device)
    batch = build_synthetic_batch(target_device)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=0.015)
    identity = f"zeta0.full-joint-dual-response.v2/seed={seed}/steps={steps}"
    metrics: list[dict[str, float]] = []

    def forward(module, item):
        return module(item)

    def make_batches(start: int, stop: int):
        for step_idx in range(start, stop):
            yield {**batch, "_optimizer_update": step_idx}

    def train(module, opt, start: int, stop: int) -> int:
        return run_phase_training(
            model=module,
            batches=make_batches(start, stop),
            optimizer=opt,
            scheduler=None,
            config=EngineConfig(
                phase="joint",
                max_updates=stop,
                accumulation_steps=1,
                max_grad_norm=1.0,
                log_every=1,
                checkpoint_every=steps,
            ),
            validity_fn=lambda item: (item["group_valid"].any(dim=1), None),
            forward_fn=forward,
            device=target_device,
            start_update=start,
            on_update=lambda update, value: metrics.append({
                "update": update,
                "mse": float(value.get("joint_objective", 0.0)),
                "total_loss": float(value.get("joint_objective", 0.0)),
                "prior_body_mse": float(value.get("prior_body_mse", 0.0)),
                "prior_prediction_mse": float(value.get("prior_prediction_mse", 0.0)),
                "dual_response": float(value.get("prior_control_response_mse", 0.0)),
                "student_action_mse": float(value.get("student_action_mse", 0.0)),
            }),
        )

    initial_loss, initial_dual = _evaluate_losses(joint, batch, update=0)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split = steps // 2

    checkpoint_keys = (
        "encoder", "readout", "teacher", "student",
        "adapters", "history", "transition", "normalizer",
    )

    train(joint, optimizer, 0, split)
    serialized = {
        f"{name}.{key}": tensor.contiguous()
        for name in checkpoint_keys
        for key, tensor in parts[name].state_dict().items()
    }
    save_file(serialized, str(output / "predictor.safetensors"))
    save_engine_state(
        output,
        optimizer=optimizer,
        scheduler=None,
        completed_updates=split,
        next_occurrence=split,
        config_identity=identity,
    )

    restored_parts = build_components()
    restored = restored_parts["joint"].to(target_device)
    loaded = load_file(str(output / "predictor.safetensors"), device=device)
    for name in checkpoint_keys:
        prefix = name + "."
        sub_state = {k[len(prefix):]: v for k, v in loaded.items() if k.startswith(prefix)}
        restored_parts[name].load_state_dict(sub_state, strict=True)
    with torch.no_grad():
        before_reload = joint({**batch, "_optimizer_update": split}).per_sample
        after_reload = restored({**batch, "_optimizer_update": split}).per_sample
        torch.testing.assert_close(before_reload, after_reload, rtol=0, atol=0)

    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.015)
    start, cursor = load_engine_state(
        output,
        optimizer=restored_optimizer,
        scheduler=None,
        config_identity=identity,
        device=target_device,
    )
    if (start, cursor) != (split, split):
        raise RuntimeError("Unexpected resume identity")

    completed = train(restored, restored_optimizer, start, steps)
    final_loss, final_dual = _evaluate_losses(restored, batch, update=completed)
    if not math.isfinite(final_loss) or not final_loss < initial_loss:
        raise RuntimeError("Synthetic joint learning check failed")

    deploy_action_shape = _verify_causal_deploy_policy(restored_parts, batch, target_device)
    predicted_body, target_body = _predict_body_preview(restored_parts, batch)
    horizon_labels = [f"+{t:.1f}s" for t in np.linspace(0.1, 0.6, GROUPS * BODY_POINTS)]
    preview_path = render_foresight_preview(
        output=output / "foresight_preview.png",
        loss_history=metrics,
        predicted_trajectory=predicted_body,
        target_trajectory=target_body,
        horizon_labels=horizon_labels,
        resume_update=start,
    )

    digest = hashlib.sha256((output / "predictor.safetensors").read_bytes()).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "format": "zeta0.quickstart.v2",
        "data": "synthetic_kinematic",
        "algorithm": "joint_scene_body_foresight_with_dual_response_transfer",
        "device": device,
        "source_commit": revision or None,
        "torch": torch.__version__,
        "seed": seed,
        "completed_updates": completed,
        "resumed_at": start,
        "checkpoint_sha256": digest,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "initial_dual_response": initial_dual,
        "final_dual_response": final_dual,
        "prediction_shape": [batch["observed"].raw_states.shape[0], GROUPS, SLOTS, LATENT_DIM],
        "deploy_policy_verified": True,
        "deploy_action_shape": deploy_action_shape,
        "preview_image": str(preview_path),
        "robot_commands_sent": False,
        "output": str(output),
        "metrics": metrics,
        "rollout_preview": {
            "horizon_labels": horizon_labels,
            "predicted_body": predicted_body,
            "target_body": target_body,
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("runs") / ("quickstart-" + uuid.uuid4().hex[:8])
    )
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(run(args.output, args.device, args.steps), indent=2))
