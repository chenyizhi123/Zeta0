"""Offline visualization studio for Zeta-0 foresight rollouts and G1 RuntimeAudit logs.

Uses only stdlib + Pillow/NumPy by default so reports render in headless CPU
environments. Optionally streams 3D URDF trajectories to Rerun when ``rerun-sdk``
is installed and ``use_rerun=True`` is requested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _draw_axes(draw, box, *, title: str, subtitle: str = "") -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.text((x0 + 12, y0 - 26), title, fill=(15, 23, 42))
    if subtitle:
        draw.text((x0 + 12, y1 + 8), subtitle, fill=(71, 85, 105))
    for frac in (0.25, 0.5, 0.75):
        gy = int(y0 + frac * (y1 - y0))
        draw.line([(x0 + 1, gy), (x1 - 1, gy)], fill=(226, 232, 240), width=1)


def _plot_series(draw, box, series: list[tuple[list[float], tuple[int, int, int], str]], *, y_min=None, y_max=None):
    x0, y0, x1, y1 = box
    pad = 16
    px0, py0, px1, py1 = x0 + pad, y0 + pad + 26, x1 - pad, y1 - pad
    all_vals = [v for vals, _, _ in series for v in vals if math.isfinite(v)]
    if not all_vals:
        draw.text((px0 + 10, (py0 + py1) // 2), "No finite samples", fill=(100, 116, 139))
        return
    lo = min(all_vals) if y_min is None else y_min
    hi = max(all_vals) if y_max is None else y_max
    if hi - lo < 1e-8:
        hi = lo + 1.0
    legend_x = px0
    legend_y = y0 + 10
    for vals, color, label in series:
        if len(vals) >= 2:
            pts = []
            for idx, v in enumerate(vals):
                if not math.isfinite(v):
                    continue
                fx = idx / max(len(vals) - 1, 1)
                fy = (v - lo) / (hi - lo)
                pts.append((px0 + fx * (px1 - px0), py1 - fy * (py1 - py0)))
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=3)
            for pt in pts:
                draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill=color)
        elif len(vals) == 1 and math.isfinite(vals[0]):
            fy = (vals[0] - lo) / (hi - lo)
            py = py1 - fy * (py1 - py0)
            draw.line([(px0, py), (px1, py)], fill=color, width=3)
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color)
        draw.text((legend_x + 16, legend_y), label, fill=(30, 41, 59))
        legend_x += 16 + max(len(label) * 7, 64)
    draw.text((x0 + 6, py0 - 2), f"{hi:.3g}", fill=(100, 116, 139))
    draw.text((x0 + 6, py1 - 10), f"{lo:.3g}", fill=(100, 116, 139))


def render_foresight_preview(
    *,
    output: str | Path,
    loss_history: list[dict[str, float]],
    predicted_trajectory: list[list[float]],
    target_trajectory: list[list[float]],
    horizon_labels: list[str] | None = None,
    resume_update: int | None = None,
    title: str = "Zeta-0 Scene & Body Foresight Preview",
) -> Path:
    """Render a clean 2-panel PNG showing training dynamics + predicted vs target body rollout."""
    from PIL import Image, ImageDraw

    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1120, 460), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1120, 52], fill=(15, 23, 42))
    draw.text((24, 18), title, fill=(248, 250, 252))

    left_box = (36, 98, 536, 396)
    right_box = (584, 98, 1084, 396)
    _draw_axes(
        draw,
        left_box,
        title="Training Objectives (Joint + Dual-Response Transfer)",
        subtitle=f"Checkpoint save & exact resume at update {resume_update}" if resume_update else "",
    )
    mse_vals = [float(item.get("total_loss", item.get("mse", 0.0))) for item in loss_history]
    body_vals = [float(item["prior_body_mse"]) for item in loss_history if "prior_body_mse" in item]
    left_series = [(mse_vals, (14, 116, 144), "joint_objective")]
    if body_vals and len(body_vals) == len(mse_vals):
        left_series.append((body_vals, (217, 119, 6), "prior_body_mse"))
    _plot_series(draw, left_box, left_series)
    if resume_update and len(mse_vals) >= 2:
        rx = left_box[0] + 16 + ((resume_update - 1) / max(len(mse_vals) - 1, 1)) * (left_box[2] - left_box[0] - 32)
        draw.line([(rx, left_box[1] + 20), (rx, left_box[3] - 16)], fill=(148, 163, 184), width=2)

    labels = horizon_labels or [f"t+{i+1}" for i in range(len(target_trajectory))]
    _draw_axes(
        draw,
        right_box,
        title="Body-Trajectory Readout D(z_hat_prior) vs Recorded Future",
        subtitle="Horizons: " + " | ".join(labels[:6]),
    )
    pred_j0 = [float(step[0]) for step in predicted_trajectory] if predicted_trajectory else []
    true_j0 = [float(step[0]) for step in target_trajectory] if target_trajectory else []
    right_series = [
        (true_j0, (15, 118, 110), "measured_j0"),
        (pred_j0, (225, 29, 72), "predicted_j0"),
    ]
    if predicted_trajectory and len(predicted_trajectory[0]) > 1:
        right_series.append(([float(step[1]) for step in target_trajectory], (59, 130, 246), "measured_j1"))
        right_series.append(([float(step[1]) for step in predicted_trajectory], (147, 51, 234), "predicted_j1"))
    _plot_series(draw, right_box, right_series)

    img.save(out_path)
    return out_path


def visualize_log(log_path: str | Path, *, output: str | Path, max_events: int = 100_000) -> dict[str, Any]:
    """Render an offline RTC latency & queue-depth dashboard from a G1 RuntimeAudit JSONL."""
    from PIL import Image, ImageDraw

    from psi.framework.logs import summarize_log

    summary = summarize_log(str(log_path), max_events=max_events)
    latencies: list[float] = []
    queues: list[float] = []
    with Path(log_path).open("r", encoding="utf-8") as stream:
        for idx, raw in enumerate(stream, 1):
            if idx > max_events:
                break
            line = raw.strip()
            if not line:
                continue
            event = json.loads(line)
            nested = event.get("data") if isinstance(event.get("data"), dict) else {}
            lat = event.get("inference_seconds", nested.get("inference_seconds"))
            q = event.get("queue_depth", nested.get("queue_depth"))
            if isinstance(lat, (int, float)) and not isinstance(lat, bool):
                latencies.append(float(lat) * 1000.0)
            if isinstance(q, (int, float)) and not isinstance(q, bool):
                queues.append(float(q))

    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1120, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1120, 54], fill=(15, 23, 42))
    draw.text(
        (24, 18),
        f"Zeta-0 RuntimeAudit Telemetry · {Path(log_path).name} ({summary['events_read']} events)",
        fill=(248, 250, 252),
    )
    lat_stats = summary.get("metrics", {}).get("inference_seconds", {})
    lat_sub = (
        f"p50={lat_stats['p50']*1000:.1f}ms  p95={lat_stats['p95']*1000:.1f}ms  max={lat_stats['max']*1000:.1f}ms"
        if lat_stats
        else "No inference_seconds samples recorded"
    )
    _draw_axes(draw, (36, 98, 536, 396), title="Closed-Loop Inference Latency (ms)", subtitle=lat_sub)
    _plot_series(draw, (36, 98, 536, 396), [(latencies, (14, 116, 144), "latency_ms")])

    q_stats = summary.get("metrics", {}).get("queue_depth", {})
    q_sub = (
        f"p50={q_stats['p50']:.1f}  p95={q_stats['p95']:.1f}  errors={len(summary.get('first_errors', []))}"
        if q_stats
        else f"errors={len(summary.get('first_errors', []))}  dropped={summary.get('reported_dropped_events', 0)}"
    )
    _draw_axes(draw, (584, 98, 1084, 396), title="RTC Action Queue Depth", subtitle=q_sub)
    _plot_series(draw, (584, 98, 1084, 396), [(queues, (15, 118, 110), "queue_depth")])

    img.save(out_path)
    return {
        "format": "zeta0.visualize_log.v1",
        "ok": True,
        "input": str(Path(log_path).resolve()),
        "output": str(out_path),
        "events_read": summary["events_read"],
        "metrics": summary["metrics"],
        "robot_commands_sent": False,
    }


def visualize_rollout(
    summary_path: str | Path,
    *,
    output: str | Path,
    urdf: str | Path | None = None,
    use_rerun: bool = False,
) -> dict[str, Any]:
    """Visualize predicted vs recorded future body trajectories from a quickstart or rollout JSON."""
    source = Path(summary_path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rollout = payload.get("rollout_preview") or payload
    predicted = rollout.get("predicted_body")
    target = rollout.get("target_body")
    if not predicted or not target:
        raise ValueError(
            "Input JSON must include 'rollout_preview' (or top-level) 'predicted_body' and 'target_body' arrays"
        )
    out_path = render_foresight_preview(
        output=output,
        loss_history=payload.get("metrics", []),
        predicted_trajectory=predicted,
        target_trajectory=target,
        horizon_labels=rollout.get("horizon_labels"),
        resume_update=payload.get("resumed_at"),
        title=f"Zeta-0 Foresight Rollout · {source.name}",
    )

    kinematics_points = None
    if urdf is not None and len(predicted[0]) >= 14:
        import numpy as np

        from psi.visualization.kinematics import TorsoArmKinematics

        fk = TorsoArmKinematics(urdf)
        kinematics_points = {
            "predicted_arm_xyz_shape": list(fk.positions(np.asarray(predicted)[..., :14]).shape),
            "target_arm_xyz_shape": list(fk.positions(np.asarray(target)[..., :14]).shape),
        }

    rerun_status = "disabled"
    if use_rerun:
        try:
            import rerun as rr
        except ImportError as exc:
            raise ImportError(
                "rerun-sdk is required for --rerun 3D streaming (install with: pip install rerun-sdk)"
            ) from exc
        rr.init("zeta0_foresight_studio", spawn=False)
        for step_idx, (pred_step, true_step) in enumerate(zip(predicted, target)):
            rr.set_time_sequence("horizon_step", step_idx)
            for j, (pv, tv) in enumerate(zip(pred_step, true_step)):
                rr.log(f"body/joint_{j}/predicted", rr.Scalar(float(pv)))
                rr.log(f"body/joint_{j}/measured", rr.Scalar(float(tv)))
        rerun_status = "logged"

    return {
        "format": "zeta0.visualize_rollout.v1",
        "ok": True,
        "input": str(source),
        "output": str(out_path),
        "horizon_steps": len(predicted),
        "kinematics": kinematics_points,
        "rerun": rerun_status,
        "robot_commands_sent": False,
    }
