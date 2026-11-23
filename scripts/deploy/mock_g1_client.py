#!/usr/bin/env python3
"""Hardware-free Sim-in-the-Loop G1 dry-run client for RTC & wire-protocol validation.

Exercises the exact ``real_sonic_protocol`` request validation, ``MeasuredHistory``,
``AbsoluteActionQueue`` (with RTC committed prefix handoff), and ``RuntimeAudit``
JSONL output without requiring a physical Unitree G1 robot or GPU.

Examples:
    # Self-contained offline dry-run (in-process synthetic server + trace output):
    PYTHONPATH=src python scripts/deploy/mock_g1_client.py --in-process --cycles 12 --output runs/mock_g1_trace.jsonl
    zeta logs summarize runs/mock_g1_trace.jsonl
    zeta visualize log runs/mock_g1_trace.jsonl --output runs/mock_g1_dashboard.png

    # Against a running real_sonic_server HTTP endpoint:
    PYTHONPATH=src python scripts/deploy/mock_g1_client.py --server http://127.0.0.1:8014 --cycles 10 --output runs/live_mock.jsonl
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from psi.deploy.real_sonic_protocol import (
    ACTION_DIM,
    CAMERAS,
    FORMAT,
    HORIZON,
    PERIOD,
    PROTOCOL,
    STATE_DIM,
    STATE_SCHEMA,
    AbsoluteActionQueue,
    MeasuredHistory,
    validate_request,
)


def _dummy_jpeg_b64() -> str:
    img = Image.new("RGB", (640, 480), color=(32, 48, 64))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _in_process_infer(request: dict, *, rtc_max_prefix: int, simulated_latency_ms: float) -> dict:
    validate_request(request, state_count=1, rtc_max_prefix_steps=rtc_max_prefix)
    prefix = np.asarray(request.get("committed_prefix", []), dtype=np.float32)
    actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    t = np.linspace(0.0, 1.0, HORIZON, dtype=np.float32)[:, None]
    actions[:, :29] = 0.05 * np.sin(2.0 * np.pi * t + request["request_seq"] * 0.2)
    if len(prefix):
        actions[: len(prefix)] = prefix
    if simulated_latency_ms > 0:
        time.sleep(min(simulated_latency_ms, 50.0) / 1000.0)
    return {
        "format": request["format"],
        "session_id": request["session_id"],
        "request_seq": request["request_seq"],
        "queue_revision": request["queue_revision"],
        "anchor_time": request["anchor_time"],
        "action_protocol_id": request["action_protocol_id"],
        "actions": actions.tolist(),
    }


def _http_post(url: str, payload: dict, timeout: float = 5.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_mock_session(
    *,
    output: Path,
    cycles: int = 10,
    rtc_prefix_steps: int = 6,
    simulated_latency_ms: float = 10.0,
    server: str | None = None,
    in_process: bool = False,
) -> dict:
    if not in_process and not server:
        raise ValueError("Specify --in-process or --server http://HOST:PORT")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing trace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    session_id = "mock-g1-sim-in-the-loop"
    history = MeasuredHistory(state_count=1)
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=rtc_prefix_steps)
    queue.reset(session_id)
    if server:
        _http_post(f"{server.rstrip('/')}/reset", {"session_id": session_id})

    encoded_img = _dummy_jpeg_b64()
    events: list[dict] = [{"event": "audit_start", "session_id": session_id, "mode": "mock_g1"}]
    accepted = 0
    anchor = 1.0

    for seq in range(cycles):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec[:29] = 0.05 * np.sin(seq * 0.2)
        history.append(anchor, state_vec)
        prefix_count = rtc_prefix_steps if seq > 0 else 0
        revision, prefix = queue.snapshot(anchor, prefix_count)
        request = {
            "format": FORMAT,
            "session_id": session_id,
            "request_seq": seq,
            "queue_revision": revision,
            "anchor_time": anchor,
            "action_protocol_id": PROTOCOL,
            "state_schema_id": STATE_SCHEMA,
            "state": history.aligned(anchor),
            "images": {cam: {"time": 0.0, "encoded": encoded_img} for cam in CAMERAS},
            "committed_prefix": prefix,
        }
        started = time.perf_counter()
        if in_process:
            response = _in_process_infer(
                request,
                rtc_max_prefix=rtc_prefix_steps,
                simulated_latency_ms=simulated_latency_ms,
            )
            wall_s = max(time.perf_counter() - started, simulated_latency_ms / 1000.0)
        else:
            response = _http_post(f"{server.rstrip('/')}/infer", request)
            wall_s = time.perf_counter() - started
        now_sim = anchor + min(wall_s, max(rtc_prefix_steps - 1, 1) * PERIOD * 0.5)
        installed = queue.install(request, response, now_sim)
        depth = queue.depth(now_sim)
        accepted += 1
        events.append({
            "event": "accepted",
            "session_id": session_id,
            "request_seq": seq,
            "inference_seconds": round(wall_s, 6),
            "anchor_to_accept_s": round(now_sim - anchor, 6),
            "queue_depth": int(depth),
            "installed_steps": int(installed),
            "committed_prefix_steps": len(prefix),
        })
        # Advance simulated robot clock by 6 control ticks (120 ms at 50 Hz).
        for step in range(1, 7):
            queue.pop(anchor + step * PERIOD)
        anchor = round(anchor + 6 * PERIOD, 6)

    events.append({"event": "audit_end", "session_id": session_id, "dropped": 0})
    output.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")
    return {
        "format": "zeta0.mock_g1_session.v1",
        "ok": True,
        "mode": "in_process" if in_process else "http",
        "cycles": cycles,
        "accepted": accepted,
        "trace": str(output),
        "robot_commands_sent": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-process", action="store_true", help="Use an offline synthetic in-process policy stub")
    parser.add_argument("--server", default=None, help="Running real_sonic_server URL (e.g. http://127.0.0.1:8014)")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--rtc-prefix-steps", type=int, default=6)
    parser.add_argument("--simulated-latency-ms", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True, help="New RuntimeAudit JSONL output path")
    args = parser.parse_args()
    report = run_mock_session(
        output=args.output,
        cycles=args.cycles,
        rtc_prefix_steps=args.rtc_prefix_steps,
        simulated_latency_ms=args.simulated_latency_ms,
        server=args.server,
        in_process=args.in_process,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
