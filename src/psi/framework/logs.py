"""Offline summary of recorded runtime audit events; never replay robot commands."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize_log(filename: str, *, max_events: int = 100000) -> dict:
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    counts, client_counts, metrics, failures = Counter(), Counter(), {}, []
    total = 0
    truncated = False
    dropped = 0
    starts = ends = 0
    with Path(filename).expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if total >= max_events:
                truncated = True
                break
            try:
                event = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON at log line {line_number}; the log may be incomplete") from error
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError(f"Missing event name at log line {line_number}")
            total += 1
            name = event["event"]
            counts[name] += 1
            if name == "audit_start":
                starts += 1
            if name == "audit_end":
                ends += 1
                value = event.get("dropped", 0)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid dropped-event count at log line {line_number}")
                dropped += value
            if name == "client_event":
                event = event.get("data")
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError(f"Missing client event name at log line {line_number}")
                name = event["event"]
                client_counts[name] += 1
            if ("error" in name or "reject" in name or event.get("error")) and len(failures) < 20:
                failures.append({"line": line_number, "event": name,
                                 "session_id": event.get("session_id"), "request_seq": event.get("request_seq")})
            for key in ("inference_seconds", "anchor_to_accept_s", "queue_depth", "actual_delay_steps"):
                value = event.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    metrics.setdefault(key, []).append(float(value))
    return {"format": "zeta0.runtime_summary.v1", "ok": True, "events_read": total,
            "truncated": truncated, "audit_starts": starts, "audit_ends": ends,
            "audit_counts_balanced": starts > 0 and starts == ends and not truncated,
            "reported_dropped_events": dropped, "event_counts": dict(counts),
            "client_event_counts": dict(client_counts),
            "first_errors": failures,
            "metrics": {key: {"samples": len(values), "p50": percentile(values, .5),
                              "p95": percentile(values, .95), "max": max(values)}
                        for key, values in metrics.items()},
            "scope": "Recorded fields only; absent fields are not inferred. Counts do not establish task success.",
            "robot_commands_sent": False}
