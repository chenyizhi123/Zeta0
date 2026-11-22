"""Asynchronous real-SONIC policy client with a physical 50Hz action queue."""
import argparse
import base64
import importlib
import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

from psi.deploy.real_sonic_protocol import (
    FORMAT,
    MAX_RTC_PREFIX,
    PERIOD,
    PROTOCOL,
    STATE_SCHEMA,
    AbsoluteActionQueue,
)
from psi.real_observation_contract import validate_deployment_observation_contract

LOG = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, endpoint, timeout=2.):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def call(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        request = Request(self.endpoint+path, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def infer(self, request):
        # Convert only on this worker: camera codecs and transport cannot stall
        # the main thread's measured-state collection or 50Hz action publishing.
        import cv2

        from psi.deploy.real_sonic_io import decode_camera_image
        payload = dict(request)
        payload["images"] = {}
        for name, item in request["images"].items():
            if "encoded" in item:
                payload["images"][name] = item
            else:
                rgb = decode_camera_image(item["_source"])
                ok, png = cv2.imencode(".png", rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise ValueError("Camera PNG encoding failed")
                payload["images"][name] = dict(encoded=base64.b64encode(png).decode(), time=item["time"])
        return self.call("/infer", payload)


def check_health(health, *, state_count=21, rtc_prefix_steps=0, action_hz=50, action_horizon=30):
    expected = dict(format=FORMAT, action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
                    action_hz=action_hz, action_horizon=action_horizon, native_dim=76)
    for name, value in expected.items():
        if health.get(name) != value:
            raise ValueError(f"Incompatible policy service: {name}")
    if health.get("state_history_count", 21) != state_count:
        raise ValueError("Incompatible policy service: state_history_count")
    capacity = health.get("rtc_max_prefix_steps")
    if type(capacity) is not int or not 0 <= capacity <= MAX_RTC_PREFIX:
        raise ValueError("Incompatible policy service: rtc_max_prefix_steps")
    if type(rtc_prefix_steps) is not int or not 0 <= rtc_prefix_steps <= capacity:
        raise ValueError(f"Requested RTC prefix exceeds checkpoint support ({capacity} steps)")
    if health.get("observation_contract") is not None:
        contract = validate_deployment_observation_contract(health["observation_contract"])
        if state_count != contract["state_history_count"]:
            raise ValueError("Collector service state count differs from client")
    return capacity


def run(io, transport, *, task, rtc_prefix_steps=0, duration=0., low_watermark=15,
        starvation_grace_s=1.):
    try:
        return _run_session(io, transport, task=task, rtc_prefix_steps=rtc_prefix_steps,
            duration=duration, low_watermark=low_watermark, starvation_grace_s=starvation_grace_s)
    finally:
        # Health/config/reset failures occur before the inference worker exists.
        # They still own sockets opened by this IO adapter.
        io.close()


def _run_session(io, transport, *, task, rtc_prefix_steps, duration, low_watermark,
                 starvation_grace_s):
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duration must be finite and nonnegative")
    health = transport.call("/health")
    capacity = check_health(health,
                            state_count=getattr(getattr(io, "history", None), "state_count", 21),
                            rtc_prefix_steps=rtc_prefix_steps)
    contract = health.get("observation_contract")
    collector_mode = contract is not None
    configure = getattr(io, "configure_observation_contract", None)
    if collector_mode and not callable(configure):
        raise ValueError("Collector checkpoint requires an IO adapter with configure_observation_contract")
    if callable(configure):
        configure(contract)
    projection = contract.get("action_projection") if collector_mode else None
    if collector_mode and getattr(io, "action_projection", None) != projection:
        raise ValueError("IO adapter action projection differs from checkpoint health")
    if not math.isfinite(starvation_grace_s) or starvation_grace_s < 0:
        raise ValueError("Action-starvation grace must be finite and nonnegative")
    supports_hold = collector_mode and getattr(io, "supports_hold", False) is True and callable(getattr(io, "hold", None))
    if collector_mode:
        LOG.info("collector_snapshot_v1: feedback age=%.0fms backend_hold_supported=%s",
                 contract["state_max_age_s"]*1000, supports_hold)
    if type(low_watermark) is not int or not max(1, rtc_prefix_steps) <= low_watermark < 30:
        raise ValueError("Queue refill threshold must cover the RTC prefix and be below 30 steps")
    session = uuid.uuid4().hex
    if transport.call("/reset", {"session_id": session}).get("session_id") != session:
        raise ValueError("Server did not acknowledge session reset")
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=capacity, action_projection=projection)
    queue.reset(session)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zeta0-infer")
    inflight = None
    seq = 0
    active = False
    waiting_since = None
    hold_acknowledged = False
    started = time.monotonic()
    tick = (math.floor(started/PERIOD)+1)*PERIOD
    log_time, error_time = started, 0.
    feedback = getattr(io, "check_feedback", None) or getattr(io, "measured_state", None)

    def current_tick(scheduled, at, *, missed_current=False):
        if at-scheduled < PERIOD:
            return scheduled
        if active and not collector_mode:
            raise RuntimeError("Control loop missed a full 20ms period")
        latest = math.floor(at/PERIOD)*PERIOD
        if collector_mode:
            missed = queue.skip_to(latest, missed_current=missed_current)
            LOG.warning("Skipped %d expired action slots; resuming the absolute 50Hz schedule", missed)
        return latest

    try:
        while not duration or time.monotonic()-started < duration:
            now = time.monotonic()
            io.poll(now)
            if now < tick:
                time.sleep(min(.001, tick-now))
                continue
            now = time.monotonic()
            tick = current_tick(tick, now)
            # Retain this slot's old command before replacing only future slots.
            # Arrival occurs after tick; a new reply cannot manufacture a command
            # for a slot that has already started.
            action = queue.pop(tick)
            observed = None
            if not active:
                prepare = getattr(io, "prepare_control", None)
                if callable(prepare) and prepare(now) is not True:
                    tick += PERIOD
                    continue
            try:
                if callable(feedback):
                    feedback(tick)
                else:
                    # Existing custom adapters only expose a combined observation;
                    # retain their previous per-tick sensor checks until they add
                    # the independent measured-feedback boundary.
                    observed = io.observation(tick)
            except ValueError as exc:
                if active:
                    raise RuntimeError(f"Sensor input interrupted: {exc}") from exc
                if now-error_time > 2.:
                    LOG.info("Waiting for real sensors: %s", exc)
                    error_time = now
                tick += PERIOD
                continue
            now = time.monotonic()
            latest = current_tick(tick, now, missed_current=action is not None)
            if latest != tick:
                tick = latest
                continue
            if action is not None:
                waiting_since = None
                hold_acknowledged = False
                io.execute(action)
                active = True
            elif active:
                supports_hold = collector_mode and getattr(io, "supports_hold", False) is True and callable(getattr(io, "hold", None))
                if not supports_hold or not starvation_grace_s:
                    detail = "; backend has no acknowledged hold capability" if collector_mode else ""
                    raise RuntimeError("Action queue exhausted; stopping instead of repeating stale commands"+detail)
                if waiting_since is None:
                    waiting_since = now
                    queue.reset(session)
                    LOG.warning("Action queue exhausted; requesting balance/hand hold, waiting for fresh inference")
                elif now-waiting_since > starvation_grace_s:
                    raise RuntimeError("Action queue exhausted: acknowledged-hold grace expired")
                hold_acknowledged = io.hold() is True
                if not hold_acknowledged and getattr(io, "asynchronous_hold", False) is not True:
                    raise RuntimeError("Robot hold was not acknowledged")
            # Network-result processing follows this slot's physical send. It
            # cannot consume the slot's remaining deadline before execution.
            if inflight is not None and inflight[1].done():
                request, future = inflight
                inflight = None
                try:
                    response = future.result()
                    now = time.monotonic()
                    latest = current_tick(tick, now)
                    if latest != tick:
                        tick = latest
                        LOG.warning("Discarded reply after control-thread delay; recapturing current observations")
                        continue
                    count = queue.install(request, response, now)
                    LOG.info("request=%d server=%.3fs receive_delay=%.3fs pending=%d",
                             request["request_seq"], response.get("inference_seconds", 0.), now-request["anchor_time"], count)
                except Exception as exc:
                    LOG.warning("Discarded reply: %s", exc)
            if inflight is None and queue.depth(now) <= low_watermark:
                priming = not active or waiting_since is not None
                stable = getattr(io, "control_stable", None)
                stable = not callable(stable) or stable(now) is True
                revision, prefix = queue.snapshot(tick, 0 if priming else rtc_prefix_steps)
                # With RTC enabled, wait for a full trained reservation. Never
                # silently shrink its length because the old queue ran low.
                after_hold = waiting_since is None or (hold_acknowledged and tick > waiting_since+1e-9)
                if after_hold and (priming or stable) and (priming or rtc_prefix_steps == 0 or len(prefix) == rtc_prefix_steps):
                    if observed is None:
                        try:
                            observed = io.observation(tick)
                        except ValueError as exc:
                            # Fresh feedback was checked above. Missing vision
                            # prevents new inference, while existing future actions
                            # remain consumable until the queue expires/exhausts.
                            if now-error_time > 2.:
                                LOG.info("Waiting for inference observations: %s", exc)
                                error_time = now
                    if observed is not None:
                        request = dict(observed, format=FORMAT, session_id=session, request_seq=seq,
                                       queue_revision=revision, anchor_time=tick, task=task,
                                       action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
                                       committed_prefix=prefix)
                        inflight = (request, worker.submit(transport.infer, request))
                        seq += 1
            if now-log_time > 2.:
                LOG.info("mode=%s queue=%d inflight=%s", "running" if active else "priming", queue.depth(now), inflight is not None)
                log_time = now
            tick += PERIOD
    finally:
        queue.reset(uuid.uuid4().hex)
        try:
            io.stop()
        finally:
            worker.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="http://GPU_SERVER:8014")
    parser.add_argument("--task", default="Place the doll on the table.")
    parser.add_argument("--config", required=True, help="Robot endpoint JSON")
    parser.add_argument("--execute", action="store_true", help="Publish to existing SONIC and BrainCo controls")
    parser.add_argument("--current-state-only", action="store_true",
                        help="Send one current state; requires a matching newly trained policy.")
    parser.add_argument("--duration", type=float, default=0.)
    parser.add_argument("--rtc-prefix-steps", type=int, choices=range(MAX_RTC_PREFIX + 1), default=0)
    parser.add_argument("--queue-low-watermark", type=int, default=20,
                        help="Refill with this many future steps remaining; must cover the RTC prefix.")
    parser.add_argument("--starvation-grace-s", type=float, default=1.,
                        help="Collector-mode recovery window; only available with acknowledged backend hold.")
    parser.add_argument("--io-adapter", help="Optional existing recording interface: module:factory(config)")
    parser.add_argument("--velocity-fallback", action="store_true",
                        help="Opt in to native zero-velocity recovery; requires rebuilt controller and BrainCo ACK support.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config) as stream:
        config = json.load(stream)
    config["execute"] = args.execute
    if args.velocity_fallback:
        config["velocity_fallback"] = True
    if args.current_state_only:
        config["state_history_count"] = 1
    if args.io_adapter:
        name, function = args.io_adapter.split(":", 1)
        io = getattr(importlib.import_module(name), function)(config)
    else:
        from psi.deploy.real_sonic_io import SonicIOConfig, SonicZmqIO
        io = SonicZmqIO(SonicIOConfig(**config))
    try:
        run(io, HttpClient(args.server), task=args.task, rtc_prefix_steps=args.rtc_prefix_steps,
            duration=args.duration, low_watermark=args.queue_low_watermark,
            starvation_grace_s=args.starvation_grace_s)
    except KeyboardInterrupt:
        LOG.info("Stopped by operator")
    except Exception:
        LOG.exception("Client stopped")
        raise


if __name__ == "__main__":
    main()
