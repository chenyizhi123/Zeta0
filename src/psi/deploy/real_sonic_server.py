"""HTTP service for the separate immutable real-SONIC research weight package."""
import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from psi.deploy.real_sonic_protocol import FORMAT, LEGACY_RTC_PREFIX, MAX_RTC_PREFIX, PROTOCOL, STATE_SCHEMA, validate_request, validate_action_grid

LOG = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, infer, manifest, seed=0, *, rtc_max_prefix_steps=LEGACY_RTC_PREFIX, observation_contract=None):
        from psi.real_observation_contract import validate_deployment_observation_contract
        if type(rtc_max_prefix_steps) is not int or not 0 <= rtc_max_prefix_steps <= MAX_RTC_PREFIX:
            raise ValueError("Unsupported real-SONIC RTC prefix capacity")
        self.rtc_max_prefix_steps = rtc_max_prefix_steps
        self.infer = infer
        self.manifest = manifest
        self.seed = seed
        self.lock = threading.Lock()
        self.session = None
        self.last_seq = -1
        self.state_count = manifest.get("temporal_abi", {}).get("state_history_count", 21)
        temporal = manifest.get("temporal_abi", {})
        self.action_hz, self.action_horizon = validate_action_grid(
            temporal.get("action_hz", 50), temporal.get("action_horizon", 30))
        if type(self.state_count) is not int or self.state_count not in (1, 21):
            raise ValueError("Unsupported real-SONIC state history count")
        if manifest.get("rtc_max_prefix_steps", rtc_max_prefix_steps) != rtc_max_prefix_steps:
            raise ValueError("RTC capacity must come from the source package manifest")
        bound_contract = manifest.get("observation_contract")
        if observation_contract is not None and observation_contract != bound_contract:
            raise ValueError("Observation contract must come from the source package manifest")
        self.observation_contract = (
            None if bound_contract is None else validate_deployment_observation_contract(bound_contract)
        )
        if self.observation_contract is not None and self.observation_contract["state_history_count"] != self.state_count:
            raise ValueError("Observation contract state count differs from source temporal ABI")

    def health(self):
        return dict(format=FORMAT, action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
                    action_hz=self.action_hz, action_horizon=self.action_horizon, native_dim=76, rtc_max_prefix_steps=self.rtc_max_prefix_steps,
                    state_history_count=self.state_count, observation_contract=self.observation_contract,
                    task=self.manifest["task"], source_completed_updates=self.manifest["source_completed_updates"],
                    weights_sha256=self.manifest["weights_sha256"],
                    validated_production_deploy_bundle=False)

    def reset(self, data):
        session = data.get("session_id")
        if not isinstance(session, str) or not session or len(session) > 128:
            raise ValueError("session_id must be a nonempty string of at most 128 characters")
        with self.lock:
            self.session, self.last_seq = session, -1
        return dict(format=FORMAT, session_id=session)

    def predict(self, request):
        validate_request(request, state_count=self.state_count, rtc_max_prefix_steps=self.rtc_max_prefix_steps,
                         observation_contract=self.observation_contract)
        # A busy GPU must reject a second stream instead of silently queuing old observations.
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Policy is busy; only one inference stream is supported")
        try:
            if request["session_id"] != self.session:
                raise ValueError("Session changed; reset the service before inference")
            if request["request_seq"] <= self.last_seq:
                raise ValueError("Duplicate or regressing request sequence")
            self.last_seq = request["request_seq"]
            started = time.perf_counter()
            actions = np.asarray(self.infer(request, seed=self.seed + request["request_seq"]), dtype=np.float32)
            if actions.shape != (self.action_horizon, 76) or not np.isfinite(actions).all():
                raise ValueError(f"Policy must return finite native actions with shape [{self.action_horizon},76]")
            response = {key: request[key] for key in ("format", "session_id", "request_seq",
                        "queue_revision", "anchor_time", "action_protocol_id")}
            response.update(actions=actions.tolist(), inference_seconds=time.perf_counter()-started,
                            action_offsets_s=[(i+1)/self.action_hz for i in range(self.action_horizon)])
            LOG.info("request=%s inference=%.3fs", request["request_seq"], response["inference_seconds"])
            return response
        finally:
            self.lock.release()


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, service.health()) if self.path == "/health" else self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("Request size must be 1 byte..8MiB")
                self.connection.settimeout(5)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("JSON body must be an object")
                if self.path == "/reset":
                    result = service.reset(data)
                elif self.path == "/infer":
                    result = service.predict(data)
                else:
                    self.reply(404, {"error": "Unknown endpoint"})
                    return
                self.reply(200, result)
            except BlockingIOError as exc:
                self.reply(409, {"error": str(exc)})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception:
                LOG.exception("Inference failed")
                self.reply(500, {"error": "Inference failed; check server log"})

        def log_message(self, format, *args):
            LOG.debug(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
    runtime = ResearchInferenceRuntime(args.weights_dir)
    policy, processor = runtime.build_policy(args.qwen_model, device=args.device)
    service = InferenceService(lambda request, seed: runtime.infer_request(policy, processor, request, seed=seed),
                               runtime.read_json("inference_manifest.json"), seed=args.seed,
                               rtc_max_prefix_steps=runtime.rtc_max_prefix_steps,
                               observation_contract=runtime.observation_contract)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    server.daemon_threads = True
    LOG.info("Ready at %s:%s; recorded q29+hand12 / motion64+hand12", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
