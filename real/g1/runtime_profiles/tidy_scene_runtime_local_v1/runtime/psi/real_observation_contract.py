"""The recorded G1/BrainCo observation composer contract, without hardware imports.

Receipts share the client workstation monotonic clock.  The ego camera receipt
anchors sensor pairing; the later decision anchors predicted action slots.
Neither clock is a claim of simultaneous physical exposure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

COLLECTOR_SNAPSHOT_MODE = "collector_snapshot_v1"
CAMERA_KEYS = tuple("observation.images." + name for name in ("ego_view", "left_wrist", "right_wrist"))
TIME_SEMANTICS = "client_workstation_receive_time"
ACTION_PROJECTION_FORMAT = "oraclevla.real_sonic.recorded_action_projection.v1"
_EPS = 1e-9
_LIMITS = ("camera_max_skew_ms", "camera_wait_ms", "max_age_ms", "body_max_offset_ms", "hand_max_offset_ms")


def build_recorded_action_projection(lower: float, upper: float, source_sha256: str) -> dict:
    """Preserve recorded targets and bound inference by that same training range."""
    return validate_action_projection(dict(format=ACTION_PROJECTION_FORMAT,
        motion_token_min=float(lower), motion_token_max=float(upper), motion_token_step=1/16,
        hand_min=0., hand_max=1., range_source="recorded_train_action_extrema", source_sha256=source_sha256))


def validate_action_projection(projection: Mapping) -> dict:
    required = {"format", "motion_token_min", "motion_token_max", "motion_token_step",
                "hand_min", "hand_max", "range_source", "source_sha256"}
    if not isinstance(projection, Mapping) or set(projection) != required:
        raise ValueError("Recorded action projection requires its complete source-bound schema")
    if (projection["format"] != ACTION_PROJECTION_FORMAT
            or projection["range_source"] != "recorded_train_action_extrema"):
        raise ValueError("Unsupported recorded action projection semantics")
    identity = projection["source_sha256"]
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Recorded action projection requires a source SHA256")
    result = dict(projection)
    for key in ("motion_token_min", "motion_token_max", "motion_token_step", "hand_min", "hand_max"):
        value = projection[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Action projection bounds must be finite numbers")
        result[key] = float(value)
    lower, upper = result["motion_token_min"], result["motion_token_max"]
    if (result["motion_token_step"] != 1/16 or not lower <= 0 <= upper
            or any(abs(value*16-round(value*16)) > _EPS for value in (lower, upper))
            or (result["hand_min"], result["hand_max"]) != (0., 1.)):
        raise ValueError("Recorded action projection must preserve its 1/16 token grid and native hand space")
    return result


def validate_observation_contract(contract: Mapping) -> dict:
    """Validate pairing v1, bounded latest v2, or unbounded recorded latest v3."""
    latest = isinstance(contract, Mapping) and contract.get("version") in (2, 3)
    unbounded = isinstance(contract, Mapping) and contract.get("version") == 3
    limits = ("max_age_ms",) if latest else _LIMITS
    required = {"version", *limits, "history_size", "image_shape"}
    if latest:
        required.update({"selection", "cameras", "timestamp_kind"})
    if unbounded:
        required.add("age_policy")
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError("Collector observation contract requires its complete versioned schema")
    if type(contract["version"]) is not int or contract["version"] not in (1, 2, 3):
        raise ValueError("Unsupported collector observation contract version")
    result = {"version": contract["version"]}
    if latest:
        if (contract["selection"] != "latest_available_v1"
                or contract["cameras"] != [key.removeprefix("observation.images.") for key in CAMERA_KEYS]
                or contract["timestamp_kind"] != "workstation_source_receive_time"):
            raise ValueError("Unsupported latest-available collector semantics")
        result.update(selection=contract["selection"], cameras=list(contract["cameras"]),
                      timestamp_kind=contract["timestamp_kind"])
    for key in limits:
        value = contract[key]
        if unbounded:
            if value is not None or contract["age_policy"] != "record_latest_available":
                raise ValueError("Collector v3 requires null max_age_ms and record_latest_available")
            result.update(max_age_ms=None, age_policy="record_latest_available")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Collector {key} must be finite and positive")
        result[key] = float(value)
    size = contract["history_size"]
    if type(size) is not int or size < 1:
        raise ValueError("Collector history_size must be a positive integer")
    shape = contract["image_shape"]
    if not isinstance(shape, (list, tuple)) or list(shape) != [480, 640, 3]:
        raise ValueError("Collector images require RGB [480,640,3]")
    result.update(history_size=size, image_shape=[480, 640, 3])
    return result


def snapshot_age_limit_s(collector: Mapping, explicit_limit_s: float | None = None) -> float:
    """Keep a v3 training/deployment cutoff separate from its unbounded producer."""
    collector = validate_observation_contract(collector)
    if collector["version"] != 3:
        if explicit_limit_s is not None:
            raise ValueError("An explicit snapshot age limit is only supported for collector v3")
        return collector["max_age_ms"] / 1000
    if (isinstance(explicit_limit_s, bool) or not isinstance(explicit_limit_s, (int, float))
            or not math.isfinite(explicit_limit_s) or explicit_limit_s <= 0):
        raise ValueError("Collector v3 requires an explicit finite positive snapshot age limit")
    return float(explicit_limit_s)


def make_deployment_observation_contract(data: Mapping) -> dict | None:
    """Derive wire limits from checksum-bound training data configuration."""
    mode = data.get("real_observation_mode")
    if mode is None:
        if (data.get("real_observation_contract") is not None or data.get("real_action_projection") is not None
                or data.get("real_snapshot_max_age_s") is not None):
            raise ValueError("Collector limits require an explicit observation mode")
        return None
    if mode != COLLECTOR_SNAPSHOT_MODE:
        raise ValueError("Unsupported real observation mode")
    collector = validate_observation_contract(data.get("real_observation_contract"))
    age = snapshot_age_limit_s(collector, data.get("real_snapshot_max_age_s"))
    if type(data.get("state_history_count")) is not int or data["state_history_count"] != 1:
        raise ValueError("Collector snapshots require one measured state")
    for key in ("state_max_age_s", "visual_max_age_s"):
        if not math.isclose(float(data.get(key, -1)), age, rel_tol=0, abs_tol=_EPS):
            raise ValueError(f"{key} must match the source collector max_age_ms")
    result = dict(mode=mode, collector=collector, state_history_count=1,
                  camera_keys=list(CAMERA_KEYS), state_max_age_s=age, visual_max_age_s=age,
                  time_semantics=TIME_SEMANTICS)
    if collector["version"] == 3:
        result["real_snapshot_max_age_s"] = age
    if data.get("real_action_projection") is not None:
        result["action_projection"] = validate_action_projection(data["real_action_projection"])
    return result


def validate_deployment_observation_contract(contract: Mapping) -> dict:
    """Reject a partial, guessed or differently ordered advertised contract."""
    required = {"mode", "collector", "state_history_count", "camera_keys", "state_max_age_s",
                "visual_max_age_s", "time_semantics"}
    optional = {"action_projection", "real_snapshot_max_age_s"}
    if not isinstance(contract, Mapping) or not required <= set(contract) or set(contract) - required - optional:
        raise ValueError("Deployment observation contract requires complete source-bound fields")
    if "action_projection" in contract:
        validate_action_projection(contract["action_projection"])
    if (contract["collector"].get("version") == 3) != ("real_snapshot_max_age_s" in contract):
        raise ValueError("Only collector v3 requires a separately declared snapshot age limit")
    if contract["camera_keys"] != list(CAMERA_KEYS) or contract["time_semantics"] != TIME_SEMANTICS:
        raise ValueError("Deployment camera order or receipt-clock semantics differ")
    result = make_deployment_observation_contract(dict(
        real_observation_mode=contract["mode"], real_observation_contract=contract["collector"],
        state_history_count=contract["state_history_count"], state_max_age_s=contract["state_max_age_s"],
        visual_max_age_s=contract["visual_max_age_s"], real_action_projection=contract.get("action_projection"),
        real_snapshot_max_age_s=contract.get("real_snapshot_max_age_s")))
    if result is None:
        raise ValueError("Deployment observation contract cannot advertise a legacy mode")
    return result


def snapshot_timing_violations(*, decision_time: float, camera_anchor_time: float,
                               camera_times: Mapping, body_time: float,
                               hand_times: Sequence, contract: Mapping,
                               snapshot_max_age_s: float | None = None) -> list[dict]:
    """Audit a whole snapshot against the original pairing and availability rules.

    Times may be absolute or relative, provided every value shares one clock.
    Return explicit exclusion evidence; malformed schemas raise rather than filter.
    """
    limits = validate_observation_contract(contract)
    if not isinstance(camera_times, Mapping) or not all(isinstance(key, str) for key in camera_times):
        raise ValueError("Snapshot camera times must be a mapping")
    cameras = {key if key.startswith("observation.images.") else "observation.images." + key: float(value)
               for key, value in camera_times.items()}
    if set(cameras) != set(CAMERA_KEYS) or len(camera_times) != len(CAMERA_KEYS):
        raise ValueError("Snapshot requires the three recorded camera receipt times")
    if len(hand_times) != 2:
        raise ValueError("Snapshot requires two measured hand channel times")
    decision, anchor, body = float(decision_time), float(camera_anchor_time), float(body_time)
    hands = [float(time) for time in hand_times]
    sources = {**cameras, "body": body, "left_hand": hands[0], "right_hand": hands[1]}
    if not all(math.isfinite(time) for time in (decision, anchor, *sources.values())):
        return [dict(code="snapshot_nonfinite_time", evidence={})]
    violations = []
    def add(code, **evidence):
        violations.append(dict(code=code, evidence=evidence))
    future = [key for key, time in sources.items() if time > decision + _EPS]
    if future or anchor > decision + _EPS:
        add("snapshot_future_input", components=future, camera_anchor_future=anchor > decision + _EPS)
    ages = {key: (decision-time) * 1000 for key, time in sources.items()}
    max_age_ms = limits["max_age_ms"]
    if snapshot_max_age_s is not None:
        max_age_ms = snapshot_age_limit_s(limits, snapshot_max_age_s) * 1000
    stale = {key: age for key, age in ages.items()
             if max_age_ms is not None and age > max_age_ms + _EPS * 1000}
    if stale:
        add("snapshot_stale", ages_ms=stale, max_age_ms=max_age_ms)
    if limits["version"] in (2, 3):
        # The producer selects the latest available sources independently.
        # Pairing skew/body offsets are v1 rules, absent from the v2 contract.
        return violations
    if abs(anchor-cameras[CAMERA_KEYS[0]]) > _EPS:
        add("snapshot_anchor_mismatch", camera_anchor_time=anchor, ego_time=cameras[CAMERA_KEYS[0]])
    skew = (max(cameras.values())-min(cameras.values())) * 1000
    if skew > limits["camera_max_skew_ms"] + _EPS * 1000:
        add("snapshot_camera_skew", skew_ms=skew, max_skew_ms=limits["camera_max_skew_ms"])
    offset = (body-anchor) * 1000
    if abs(offset) > limits["body_max_offset_ms"] + _EPS * 1000:
        add("snapshot_body_offset", offset_ms=offset, max_offset_ms=limits["body_max_offset_ms"])
    offsets = [(time-anchor) * 1000 for time in hands]
    if any(abs(offset) > limits["hand_max_offset_ms"] + _EPS * 1000 for offset in offsets):
        add("snapshot_hand_offset", offsets_ms=offsets, max_offset_ms=limits["hand_max_offset_ms"])
    return violations


def validate_snapshot_timing(**kwargs) -> None:
    violations = snapshot_timing_violations(**kwargs)
    if violations:
        raise ValueError("Invalid collector snapshot: " + ", ".join(item["code"] for item in violations))
