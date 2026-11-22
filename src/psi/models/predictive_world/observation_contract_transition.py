"""Explicit source-bound initialization for recorded real-robot snapshots."""

from dataclasses import asdict, is_dataclass

from psi.data.predictive_world.cache import canonical_sha256
from psi.real_observation_contract import make_deployment_observation_contract, validate_observation_contract, snapshot_age_limit_s

REAL_TEMPORAL_FIELDS = frozenset({
    "real_observation_mode", "real_observation_contract", "real_action_label_mode", "real_action_projection",
    "real_snapshot_max_age_s",
})


def _source_future_limits(payload):
    data = payload["data"]
    return {
        "future_measurement_max_age_s": (
            data.get("future_measurement_max_age_s")
            if data.get("future_measurement_max_age_s") is not None else data.get("state_max_age_s", .04)
        ),
        "future_visual_max_age_s": (
            data.get("future_visual_max_age_s")
            if data.get("future_visual_max_age_s") is not None else data.get("visual_max_age_s", .08)
        ),
        "action_max_age_s": data.get("action_max_age_s", .04),
    }


def observation_contract_transition(source, source_payload, target_temporal):
    """Bind one compact checkpoint ABI transition, without importing models.

    The separate current-state migration owns position-tensor resizing. This
    record permits that declared 21 -> 1 change alongside snapshot semantics;
    every other pre-existing temporal field must remain identical.
    """
    metadata = source["metadata"]
    if metadata["resolved_config_sha256"] != canonical_sha256(source_payload):
        raise ValueError("observation transition source config checksum mismatch")
    previous = dict(metadata["temporal_abi"])
    target = dict(target_temporal)
    if previous.get("real_observation_mode") is not None:
        raise ValueError("source already uses real snapshot semantics; use exact resume")
    if target.get("real_action_label_mode") != "held_command_v1":
        raise ValueError("observation transition requires the explicit held-command label contract")
    collector = validate_observation_contract(target.get("real_observation_contract"))
    age = snapshot_age_limit_s(collector, target.get("real_snapshot_max_age_s"))
    deployment = make_deployment_observation_contract({
        **{key: target.get(key) for key in REAL_TEMPORAL_FIELDS},
        "state_history_count": target.get("state_history_count"),
        "state_max_age_s": age, "visual_max_age_s": age,
    })
    if deployment is None:
        raise ValueError("observation transition requires recorded collector snapshots")
    unchanged = {key: value for key, value in target.items() if key not in REAL_TEMPORAL_FIELDS}
    source_count, target_count = previous.get("state_history_count"), unchanged.get("state_history_count")
    if source_count != target_count:
        if (source_count, target_count) != (21, 1):
            raise ValueError("observation transition only composes with the explicit 21 -> 1 state migration")
        unchanged["state_history_count"] = source_count
    if unchanged != previous:
        raise ValueError("observation transition cannot change existing model/teacher temporal fields")
    return {
        "format": "zeta0.predictive_world.observation_initialization.v1",
        "source_weights_sha256": source["weights_sha256"],
        "source_config_sha256": canonical_sha256(source_payload),
        "source_temporal_abi_sha256": canonical_sha256(previous),
        "target_temporal_abi_sha256": canonical_sha256(target),
        "target_observation_contract": deployment,
        "preserved_future_limits": _source_future_limits(source_payload),
        "target_action_label_mode": "held_command_v1",
        # Keep the previously published three-field v4 records unchanged.
        "allowed_temporal_fields": sorted(set(target) & REAL_TEMPORAL_FIELDS),
        "observation_jitter": "recorded_real_snapshots_no_synthetic_timestamp_relabeling",
    }


def verify_observation_contract_transition(record, source, source_payload, target_temporal, *, target_data=None):
    expected = observation_contract_transition(source, source_payload, target_temporal)
    if record != expected:
        raise ValueError("observation transition is not bound to the exact source and target ABI")
    if target_data is not None:
        data = asdict(target_data) if is_dataclass(target_data) else dict(target_data)
        if make_deployment_observation_contract(data) != expected["target_observation_contract"]:
            raise ValueError("observation transition target data disagrees with the recorded collector contract")
        if _source_future_limits({"data": data}) != expected["preserved_future_limits"]:
            raise ValueError("observation transition cannot change future teacher/action age limits")
        if data.get("real_action_label_mode") != expected["target_action_label_mode"]:
            raise ValueError("observation transition target action labels disagree")
    return dict(record)
