"""Source-bound transition from state history to one current observation."""
from psi.data.predictive_world.cache import canonical_sha256


def current_state_transition(source, source_payload):
    count = source["metadata"]["temporal_abi"]["state_history_count"]
    if type(count) is not int or count <= 1:
        raise ValueError("current-state transition requires a source with state history")
    return {
        "format": "zeta0.predictive_world.current_state_initialization.v1",
        "source_weights_sha256": source["weights_sha256"],
        "source_config_sha256": canonical_sha256(source_payload),
        "source_state_history_count": count,
        "target_state_history_count": 1,
        "state_position_transfer": "last_current_row",
    }


def verify_current_state_transition(record, source, source_payload, target_temporal):
    if record != current_state_transition(source, source_payload):
        raise ValueError("current-state transition is not bound to the exact source")
    expected = {**source["metadata"]["temporal_abi"], "state_history_count": 1}
    if target_temporal != expected:
        raise ValueError("current-state transition may only change state_history_count to 1")
    return dict(record)
