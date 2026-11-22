import copy

import numpy as np
import pytest
import torch
from torch import nn

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.cache import canonical_sha256
from psi.deploy.real_sonic_protocol import MeasuredHistory, validate_request
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.initialization import transfer_joint_tensors
from psi.models.predictive_world.latent_transition import TemporalLatentTransition
from psi.models.predictive_world.state_history_transition import (
    current_state_transition, verify_current_state_transition,
)


def small_modules(count):
    protocols = {"real": {"dim": 76, "mean": [0.] * 76, "std": [1.] * 76}}
    states = {"real": {"dim": 41}}
    def expert():
        return TemporalNativeActionExpert(protocols, {"real": 41}, qwen_dim=6, context_tokens=2,
            latent_dim=4, latent_groups=3, latent_slots=8, state_history_count=count,
            action_steps=30, hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32)
    return dict(encoder=nn.Linear(2, 2), normalizer=nn.Linear(2, 2), readout=nn.Linear(2, 2),
        history_encoder=PredictiveHistoryEncoder(qwen_dim=6, visual_dim=6, state_history_count=count,
            context_tokens=2, hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32),
        history_state_adapters=StateAdapterBank({"real": 41}), teacher=expert(), student=expert(),
        latent_transition=TemporalLatentTransition(protocols, context_dim=16, latent_shape=(3, 8, 4),
            hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32)), protocols, states


def test_current_state_transfer_keeps_latest_positions_and_other_weights_exact():
    source, protocols, states = small_modules(21)
    target, _, _ = small_modules(1)
    with torch.no_grad():
        source["history_encoder"].state_history_position.copy_(torch.arange(21)[:, None].expand(21, 16))
    tensors = {f"{name}.{key}": value.clone() for name, module in source.items() for key, value in module.state_dict().items()}
    transfer_joint_tensors(tensors, target, source_protocols=protocols, target_protocols=protocols,
        source_states=states, target_states=states, current_state_source_count=21)
    for name, module in target.items():
        for key, value in module.state_dict().items():
            original = tensors[f"{name}.{key}"]
            if key in {"state_history_position", "state_position"}:
                original = original[-1:]
            torch.testing.assert_close(value, original, rtol=0, atol=0)
    # Dropping masked history is equivalent to a true one-state H after migration.
    history = torch.randn(2, 21, 64)
    valid = torch.zeros(2, 21, dtype=torch.bool)
    valid[:, -1] = True
    times = torch.linspace(-.4, 0, 21)[None].expand(2, 21)
    arguments = dict(qwen_context=torch.randn(2, 3, 6), context_valid=torch.ones(2, 3, dtype=torch.bool),
                     current_visual=torch.randn(2, 16, 6))
    source["history_encoder"].eval()
    target["history_encoder"].eval()
    expected = source["history_encoder"](**arguments, adapted_states=history, state_valid=valid, state_times=times)
    actual = target["history_encoder"](**arguments, adapted_states=history[:, -1:], state_valid=valid[:, -1:], state_times=times[:, -1:])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_wrong_position_migration_rejects_before_mutating_target():
    source, protocols, states = small_modules(21)
    target, _, _ = small_modules(1)
    before = copy.deepcopy(target["encoder"].state_dict())
    tensors = {f"{name}.{key}": value.clone() for name, module in source.items() for key, value in module.state_dict().items()}
    with pytest.raises(ValueError, match="position transfer"):
        transfer_joint_tensors(tensors, target, source_protocols=protocols, target_protocols=protocols,
            source_states=states, target_states=states, current_state_source_count=20)
    for key, value in target["encoder"].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_transition_is_source_bound_and_rejects_other_temporal_changes():
    payload = {"data": {"state_history_count": 21}}
    temporal = {"state_history_count": 21, "state_query_hz": 50, "action_hz": 50}
    source = {"weights_sha256": "source", "metadata": {"temporal_abi": temporal}}
    record = current_state_transition(source, payload)
    assert record["source_config_sha256"] == canonical_sha256(payload)
    verify_current_state_transition(record, source, payload, {**temporal, "state_history_count": 1})
    with pytest.raises(ValueError, match="exact source"):
        verify_current_state_transition({**record, "source_weights_sha256": "other"}, source, payload, temporal)
    with pytest.raises(ValueError, match="may only change"):
        verify_current_state_transition(record, source, payload, {**temporal, "state_history_count": 1, "action_hz": 30})


def test_real_wire_current_state_has_no_earlier_slots_and_rejects_history():
    from test_real_sonic_deployment import request
    history = MeasuredHistory(state_count=1)
    history.append(.6, np.zeros(41))
    history.append(.99, np.ones(41))
    history.append(1.01, np.full(41, 2))
    req = request()
    req["state"] = history.aligned(1.)
    assert np.asarray(req["state"]["values"]).shape == (1, 41)
    assert req["state"]["values"] == [[1.] * 41]
    assert req["state"]["times"] == pytest.approx([-.01])
    validate_request(req, state_count=1)
    with pytest.raises(ValueError, match="State requires"):
        validate_request(request(), state_count=1)
    with pytest.raises(ValueError, match="State requires"):
        validate_request(req)
    from psi.deploy.real_sonic_client import check_health
    from psi.deploy.real_sonic_server import InferenceService
    service = InferenceService(lambda request, seed: np.zeros((30, 76)),
        {"task": "doll", "source_completed_updates": 1, "weights_sha256": "test",
         "temporal_abi": {"state_history_count": 1}})
    check_health(service.health(), state_count=1)
    with pytest.raises(ValueError, match="state_history_count"):
        check_health(service.health())


def test_real_training_index_uses_only_latest_state_without_changing_labels(tmp_path):
    from test_real_sonic_post_training import recording, registries
    from psi.data.predictive_world.real_sonic_post_training import inspect_dataset, extend_registries, episode_windows, MEASUREMENT
    from psi.data.predictive_world.index import predictive_world_temporal_abi
    from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
    recording(tmp_path)
    info, instruction, episodes, audit, stats = inspect_dataset(tmp_path, task="doll")
    registry = extend_registries(registries(), info, stats, task="doll", audit_identity="test")
    schema = MeasurementSchema.from_mapping(MEASUREMENT, registry["measurement_registry"][MEASUREMENT])
    capability = SourceCapability.from_mapping("doll", registry["capability_registry"]["doll"])
    def rows(count):
        temporal = predictive_world_temporal_abi(history_visual_offsets_s=[-.4, -.2, 0], state_history_count=count,
            state_query_hz=50, future_group_ends_s=[.2, .4, .6], action_horizon=30, action_hz=50,
            visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04)
        return episode_windows(episodes[0], task="doll", instruction=instruction, temporal=temporal,
                               capability=capability, measurement_schema=schema)
    for old, current in zip(rows(21), rows(1)):
        assert len(current["state_indices"]) == len(current["state_valid"]) == 1
        assert current["state_indices"] == old["state_indices"][-1:]
        assert current["state_valid"] == old["state_valid"][-1:]
        np.testing.assert_allclose(current["state_actual_times"], old["state_actual_times"][-1:], equal_nan=True)
        assert current["action_indices"] == old["action_indices"]
        assert current["action_valid"] == old["action_valid"]
        assert current["future_visual_indices"] == old["future_visual_indices"]
