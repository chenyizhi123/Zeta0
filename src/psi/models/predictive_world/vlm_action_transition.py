"""Source-bound frozen-VLM replacement and action-grid initialization.

This is a new task adaptation, not an exact resume or equivalent backbone.
Only the two action-position tables are resized; task weights and fitted
latent statistics otherwise retain their pretrained initialization.
"""
from __future__ import annotations

import copy
import math

from psi.data.predictive_world.cache import canonical_sha256


def make_vlm_action_transition(source, payload, *, encoder_id, processor_id, temporal, qwen_model,
                               processor_migration=None, source_config=None):
    metadata = source["metadata"]
    if (source["phase"] != "joint"
            or metadata["resolved_config_sha256"] != canonical_sha256(payload)):
        raise ValueError("VLM/action transition requires the exact joint source config")
    previous = metadata["temporal_abi"]
    restored = dict(temporal)
    for key in ("action_horizon", "action_hz"):
        restored[key] = previous[key]
    # The independently verified current-state migration owns this change.
    if restored.get("state_history_count") != previous.get("state_history_count"):
        if restored.get("state_history_count") != 1 or previous.get("state_history_count", 0) <= 1:
            raise ValueError("VLM/action transition only composes with current-state initialization")
        restored["state_history_count"] = previous["state_history_count"]
    if restored != previous:
        raise ValueError("VLM/action transition cannot change other temporal fields")
    horizon, hz = temporal["action_horizon"], temporal["action_hz"]
    if (type(horizon) is not int or horizon < 2 or isinstance(hz, bool)
            or not math.isfinite(hz) or hz <= 0 or horizon / hz < temporal["future_group_ends_s"][-1]):
        raise ValueError("invalid target action grid")
    same_encoder = encoder_id == metadata["encoder_id"]
    if processor_migration is not None:
        if not same_encoder or source_config is None:
            raise ValueError("processor migration requires the same encoder and exact source config")
        from psi.data.predictive_world.processor_migration import verify_processor_migration
        verify_processor_migration(processor_migration, source, processor_id, source_config)
    if same_encoder and (
        (processor_id != metadata["processor_id"] and processor_migration is None)
        or all(temporal[key] == previous[key] for key in ("action_horizon", "action_hz"))
    ):
        raise ValueError("action-grid-only transition requires unchanged VLM identity and a changed grid")
    if (not encoder_id.startswith("qwen3vl:") or not processor_id.startswith("qwen3vl-processor:")
            or not qwen_model):
        raise ValueError("VLM replacement requires a distinct, fingerprinted Qwen backbone")
    result = {
        "format": ("zeta0.predictive_world.action_grid_initialization.v1" if same_encoder
                   else "zeta0.predictive_world.vlm_action_initialization.v1"),
        "source_weights_sha256": source["weights_sha256"],
        "source_config_sha256": canonical_sha256(payload),
        "source_encoder_id": metadata["encoder_id"], "target_encoder_id": encoder_id,
        "source_processor_id": metadata["processor_id"], "target_processor_id": processor_id,
        "target_qwen_model": str(qwen_model),
        "source_temporal_abi": copy.deepcopy(previous), "target_temporal_abi": copy.deepcopy(temporal),
        "action_position_transfer": "linear_index_interpolation_align_corners_true",
        "vlm_trainable": False, "reuse_source_feature_cache": False,
        "preserve_pretrained_latent_normalizer": True,
        "optimizer_restored": False, "historical_equivalence": False,
    }

    if processor_migration is not None:
        result["processor_migration"] = copy.deepcopy(processor_migration)
    return result

def verify_vlm_action_transition(record, source, payload, expected, *,
                                 processor_migration=None, source_config=None):
    wanted = make_vlm_action_transition(source, payload, encoder_id=expected["encoder_id"],
        processor_id=expected["processor_id"], temporal=expected["temporal_abi"],
        qwen_model=expected["architecture"]["qwen_model"],
        processor_migration=processor_migration, source_config=source_config)
    if record != wanted:
        raise ValueError("VLM/action transition is not bound to the exact source and target")
    previous, target = dict(source["metadata"]["architecture"]), dict(expected["architecture"])
    previous.pop("qwen_model", None)
    target.pop("qwen_model", None)
    if previous != target:
        raise ValueError("VLM/action transition cannot change the task model architecture")
    return dict(record)


def migrate_action_position_tensors(tensors, record):
    import torch.nn.functional as functional

    source_steps = record["source_temporal_abi"]["action_horizon"]
    target_steps = record["target_temporal_abi"]["action_horizon"]
    result = dict(tensors)
    for name in ("teacher", "student"):
        key = name + ".action_position"
        value = tensors[key]
        if value.ndim != 2 or value.shape[0] != source_steps:
            raise ValueError("source action-position table disagrees with its temporal ABI")
        if source_steps != target_steps:
            result[key] = functional.interpolate(value.float().T[None], size=target_steps,
                mode="linear", align_corners=True)[0].T.to(value.dtype).contiguous()
    return result
