"""Source-bound post-training pooling and latent coordinate adaptation.

Calibration is a separate pass over training data before any optimizer update.
F keeps its pretrained output coordinates internally; a persisted affine map
converts them to the newly fitted regression coordinates at every F call.
"""
from __future__ import annotations

import copy

from psi.data.predictive_world.cache import canonical_sha256


RAW_LATENT_TRANSITION_FORMAT = "zeta0.predictive_world.raw_latent_initialization.v1"


def make_raw_latent_transition(source, payload):
    """Bind a whitened -> raw output conversion to one immutable joint source.

    This conversion does not estimate any new statistics. The source normalizer
    is used once to compose the predictor's persisted output affine transform.
    """
    metadata = source["metadata"]
    if (source.get("phase") != "joint"
            or metadata["resolved_config_sha256"] != canonical_sha256(payload)):
        raise ValueError("raw latent transition requires the exact joint source/config")
    if (payload.get("training", {}).get("latent_prediction_space", "whitened") != "whitened"
            or metadata.get("loss_contract", {}).get("latent_prediction_space", "whitened") != "whitened"):
        raise ValueError("raw latent transition requires a whitened source")
    return dict(
        format=RAW_LATENT_TRANSITION_FORMAT,
        source_weights_sha256=source["weights_sha256"],
        source_config_sha256=canonical_sha256(payload),
        source_normalization_id=metadata["normalization_id"],
        source_prediction_space="whitened", target_prediction_space="raw",
        output_coordinate_transfer="compose_source_unwhitening_per_group_slot_channel_v1",
        calibration_batches=0, optimizer_restored=False,
    )


def verify_raw_latent_transition(record, source, payload, expected):
    """Validate this delta and restore source coordinates for orthogonal checks."""
    if record != make_raw_latent_transition(source, payload):
        raise ValueError("raw latent migration is not bound to the exact source")
    previous = copy.deepcopy(expected)
    if previous["loss_contract"].pop("latent_prediction_space", None) != "raw":
        raise ValueError("raw latent migration requires a raw prediction loss contract")
    if previous["architecture"]["latent_transition"].pop("output_affine", False) is not True:
        raise ValueError("raw latent migration requires a persisted output affine")
    source_affine = source["metadata"]["architecture"]["latent_transition"].get("output_affine", False)
    if source_affine:
        previous["architecture"]["latent_transition"]["output_affine"] = True
    return previous


def migrate_raw_latent_tensors(tensors, transition, *, source_output_affine=False):
    """Compose old unwhitening into F without modifying any source tensor.

    For an already migrated source F, ``u = a*f(h) + b``. The new F returns
    ``std*a*f(h) + mean + std*b``. Shapes remain (group, slot, channel); a shared
    output Linear cannot in general represent this group/slot-dependent affine.
    """
    import torch

    if not transition.output_affine:
        raise ValueError("raw latent migration requires an affine predictor")
    shape = transition.latent_shape
    try:
        mean, std, fitted = (tensors["normalizer." + key] for key in ("mean", "std", "fitted"))
    except KeyError as error:
        raise ValueError("raw latent migration requires source normalizer tensors") from error
    if (tuple(mean.shape) != shape or tuple(std.shape) != shape
            or fitted.shape != torch.Size([]) or fitted.dtype != torch.bool or not bool(fitted)
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("raw latent migration requires fitted finite source coordinates")
    names = tuple("latent_transition." + key for key in
                  ("output_scale", "output_shift", "output_calibrated"))
    if source_output_affine:
        if any(name not in tensors for name in names):
            raise ValueError("source affine tensors are incomplete")
        scale, shift, calibrated = (tensors[name] for name in names)
        if (tuple(scale.shape) != shape or tuple(shift.shape) != shape
                or calibrated.shape != torch.Size([]) or calibrated.dtype != torch.bool or not bool(calibrated)
                or not torch.isfinite(scale).all() or not torch.isfinite(shift).all()):
            raise ValueError("raw latent migration requires calibrated finite source affine")
    else:
        if any(name in tensors for name in names):
            raise ValueError("source affine tensors disagree with source architecture")
        scale, shift = torch.ones_like(std), torch.zeros_like(mean)
    raw_scale, raw_shift = std * scale, mean + std * shift
    if not torch.isfinite(raw_scale).all() or not torch.isfinite(raw_shift).all():
        raise ValueError("raw latent migration produces nonfinite output coordinates")
    result = dict(tensors)
    result.update(zip(names, (raw_scale, raw_shift, torch.tensor(True, device=mean.device))))
    return result


def make_posttrain_transition(source, payload, *, current_visual_pooling, calibration_batches):
    if (source.get("phase") != "joint"
            or source["metadata"]["resolved_config_sha256"] != canonical_sha256(payload)):
        raise ValueError("post-training transition requires the exact joint source/config")
    if type(calibration_batches) is not int or calibration_batches < 0:
        raise ValueError("calibration_batches must be a nonnegative integer")
    if current_visual_pooling not in {"canvas_v1", "content_area_v1"}:
        raise ValueError("unsupported current visual pooling")
    if (payload["data"].get("current_visual_pooling", "canvas_v1") != "canvas_v1"
            or payload["model"]["latent_transition"].get("output_affine", False)):
        raise ValueError("post-training transition requires a source without prior coordinate/pooling migration")
    if current_visual_pooling == "canvas_v1" and not calibration_batches:
        raise ValueError("post-training transition must declare an actual change")
    return dict(
        format="zeta0.predictive_world.posttrain_initialization.v1",
        source_weights_sha256=source["weights_sha256"],
        source_config_sha256=canonical_sha256(payload),
        source_normalization_id=source["metadata"]["normalization_id"],
        current_visual_pooling=current_visual_pooling,
        calibration_batches=calibration_batches,
        calibration_split="train",
        calibration_encoder="post_migration_frozen_initial_ema",
        calibration_statistics="global_fp64_population_moments_std_floor_0.05",
        prediction_coordinates="per_group_slot_channel_affine_v1" if calibration_batches else "source",
        calibration_optimizer_updates=0,
    )


def verify_posttrain_transition(record, source, payload, expected):
    """Verify the new changes, then return metadata for existing orthogonal migrations."""
    pooling = expected["temporal_abi"].get("current_visual_pooling", "canvas_v1")
    wanted = make_posttrain_transition(source, payload, current_visual_pooling=pooling,
                                      calibration_batches=record.get("calibration_batches"))
    if record != wanted:
        raise ValueError("post-training migration is not bound to the exact source/target")
    previous_contract = copy.deepcopy(expected)
    affine = previous_contract["architecture"]["latent_transition"].pop("output_affine", False)
    if affine != bool(record["calibration_batches"]):
        raise ValueError("post-training calibration requires matching output affine architecture")
    previous_contract["temporal_abi"].pop("current_visual_pooling", None)
    return previous_contract


def add_output_coordinate_tensors(tensors, transition):
    """Explicit additive transfer; all pre-existing keys still undergo strict validation."""
    import torch

    if not transition.output_affine:
        return tensors
    result = dict(tensors)
    for key, value in (("output_scale", torch.ones(transition.latent_shape)),
                       ("output_shift", torch.zeros(transition.latent_shape)),
                       ("output_calibrated", torch.tensor(False))):
        name = "latent_transition." + key
        if name in result:
            raise ValueError("source already contains migrated output coordinates")
        result[name] = value
    return result


def install_calibrated_coordinates(normalizer, transition, count, total, squared_total):
    """Validate all statistics before changing buffers; preserve raw predictions exactly algebraically."""
    import torch

    if not transition.output_affine or bool(transition.output_calibrated) or not bool(normalizer.fitted):
        raise ValueError("calibration requires a fitted source normalizer and an uncalibrated affine predictor")
    shape = normalizer.latent_shape
    if transition.latent_shape != shape or not torch.isfinite(normalizer.mean).all() or (
        not torch.isfinite(normalizer.std).all() or (normalizer.std <= 0).any()
    ):
        raise ValueError("source normalizer and predictor coordinates must agree and be finite")
    if any(tuple(x.shape) != shape for x in (count, total, squared_total)):
        raise ValueError("calibration moments disagree with latent shape")
    if any(not torch.isfinite(x).all() for x in (count, total, squared_total)) or (count <= 0).any():
        raise ValueError("all latent coordinates require finite calibration moments and positive counts")
    mean = (total.double() / count.double()).float()
    variance = (squared_total.double() / count.double() - (total.double() / count.double()).square()).clamp_min(0)
    raw_std = variance.sqrt().float()
    std = raw_std.clamp_min(normalizer.std_floor)
    scale = normalizer.std / std
    shift = (normalizer.mean - mean) / std
    if any(not torch.isfinite(x).all() for x in (mean, std, scale, shift)):
        raise ValueError("calibration produces nonfinite coordinate transform")
    with torch.no_grad():
        normalizer.mean.copy_(mean)
        normalizer.std.copy_(std)
        transition.output_scale.copy_(scale)
        transition.output_shift.copy_(shift)
        transition.output_calibrated.fill_(True)
    return dict(samples_per_coordinate_min=int(count.min()), samples_per_coordinate_max=int(count.max()),
                std_min=float(std.min()), std_median=float(std.median()), std_max=float(std.max()),
                std_floor_fraction=float((raw_std < normalizer.std_floor).float().mean()))


def calibrate_posttrain(built, batches, *, batch_count, device, seed, distributed=False):
    """Use the real observation/E paths; no loss/backward, EMA update, or RNG drift."""
    import torch
    import torch.distributed as dist
    from psi.models.predictive_world.engine import move_to_device
    from psi.models.predictive_world.monitoring import isolated_evaluation
    from psi.models.predictive_world.phases import _encoder_arguments
    from psi.models.predictive_world.training import module_state_sha256, tensor_mapping_sha256

    if type(batch_count) is not int or batch_count <= 0:
        raise ValueError("calibration requires a positive batch count")
    normalizer = built.reference_modules["normalizer"]
    transition = built.forecast_modules["latent_transition"]
    encoder = built.reference_modules["encoder"]
    before = module_state_sha256(normalizer)
    count = torch.zeros(normalizer.latent_shape, dtype=torch.float64, device=device)
    total, squared_total = torch.zeros_like(count), torch.zeros_like(count)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with isolated_evaluation(built.model, seed=seed, cuda_devices=cuda_devices):
        iterator = iter(batches)
        for _ in range(batch_count):
            batch = move_to_device(next(iterator), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features = built.model.observation_encoder(batch["observed"])
                batch["anchor_patches"] = features.current_visual
                batch["anchor_patch_valid"] = torch.ones(features.current_visual.shape[:2], dtype=torch.bool, device=device)
                latent = encoder(**_encoder_arguments(batch)).double()
            mask = batch["group_valid"][:, :, None, None].expand_as(latent).bool()
            finite = torch.isfinite(latent[mask]).all().to(torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise ValueError("calibration encountered nonfinite valid latent targets")
            clean = torch.where(mask, latent, torch.zeros_like(latent))
            count += mask.sum(dim=0)
            total += clean.sum(dim=0)
            squared_total += clean.square().sum(dim=0)
        if distributed:
            for value in (count, total, squared_total):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        report = install_calibrated_coordinates(normalizer, transition, count, total, squared_total)
    report.update(format="zeta0.predictive_world.posttrain_calibration.v1", batches_per_rank=batch_count,
                  world_size=dist.get_world_size() if distributed else 1,
                  source_normalization_id=before, normalization_id=module_state_sha256(normalizer),
                  output_coordinates_sha256=tensor_mapping_sha256({
                      key: getattr(transition, key) for key in ("output_scale", "output_shift", "output_calibrated")
                  }), optimizer_updates=0)
    return report
