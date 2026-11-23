"""Load this research inference weight package into an existing Zeta0-Humanoid graph.

This helper does not create an HTTP server, change robot control code, or bypass
formal deployment evaluation. Build matching modules using the original config,
registries and external Qwen base model. The normalizer module uses the source
checkpoint name ``normalizer`` (not ``latent_normalizer``).
"""
from pathlib import Path
import hashlib
import json
import shutil
from safetensors.torch import load_file

INFERENCE_FORMAT = "zeta0.research_inference_weights.v1"
INFERENCE_MODULES = frozenset({
    "history_state_adapters", "history_encoder", "latent_transition", "normalizer", "student",
})
RUNTIME_METADATA_FILES = frozenset({
    "original_run_config.json", "source_checkpoint.json", "feature_provenance.json",
    "source_index_metadata.json",
    "registries/action_protocol_registry.json", "registries/state_schema_registry.json",
    "registries/measurement_registry.json", "registries/capability_registry.json",
})


def verify_inference_package(package_dir, *, verify_weights=True):
    """Check packaged bytes and their source contract without training assets.

    Metadata-only verification is used before allocating the policy graph. The
    weight loader always performs the full weight check before copying tensors.
    Old packages without a metadata inventory retain their existing hash contract.
    """
    from psi.data.predictive_world.cache import file_sha256

    root = Path(package_dir).resolve()
    manifest = json.loads((root / "inference_manifest.json").read_text())
    if manifest.get("format") != INFERENCE_FORMAT or set(manifest.get("module_names", ())) != INFERENCE_MODULES:
        raise ValueError("Research inference package format/modules mismatch")
    files = manifest.get("package_files_sha256")
    if files is not None:
        if (not (root / ".complete").is_file() or (root / ".complete").read_text().strip() != "complete"
                or not isinstance(files, dict)
                or not RUNTIME_METADATA_FILES.issubset(files)):
            raise ValueError("Research inference publication is incomplete")
        for name, digest in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Research inference metadata checksum mismatch: {name}")
        _verify_source_contract(root, manifest)
    weights = (root / manifest["weights_file"]).resolve()
    if (weights.parent != root or not weights.is_file()
            or (verify_weights and file_sha256(weights) != manifest["weights_sha256"])):
        raise ValueError("Inference weight checksum mismatch")
    return manifest


def _verify_source_contract(root, manifest):
    """Bind executable manifest fields to the copied immutable source evidence."""
    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.deploy.real_sonic_protocol import CAMERAS, PROTOCOL, STATE_SCHEMA
    from psi.real_observation_contract import make_deployment_observation_contract

    def read(name):
        return json.loads((root / name).read_text())

    source = read("source_checkpoint.json")
    config = read("original_run_config.json")
    metadata = source["metadata"]
    if (source.get("phase") != "joint" or type(source.get("completed_updates")) is not int
            or source["completed_updates"] <= 0
            or canonical_sha256(config) != metadata["resolved_config_sha256"]
            or file_sha256(root / "source_checkpoint.json") != manifest["source_checkpoint_sha256"]):
        raise ValueError("Research inference source checkpoint/config identity mismatch")
    expected = {
        "source_completed_updates": source["completed_updates"],
        "source_weights_sha256": source["weights_sha256"],
        "architecture": metadata["architecture"], "temporal_abi": metadata["temporal_abi"],
        "encoder_id": metadata["encoder_id"], "processor_id": metadata["processor_id"],
        "source_code_revision": metadata["code_revision"], "source_code_fingerprint": metadata["code_fingerprint"],
        "action_protocol_id": PROTOCOL, "state_schema_id": STATE_SCHEMA, "visual_keys": list(CAMERAS),
        "rtc_max_prefix_steps": metadata["optimization_contract"]["rtc_training_max_delay"],
        "action_condition_time_mode": config["training"].get("action_condition_time_mode"),
        "future_component_alignment": config["data"].get("future_component_alignment"),
        "observation_contract": make_deployment_observation_contract(config["data"]),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"Research inference manifest/source {name.replace('_', ' ')} mismatch")
    if (expected["rtc_max_prefix_steps"] != config["training"].get("rtc_training_max_delay")
            or expected["action_condition_time_mode"] != metadata["optimization_contract"].get("action_condition_time_mode")):
        raise ValueError("Research inference source RTC/action-time contract mismatch")
    provenance = read("feature_provenance.json")
    if any(provenance.get(name) != metadata[name] for name in ("encoder_id", "processor_id")):
        raise ValueError("Research inference source feature provenance mismatch")
    binding = read("source_index_metadata.json")["provenance"]["registry_binding"]
    for name in ("action_protocol_registry", "state_schema_registry", "measurement_registry", "capability_registry"):
        digest = file_sha256(root / "registries" / (name + ".json"))
        if binding.get(name + "_sha256") != digest:
            raise ValueError(f"Research inference source registry binding mismatch: {name}")
        metadata_key = {"action_protocol_registry": "protocol_registry_sha256",
                        "measurement_registry": "measurement_registry_sha256"}.get(name)
        if metadata_key is not None and metadata[metadata_key] != digest:
            raise ValueError(f"Research inference checkpoint registry binding mismatch: {name}")


def export_real_sonic_research_policy(checkpoint_dir, config_path, output_dir):
    """Package an exact joint checkpoint for real-SONIC research inference.

    Does not construct models, load Qwen, initialize CUDA, or grant formal
    deployment acceptance. Source data and historical packages are immutable.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    from psi.config.predictive_world import (
        load_predictive_world_config, predictive_world_architecture_contract,
    )
    from psi.data.predictive_world.assets import resolve_asset_path
    from psi.data.predictive_world.cache import (
        canonical_sha256, file_sha256, verify_predictive_world_feature_manifest,
    )
    from psi.data.predictive_world.index import (
        INDEX_BINDING_FORMAT, assert_predictive_world_index_temporal_abi,
    )
    from psi.deploy.real_sonic_protocol import CAMERAS, MAX_RTC_PREFIX, PROTOCOL, STATE_SCHEMA, validate_action_grid
    from psi.models.predictive_world.checkpoint import read_predictive_world_bundle
    from psi.real_observation_contract import make_deployment_observation_contract

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Research inference output already exists: {output}")
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    if output.is_relative_to(checkpoint):
        raise ValueError("Research output must be outside the immutable source checkpoint")
    source = read_predictive_world_bundle(checkpoint)
    if source["phase"] != "joint" or source["completed_updates"] <= 0:
        raise ValueError("Research real-SONIC packaging requires a completed joint checkpoint")
    config, payload = load_predictive_world_config(config_path, allow_existing_output_override=True)
    metadata = source["metadata"]
    if metadata["resolved_config_sha256"] != canonical_sha256(payload):
        raise ValueError("Source config checksum mismatch")
    temporal = {
        "history_visual_offsets_s": list(config.data.history_visual_offsets_s),
        **config.data.observation_contract,
        "state_history_count": config.data.state_history_count,
        "state_query_hz": config.data.state_query_hz,
        "future_group_ends_s": list(config.data.future_group_ends_s),
        "action_horizon": config.data.action_horizon,
        "action_hz": config.data.action_hz,
    }
    if (config.training.phase != "joint" or temporal != metadata["temporal_abi"]
            or metadata["architecture"] != predictive_world_architecture_contract(config.model)
            or config.data.observed_visual_mode != "current_multiview"
            or config.training.student_latent_mode != "predicted"
            or config.data.state_query_hz != 50
            or config.data.state_history_count not in (1, 21)):
        raise ValueError("Source config architecture/temporal ABI is incompatible with real-SONIC")
    validate_action_grid(config.data.action_hz, config.data.action_horizon)
    trained = config.training.rtc_training_max_delay
    recorded = metadata.get("optimization_contract", {})
    if (type(trained) is not int or not 0 <= trained <= MAX_RTC_PREFIX
            or recorded.get("rtc_training_max_delay") != trained
            or recorded.get("action_condition_time_mode") != config.training.action_condition_time_mode):
        raise ValueError("Source RTC/action-time contract mismatch or unsupported capacity")
    observation_contract = make_deployment_observation_contract(payload["data"])
    if observation_contract is not None and (
        config.data.real_action_label_mode != "held_command_v1"
        or config.training.action_condition_time_mode != "fixed_action_grid_v1"
    ):
        raise ValueError("Collector snapshots require held-command labels and fixed action slots")

    def asset(path):
        return resolve_asset_path(path, config.runtime.asset_path_map).resolve()

    registries = {
        "action_protocol_registry": asset(config.data.action_protocol_registry),
        "state_schema_registry": asset(config.data.state_schema_registry),
        "measurement_registry": asset(config.data.measurement_registry),
        "capability_registry": asset(config.data.canonical_future_capabilities),
    }
    registry_values = {name: json.loads(path.read_text()) for name, path in registries.items()}
    if (registry_values["action_protocol_registry"].get(PROTOCOL, {}).get("dim") != 76
            or registry_values["state_schema_registry"].get(STATE_SCHEMA, {}).get("dim") != 41):
        raise ValueError("Source native action/state registry does not describe motion76/q41")
    capabilities = registry_values["capability_registry"]
    if len(capabilities) != 1:
        raise ValueError("Research real-SONIC packaging requires one audited source task")
    source_id, capability = next(iter(capabilities.items()))
    if (capability.get("action_protocol_id") != PROTOCOL
            or sorted(capability.get("visual_keys", ())) != list(CAMERAS)):
        raise ValueError("Source capabilities disagree with the real-SONIC camera/action interface")
    index = asset(config.data.index_path)
    index_meta = assert_predictive_world_index_temporal_abi(index, config.data)
    alignment = index_meta.get("provenance", {}).get("time_alignment", {}).get("future_component_alignment")
    if alignment != config.data.future_component_alignment:
        raise ValueError("Source config/index future_component_alignment mismatch")
    alignment_snapshot = index_meta.get("provenance", {}).get("time_alignment", {})
    for name in ("real_observation_mode", "real_observation_contract", "real_action_label_mode", "real_action_projection",
                 "real_snapshot_max_age_s"):
        if alignment_snapshot.get(name) != payload["data"].get(name):
            raise ValueError(f"Source config/index {name} mismatch")
    registry_binding = index_meta.get("provenance", {}).get("registry_binding", {})
    for name, path in registries.items():
        if registry_binding.get(name + "_sha256") != file_sha256(path):
            raise ValueError(f"Source registry binding checksum mismatch: {name}")
    if (metadata["protocol_registry_sha256"] != file_sha256(registries["action_protocol_registry"])
            or metadata["measurement_registry_sha256"] != file_sha256(registries["measurement_registry"])):
        raise ValueError("Source checkpoint registry checksum mismatch")
    feature_path = asset(config.data.feature_cache)
    if feature_path.is_dir():
        feature_path /= "feature_manifest.json"
    feature = verify_predictive_world_feature_manifest(feature_path)
    expected_binding = {
        "format": INDEX_BINDING_FORMAT, "semantic_identity": index_meta["semantic_identity"],
        "index_sha256": index_meta["index_sha256"], "lookup_sha256": index_meta["lookup_sha256"],
        "temporal_abi_sha256": canonical_sha256(index_meta["temporal_abi"]),
    }
    if feature["index_binding"] != expected_binding:
        raise ValueError("Source feature cache is bound to a different index")
    provenance = feature["provenance"]
    if (provenance.get("feature_shape") != [16, 2048] or provenance.get("dtype") != "float16"
            or provenance.get("image_preprocessing") != "aspect_v1"
            or provenance.get("feature_layer") != "post_merger_content_area_overlap_4x4_v1"):
        raise ValueError("Source Qwen feature ABI is incompatible with real-SONIC inference")
    if any(provenance.get(key) != metadata[key] for key in ("encoder_id", "processor_id")):
        raise ValueError("Source feature encoder/processor identity mismatch")
    fingerprint = {
        "index": index_meta["index_sha256"], "feature_cache": feature["cache_identity"],
        "capabilities": file_sha256(registries["capability_registry"]),
    }
    if config.model.goal_intent_enabled:
        fingerprint["goal_supervision"] = file_sha256(asset(config.data.goal_supervision_path))
    if canonical_sha256(fingerprint) != metadata["data_fingerprint"]:
        raise ValueError("Source checkpoint data fingerprint mismatch")
    with index.open() as stream:
        first = json.loads(next(line for line in stream if line.strip()))
    instruction = first.get("materialization", {}).get("instruction")
    if first.get("source_id") != source_id or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Source index lacks its audited task instruction")

    tensors = {}
    with safe_open(checkpoint / source["weights_file"], framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        actual_modules = {key.split(".", 1)[0] for key in keys}
        if actual_modules != set(source["module_names"]) or not INFERENCE_MODULES.issubset(actual_modules):
            raise ValueError("Source weight modules disagree with the checkpoint manifest")
        for key in keys:
            if key.split(".", 1)[0] in INFERENCE_MODULES:
                tensors[key] = reader.get_tensor(key)
    output.mkdir(parents=True, exist_ok=False)
    try:
        save_file(tensors, output / "inference.safetensors", metadata={"format": INFERENCE_FORMAT})
        shutil.copyfile(config_path, output / "original_run_config.json")
        shutil.copyfile(checkpoint / "predictive_world_checkpoint.json", output / "source_checkpoint.json")
        (output / "registries").mkdir()
        for name, path in registries.items():
            shutil.copyfile(path, output / "registries" / (name + ".json"))
        for name, value in (("feature_provenance.json", provenance), ("source_index_metadata.json", index_meta)):
            (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        gr00t_flag = " --gr00t-model GR00T_BASE" if config.model.host_architecture == "gr00t_n1d7" else ""
        gr00t_line = (
            f"Required external GR00T base: {config.model.gr00t_model}\n"
            if config.model.host_architecture == "gr00t_n1d7"
            else ""
        )
        (output / "RUN_INFERENCE.txt").write_text(
            "Research inference only; formal export/evaluation acceptance is unchanged.\n"
            "Use the matching Zeta0-Humanoid checkout/runtime and external Qwen/processor.\n"
            f"python -m psi.deploy.real_sonic_server --weights-dir PACKAGE --qwen-model QWEN{gr00t_flag}\n"
            "The service derives current-state count, RTC capacity and time mode from this package.\n"
            f"Host architecture: {config.model.host_architecture}\n"
            f"Action grid: {config.data.action_hz:g}Hz, {config.data.action_horizon} steps, "
            f"RTC prefix up to {trained} steps.\n"
            f"Required external VLM: {config.model.qwen_model}\n"
            f"{gr00t_line}"
            f"Required encoder identity: {metadata['encoder_id']}\n"
            "For 30Hz packages, use the runtime/server supporting that grid; "
            "the legacy physical 50Hz queue/client cannot consume it directly.\n"
            "No dataset, feature tensors, training-only models or optimizer states are packaged.\n"
        )
        external_dependency = {"qwen_model": str(config.model.qwen_model), "included": False}
        if config.model.host_architecture == "gr00t_n1d7":
            external_dependency["gr00t_model"] = str(config.model.gr00t_model)
        manifest = {
            "format": INFERENCE_FORMAT, "module_names": sorted(INFERENCE_MODULES),
            "weights_file": "inference.safetensors", "weights_sha256": file_sha256(output / "inference.safetensors"),
            "tensor_count": len(tensors), "source_completed_updates": source["completed_updates"],
            "source_weights_sha256": source["weights_sha256"],
            "source_checkpoint_sha256": file_sha256(output / "source_checkpoint.json"),
            "architecture": metadata["architecture"], "temporal_abi": metadata["temporal_abi"],
            "encoder_id": metadata["encoder_id"], "processor_id": metadata["processor_id"],
            "action_protocol_id": PROTOCOL, "state_schema_id": STATE_SCHEMA,
            "visual_keys": list(CAMERAS), "task": instruction,
            "rtc_max_prefix_steps": trained, "action_condition_time_mode": config.training.action_condition_time_mode,
            "future_component_alignment": payload["data"].get("future_component_alignment"),
            "observation_contract": observation_contract,
            "validated_production_deploy_bundle": False, "evaluation_report": None,
            "external_dependency": external_dependency,
            "source_code_revision": metadata["code_revision"], "source_code_fingerprint": metadata["code_fingerprint"],
            "package_files_sha256": {
                str(path.relative_to(output)): file_sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file() and path.name != "inference.safetensors"
            },
        }
        (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (output / ".complete").write_text("complete\n")
        verify_inference_package(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def load_inference_weights(package_dir, modules):
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "inference_manifest.json").read_text())
    if set(modules) != set(manifest["module_names"]):
        raise ValueError("Provide exactly the five inference modules listed in inference_manifest.json")
    if "package_files_sha256" in manifest:
        manifest = verify_inference_package(package_dir)
    path = package_dir / manifest["weights_file"]
    if "package_files_sha256" not in manifest:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != manifest["weights_sha256"]:
            raise ValueError("Inference weight checksum mismatch")
    weights = load_file(path, device="cpu")
    expected = {f"{name}.{key}": value for name, module in modules.items() for key, value in module.state_dict().items()}
    if (set(weights) != set(expected)
            or ("tensor_count" in manifest and manifest["tensor_count"] != len(weights))):
        raise ValueError("Inference weight tensor keys/count disagree with the complete policy graph")
    # Validate every module before mutating any one of them. load_state_dict
    # otherwise permits partial loading and silently casts mismatched dtypes.
    for key, value in weights.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Inference weight tensor shape/dtype mismatch: {key}")
    for name, module in modules.items():
        prefix = name + "."
        state = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        module.load_state_dict(state, strict=True)
    return manifest
