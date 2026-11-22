import json
import sqlite3
import shutil
from dataclasses import asdict, replace
from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from psi.config.predictive_world import (
    PredictiveWorldConfig, PredictiveWorldDataConfig, PredictiveWorldModelConfig,
    PredictiveWorldRuntimeConfig, PredictiveWorldTrainingConfig,
    predictive_world_architecture_contract,
)
from psi.data.predictive_world.cache import FEATURE_FORMAT, canonical_sha256, file_sha256
from psi.data.predictive_world.index import (
    INDEX_BINDING_FORMAT, INDEX_FORMAT, LOOKUP_FORMAT, temporal_abi_from_data_config,
)
from psi.deploy.real_sonic_protocol import CAMERAS, PROTOCOL, STATE_SCHEMA
from psi.deploy.research_inference_weights import (
    INFERENCE_MODULES, export_real_sonic_research_policy,
    load_inference_weights, verify_inference_package,
)
from psi.models.predictive_world.checkpoint import PHASE_MODULES, save_predictive_world_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def source(tmp_path, request):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    values = {
        "action_protocol_registry": {PROTOCOL: dict(dim=76, normalization_space="native", mean=[0.] * 76, std=[1.] * 76)},
        "state_schema_registry": {STATE_SCHEMA: dict(dim=41, mean=[0.] * 41, std=[1.] * 41)},
        "measurement_registry": {"measured": dict(width=41)},
        "capability_registry": {"real/doll": dict(action_protocol_id=PROTOCOL, visual_keys=list(CAMERAS))},
    }
    paths = {name: artifacts / (name + ".json") for name in values}
    for name, value in values.items():
        write_json(paths[name], value)
    observation = {}
    if getattr(request, "param", None) in ("collector", "collector_projection"):
        from test_observation_contract_transition import collector_contract
        observation = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=collector_contract(),
            real_action_label_mode="held_command_v1", state_max_age_s=.2, visual_max_age_s=.2,
            future_measurement_max_age_s=.04, future_visual_max_age_s=.08)
        if request.param == "collector_projection":
            from psi.real_observation_contract import build_recorded_action_projection
            observation["real_action_projection"] = build_recorded_action_projection(-.75, .6875, "a" * 64)
    config = PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(
            observed_visual_mode="current_multiview", state_history_count=1,
            future_component_alignment="independent_components_v1",
            **observation,
            action_protocol_registry=str(paths["action_protocol_registry"]),
            state_schema_registry=str(paths["state_schema_registry"]),
            measurement_registry=str(paths["measurement_registry"]),
            canonical_future_capabilities=str(paths["capability_registry"]),
            index_path=str(artifacts / "index.jsonl"), feature_cache=str(artifacts / "features"),
        ),
        model=PredictiveWorldModelConfig(qwen_model="external-qwen", measurement_dim=29),
        training=PredictiveWorldTrainingConfig(
            phase="joint", world_model="parent-world", max_updates=20,
            rtc_training_max_delay=15, action_condition_time_mode="fixed_action_grid_v1",
            micro_batch=1, effective_global_batch=1, normalizer_calibration_updates=0,
            forecast_warmup_updates=0, forecast_ramp_updates=0,
        ),
        runtime=PredictiveWorldRuntimeConfig(nodes=1, gpus_per_node=1, output_dir=str(tmp_path / "training")),
    )
    native_modules = getattr(request, "param", None) in ("native_modules", "native_modules_30hz", "native_modules_raw")
    if getattr(request, "param", None) == "native_modules_raw":
        config = replace(config, training=replace(config.training, latent_prediction_space="raw"),
            model=replace(config.model, latent_transition=replace(config.model.latent_transition, output_affine=True)))
    if getattr(request, "param", None) == "native_modules_30hz":
        config = replace(config, data=replace(config.data, action_hz=30, action_horizon=50))
    if native_modules:
        config = replace(config, model=replace(config.model,
            history_hidden_dim=16, history_blocks=1, history_heads=2, history_ff_dim=32,
            representation_hidden_dim=16, representation_blocks=1,
            representation_heads=2, representation_ff_dim=32,
            expert_hidden_dim=16, expert_blocks=1, expert_heads=2, expert_ff_dim=32,
            latent_transition=replace(config.model.latent_transition,
                hidden_dim=16, num_blocks=1, num_heads=2, ff_dim=32)))
    payload = json.loads(json.dumps(asdict(config)))
    config_path = tmp_path / "config.json"
    write_json(config_path, payload)
    temporal = temporal_abi_from_data_config(config.data)
    index = artifacts / "index.jsonl"
    # Exactly one JSONL row; build a tiny actual lookup rather than mocking IO.
    index.write_text(json.dumps(dict(source_id="real/doll", materialization=dict(instruction="Place the doll on the table."))) + "\n")
    lookup = index.with_suffix(index.suffix + ".sqlite3")
    with sqlite3.connect(lookup) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
        connection.execute("INSERT INTO metadata VALUES('format',?)", (LOOKUP_FORMAT,))
    provenance = dict(registry_binding={name + "_sha256": file_sha256(path) for name, path in paths.items()},
        time_alignment=dict(future_component_alignment=config.data.future_component_alignment))
    provenance["time_alignment"].update({key: value for key, value in observation.items() if key.startswith("real_")})
    index_meta = dict(format=INDEX_FORMAT, rows=1, index_sha256=file_sha256(index), lookup_sha256=file_sha256(lookup),
        temporal_abi=temporal, provenance=provenance,
        semantic_identity=canonical_sha256(dict(provenance=provenance, temporal_abi=temporal)))
    write_json(index.with_suffix(index.suffix + ".meta.json"), index_meta)
    binding = dict(format=INDEX_BINDING_FORMAT, semantic_identity=index_meta["semantic_identity"],
        index_sha256=index_meta["index_sha256"], lookup_sha256=index_meta["lookup_sha256"],
        temporal_abi_sha256=canonical_sha256(temporal))
    feature_provenance = dict(encoder_id="qwen-identity", processor_id="processor-identity", feature_shape=[16, 2048],
        image_preprocessing="aspect_v1", feature_layer="post_merger_content_area_overlap_4x4_v1", dtype="float16")
    feature = dict(format=FEATURE_FORMAT, provenance=feature_provenance, cache_identity=canonical_sha256(feature_provenance),
        index_binding=binding, records=[], mapping_sha256=canonical_sha256(dict(records=[])), feature_records_sha256="a" * 64)
    feature["manifest_identity"] = canonical_sha256({key: feature[key] for key in (
        "cache_identity", "index_binding", "mapping_sha256", "feature_records_sha256")})
    feature_dir = artifacts / "features"
    feature_dir.mkdir()
    write_json(feature_dir / "feature_manifest.json", feature)
    temporal_model = dict(history_visual_offsets_s=list(config.data.history_visual_offsets_s),
        **config.data.observation_contract, state_history_count=1, state_query_hz=50.,
        future_group_ends_s=list(config.data.future_group_ends_s),
        action_horizon=config.data.action_horizon, action_hz=config.data.action_hz)
    metadata = dict(
        architecture=predictive_world_architecture_contract(config.model), temporal_abi=temporal_model,
        encoder_id=feature_provenance["encoder_id"], processor_id=feature_provenance["processor_id"],
        protocol_registry_sha256=file_sha256(paths["action_protocol_registry"]),
        measurement_registry_sha256=file_sha256(paths["measurement_registry"]), normalization_id="normalizer-hash",
        reference_id="reference-hash", resolved_config_sha256=canonical_sha256(payload), code_revision="revision",
        code_fingerprint="code-hash", environment_fingerprint="environment-hash",
        optimization_contract=dict(rtc_training_max_delay=15, action_condition_time_mode="fixed_action_grid_v1"),
        data_fingerprint=canonical_sha256(dict(index=index_meta["index_sha256"], feature_cache=feature["cache_identity"],
            capabilities=file_sha256(paths["capability_registry"]))),
        parent_world_model=dict(phase="world_model", completed_updates=15000, weights_sha256="c" * 64,
            reference_id="parent-reference", normalization_id="parent-normalizer"),
    )
    checkpoint = tmp_path / "joint-checkpoint"
    modules = {name: nn.Linear(1, 1) for name in PHASE_MODULES["joint"]}
    if native_modules:
        from psi.models.predictive_world.factory import _forecast_modules, _reference_modules

        protocols = values["action_protocol_registry"]
        dims = {STATE_SCHEMA: 41}
        stats = {STATE_SCHEMA: (torch.zeros(41), torch.ones(41))}
        modules = _reference_modules(config, protocols, dims, stats, qwen_dim=2048)
        modules.update(_forecast_modules(config, protocols, dims, stats, qwen_dim=2048))
        modules["student"] = deepcopy(modules["teacher"])
        if config.model.latent_transition.output_affine:
            modules["latent_transition"].output_scale.fill_(2)
            modules["latent_transition"].output_shift.fill_(3)
            modules["latent_transition"].output_calibrated.fill_(True)
    save_predictive_world_bundle(checkpoint, phase="joint", modules=modules, completed_updates=20, metadata=metadata)
    return dict(checkpoint=checkpoint, config=config_path, paths=paths, feature=feature_dir / "feature_manifest.json",
        output=tmp_path / "research-package", modules=modules)


@pytest.mark.parametrize("source", ["native_modules", "native_modules_30hz", "native_modules_raw"], indirect=True)
def test_deployment_constructs_and_strictly_loads_native_modules(source):
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    modules = ResearchInferenceRuntime(source["output"]).build_modules()
    assert set(modules) == INFERENCE_MODULES
    for name, module in modules.items():
        assert not module.training
        expected = source["modules"][name].state_dict()
        assert module.state_dict().keys() == expected.keys()
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("source", ["native_modules_30hz"], indirect=True)
def test_native_30hz_runtime_uses_50_step_noise_and_rtc_grid(source, tmp_path):
    import base64
    import numpy as np
    from PIL import Image
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
    from test_current_multiview import Processor
    from test_real_sonic_deployment import request

    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    runtime = ResearchInferenceRuntime(source["output"])
    req = request(prefix=np.full((15, 76), .25).tolist())
    image = tmp_path / 'frame.png'
    Image.new('RGB', (640, 480)).save(image)
    req['state'] = {'values': [[0.] * 41], 'valid': [[True] * 41], 'times': [0.]}
    req['images'] = {camera: {'encoded': base64.b64encode(image.read_bytes()).decode(), 'time': 0.} for camera in CAMERAS}
    observed = runtime.prepare_observation(req, Processor(), device='cpu')
    assert observed.committed_actions.values.shape == (1, 50, 76)
    expected = (torch.arange(1, 16, dtype=torch.float64) / 30).float()
    torch.testing.assert_close(observed.committed_actions.times[0, :15], expected, rtol=0, atol=0)
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
        def sample(self, observed, *, action_noise):
            assert action_noise.shape == (1, 50, 76)
            return action_noise
    assert runtime.infer_request(Policy(), Processor(), req).shape == (50, 76)


def test_research_export_roundtrip_matches_cpu_weights_and_real_runtime(source):
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    manifest = export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    assert manifest["temporal_abi"]["state_history_count"] == 1
    assert manifest["rtc_max_prefix_steps"] == 15
    assert manifest["action_condition_time_mode"] == "fixed_action_grid_v1"
    assert manifest["future_component_alignment"] == "independent_components_v1"
    assert manifest["validated_production_deploy_bundle"] is False and manifest["evaluation_report"] is None
    assert (source["output"] / "original_run_config.json").read_bytes() == source["config"].read_bytes()
    assert set(load_file(source["output"] / "inference.safetensors")) == {
        f"{name}.{key}" for name in INFERENCE_MODULES for key in source["modules"][name].state_dict()
    }
    target = {name: nn.Linear(1, 1) for name in INFERENCE_MODULES}
    load_inference_weights(source["output"], target)
    for name in INFERENCE_MODULES:
        for key, tensor in target[name].state_dict().items():
            torch.testing.assert_close(tensor, source["modules"][name].state_dict()[key], rtol=0, atol=0)
    runtime = ResearchInferenceRuntime(source["output"])
    assert runtime.rtc_max_prefix_steps == 15 and runtime.action_condition_time_mode == "fixed_action_grid_v1"
    with pytest.raises(FileExistsError, match="already exists"):
        export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])


@pytest.mark.parametrize("change,match", [
    ("config", "config checksum"), ("incomplete", "incomplete"), ("weights", "weight hash"),
    ("registry", "registry binding"), ("rtc", "RTC/action-time"), ("missing_module", "weight modules"),
    ("feature", "provenance identity"), ("index", "content hash"),
    ("alignment", "future_component_alignment mismatch"),
])
def test_export_rejects_inconsistent_sources_without_publishing(source, change, match):
    sidecar_path = source["checkpoint"] / "predictive_world_checkpoint.json"
    if change == "config":
        payload = json.loads(source["config"].read_text())
        payload["training"]["rtc_training_max_delay"] = 6
        write_json(source["config"], payload)
    elif change == "incomplete":
        (source["checkpoint"] / ".complete").unlink()
    elif change == "weights":
        with (source["checkpoint"] / "predictive_world.safetensors").open("ab") as stream:
            stream.write(b"bad")
    elif change == "registry":
        path = source["paths"]["state_schema_registry"]
        value = json.loads(path.read_text())
        value[STATE_SCHEMA]["mean"][0] = 1.
        write_json(path, value)
    elif change == "rtc":
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["metadata"]["optimization_contract"]["rtc_training_max_delay"] = 6
        write_json(sidecar_path, sidecar)
    elif change == "feature":
        value = json.loads(source["feature"].read_text())
        value["provenance"]["processor_id"] = "different-processor"
        write_json(source["feature"], value)
    elif change == "index":
        index = source["paths"]["state_schema_registry"].parent / "index.jsonl"
        index.write_text("{}\n")
    elif change == "alignment":
        path = source["paths"]["state_schema_registry"].parent / "index.jsonl.meta.json"
        value = json.loads(path.read_text())
        value["provenance"]["time_alignment"]["future_component_alignment"] = None
        value["semantic_identity"] = canonical_sha256({key: value[key] for key in ("provenance", "temporal_abi")})
        write_json(path, value)
    else:
        weights_path = source["checkpoint"] / "predictive_world.safetensors"
        tensors = {key: value for key, value in load_file(weights_path).items() if not key.startswith("student.")}
        save_file(tensors, weights_path)
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["weights_sha256"] = file_sha256(weights_path)
        write_json(sidecar_path, sidecar)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    assert not source["output"].exists()


def test_published_metadata_tampering_is_detected_before_weight_loading(source):
    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    path = source["output"] / "registries/state_schema_registry.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="metadata checksum"):
        load_inference_weights(source["output"], {name: nn.Linear(1, 1) for name in INFERENCE_MODULES})


def test_published_completion_marker_is_required(source):
    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    (source["output"] / ".complete").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_inference_package(source["output"])


def test_export_does_not_write_inside_immutable_checkpoint(source):
    with pytest.raises(ValueError, match="outside the immutable source"):
        export_real_sonic_research_policy(source["checkpoint"], source["config"], source["checkpoint"] / "new-package")


def test_published_metadata_hashes_cannot_omit_runtime_dependencies(source):
    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    path = source["output"] / "inference_manifest.json"
    value = json.loads(path.read_text())
    value["package_files_sha256"] = {}
    write_json(path, value)
    with pytest.raises(ValueError, match="incomplete"):
        verify_inference_package(source["output"])


def test_relocated_package_loads_without_original_training_assets(source, tmp_path):
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    portable = tmp_path / "portable/model"
    portable.parent.mkdir()
    source["output"].rename(portable)
    shutil.rmtree(source["paths"]["state_schema_registry"].parent)
    shutil.rmtree(source["checkpoint"])
    source["config"].unlink()
    runtime = ResearchInferenceRuntime(portable)
    assert runtime.rtc_max_prefix_steps == 15
    assert runtime.action_condition_time_mode == "fixed_action_grid_v1"
    modules = {name: nn.Linear(1, 1) for name in INFERENCE_MODULES}
    load_inference_weights(portable, modules)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, source["modules"][name].state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["architecture", "source_completed_updates", "rtc_max_prefix_steps"])
def test_runtime_rejects_executable_manifest_relabeling_before_model_allocation(source, field):
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    path = source["output"] / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    if field == "architecture":
        manifest[field]["action_solver_steps"] += 1
    else:
        manifest[field] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="manifest/source"):
        ResearchInferenceRuntime(source["output"])


@pytest.mark.parametrize("change", ["unknown_module", "shape", "dtype"])
def test_weight_preflight_rejects_entire_graph_before_copying_any_module(source, change):
    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    path = source["output"] / "inference.safetensors"
    weights = load_file(path)
    if change == "unknown_module":
        weights["teacher.unexpected"] = torch.zeros(1)
    elif change == "shape":
        weights["student.weight"] = torch.zeros(2, 1)
    else:
        weights["student.weight"] = weights["student.weight"].double()
    save_file(weights, path)
    manifest_path = source["output"] / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(weights_sha256=file_sha256(path), tensor_count=len(weights))
    write_json(manifest_path, manifest)
    target = {name: nn.Linear(1, 1) for name in INFERENCE_MODULES}
    before = {name: {key: value.clone() for key, value in module.state_dict().items()}
              for name, module in target.items()}
    with pytest.raises(ValueError, match="tensor keys/count|tensor shape/dtype"):
        load_inference_weights(source["output"], target)
    for name, module in target.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)


def test_qwen_relative_override_preserves_actual_file_and_processor_provenance(tmp_path, monkeypatch):
    from transformers import AutoProcessor
    from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
    from psi.data.predictor_pretrain.runtime import fingerprint_model_directory
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    class ImageProcessor:
        def to_json_string(self):
            return '{"size": 640}'

    class Processor:
        image_processor = ImageProcessor()
        chat_template = "fixture template"

    processor = Processor()
    qwen = tmp_path / "qwen3vl"
    qwen.mkdir()
    (qwen / "config.json").write_text('{"model_type":"qwen3_vl"}')
    save_file({"tiny_test_weight": torch.ones(1)}, qwen / "model.safetensors")
    model = tmp_path / "model"
    model.mkdir()
    provenance = dict(encoder_id="qwen3vl:" + fingerprint_model_directory(qwen),
        processor_id="qwen3vl-processor:" + canonical_sha256(build_action_aligned_vlm_processor_fingerprint(processor)),
        image_preprocessing="aspect_v1", feature_layer="post_merger_content_area_overlap_4x4_v1",
        dtype="float16", feature_shape=[16, 2048])
    write_json(model / "feature_provenance.json", provenance)
    runtime = ResearchInferenceRuntime.__new__(ResearchInferenceRuntime)
    runtime.root = model
    seen = []

    def load_processor(path, **kwargs):
        seen.append(path)
        assert kwargs == {"local_files_only": True}
        return processor

    def stop_before_allocating_policy():
        raise RuntimeError("verified before constructing policy or loading Qwen")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(runtime, "build_modules", stop_before_allocating_policy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="verified before constructing"):
        runtime.build_policy("qwen3vl", device="cpu")
    assert seen == [str(qwen)]
    (qwen / "config.json").write_text('{"model_type":"different_model"}')
    with pytest.raises(ValueError, match="encoder identity mismatch"):
        runtime.build_policy("qwen3vl", device="cpu")


@pytest.mark.parametrize("source", ["collector", "collector_projection"], indirect=True)
def test_collector_contract_is_source_bound_through_export_runtime_and_health(source):
    import numpy as np
    from test_real_sonic_deployment import request
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
    from psi.deploy.real_sonic_server import InferenceService

    manifest = export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    runtime = ResearchInferenceRuntime(source["output"])
    assert runtime.observation_contract["state_max_age_s"] == .2
    assert runtime.observation_contract == manifest["observation_contract"]
    service = InferenceService(lambda data, seed: np.zeros((30, 76)), manifest, rtc_max_prefix_steps=15,
        observation_contract=runtime.observation_contract)
    assert service.health()["observation_contract"] == runtime.observation_contract
    projection = runtime.observation_contract.get("action_projection")
    if projection is not None:
        assert projection["motion_token_min"] == -.75 and projection["motion_token_max"] == .6875
        assert projection == json.loads(source["config"].read_text())["data"]["real_action_projection"]
    req = request()
    req["state"] = dict(values=[[0.] * 41], valid=[[True] * 41], times=[-.115])
    for camera, time in zip(CAMERAS, [-.1, -.12, -.105]):
        req["images"][camera]["time"] = time
    req["snapshot_timing"] = dict(camera_anchor_time=-.1, body_time=-.12, hand_times=[-.115, -.115])
    service.reset(dict(session_id="test"))
    assert np.asarray(service.predict(req)["actions"]).shape == (30, 76)
    altered = json.loads(json.dumps(runtime.observation_contract))
    altered["state_max_age_s"] = .3
    with pytest.raises(ValueError, match="source package manifest"):
        InferenceService(service.infer, manifest, observation_contract=altered)
    path = source["output"] / "inference_manifest.json"
    manifest["observation_contract"] = altered
    write_json(path, manifest)
    with pytest.raises(ValueError, match="observation contract mismatch"):
        ResearchInferenceRuntime(source["output"])


@pytest.mark.parametrize("source", ["collector_projection"], indirect=True)
def test_portable_runtime_prepares_real_jpegs_current_state_and_rtc15(source):
    import base64
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from test_real_sonic_deployment import request
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime

    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    runtime = ResearchInferenceRuntime(source["output"])
    req = request(prefix=np.zeros((15, 76), np.float32).tolist())
    values = np.arange(41, dtype=np.float32) / 41
    req["state"] = dict(values=[values.tolist()], valid=[[True] * 41], times=[-.115])
    for ordinal, (camera, time) in enumerate(zip(CAMERAS, [-.1, -.12, -.105])):
        buffer = BytesIO()
        Image.new("RGB", (640, 480), (10 + ordinal, 20, 30)).save(buffer, format="JPEG")
        req["images"][camera] = dict(time=time, encoded=base64.b64encode(buffer.getvalue()).decode())
    req["snapshot_timing"] = dict(camera_anchor_time=-.1, body_time=-.12, hand_times=[-.115, -.115])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "encoded text"

        def __call__(self, *, text, images, **kwargs):
            assert len(images) == 3
            assert all(image.size == (320, 256) for image in images)
            return dict(input_ids=torch.ones((1, 2), dtype=torch.long),
                attention_mask=torch.ones((1, 2), dtype=torch.long), pixel_values=torch.zeros((3, 3)),
                image_grid_thw=torch.tensor([[1, 16, 20]] * 3))

    processor = Processor()
    observed = runtime.prepare_observation(req, processor, device="cpu")
    assert observed.raw_states.shape == (1, 1, 41)
    torch.testing.assert_close(observed.raw_states[0, 0], torch.tensor(values), rtol=0, atol=0)
    assert observed.qwen_inputs["image_counts"].tolist() == [3]
    assert observed.qwen_inputs["primary_current_image_indices"].tolist() == [0]
    prefix = observed.committed_actions
    assert prefix.values.shape == (1, 30, 76)
    assert prefix.valid[0, :15].all() and not prefix.valid[0, 15:].any()
    torch.testing.assert_close(prefix.times[0, :15], (torch.arange(1, 16, dtype=torch.float64) / 50).float(),
                               rtol=0, atol=0)
    text = [item["text"] for item in processor.messages[0]["content"] if item["type"] == "text"]
    assert [item.split(";")[0] for item in text[:3]] == [f"Camera: {key}" for key in CAMERAS]


@pytest.mark.parametrize("source", ["collector_projection"], indirect=True)
def test_recorded_projection_metadata_cannot_disagree_with_the_training_index(source):
    path = source["paths"]["state_schema_registry"].parent / "index.jsonl.meta.json"
    value = json.loads(path.read_text())
    value["provenance"]["time_alignment"]["real_action_projection"]["motion_token_min"] = -.625
    value["semantic_identity"] = canonical_sha256({key: value[key] for key in ("provenance", "temporal_abi")})
    write_json(path, value)
    with pytest.raises(ValueError, match="real_action_projection mismatch"):
        export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
