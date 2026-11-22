"""Portable inference from a source-bound real-SONIC research package."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from psi.deploy.research_inference_weights import load_inference_weights, verify_inference_package
from psi.deploy.real_sonic_protocol import MAX_RTC_PREFIX, validate_request
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.contracts import NativeActionBatch
from dataclasses import replace
from io import BytesIO
import base64
from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.contracts import ObservedHistory
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import LatentNormalizer, TemporalLatentTransition
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.observation import FrozenQwenObservationEncoder
from psi.models.predictive_world.policy import PredictiveWorldPolicy
from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
from psi.data.predictive_world.provenance import verify_qwen_provenance
from psi.real_observation_contract import make_deployment_observation_contract

class ResearchInferenceRuntime:
    """Read the immutable research weight package; do not relabel it as an export."""
    def __init__(self, package_dir):
        self.root = Path(package_dir).expanduser().resolve()
        manifest = self.read_json("inference_manifest.json")
        if "package_files_sha256" in manifest:
            # Validate small packaged metadata before constructing H/F/G. The
            # subsequent strict loader also verifies the large weight bytes.
            manifest = verify_inference_package(self.root, verify_weights=False)
        if (manifest["format"] != "oraclevla.research_inference_weights.v1"
                or manifest["action_protocol_id"] != "real_sonic_motion64_brainco12_v1"
                or manifest["state_schema_id"] != "real_g1_q29_brainco12_v1"):
            raise ValueError("Requires the matching recorded G1/BrainCo research package")
        temporal = manifest["temporal_abi"]
        if ((temporal["action_hz"], temporal["action_horizon"], temporal["state_query_hz"]) != (50, 30, 50)
                or temporal["state_history_count"] not in (1, 21)):
            raise ValueError("Unsupported temporal ABI")
        # Advertise only the prefix range actually trained by this source.
        source_path = self.root / "source_checkpoint.json"
        if file_sha256(source_path) != manifest["source_checkpoint_sha256"]:
            raise ValueError("RTC source checkpoint checksum mismatch")
        source = self.read_json("source_checkpoint.json")
        config = self.read_json("original_run_config.json")
        metadata = source["metadata"]
        if canonical_sha256(config) != metadata["resolved_config_sha256"]:
            raise ValueError("RTC source config checksum mismatch")
        trained = metadata["optimization_contract"].get("rtc_training_max_delay")
        if (type(trained) is not int or not 0 <= trained < temporal["action_horizon"]
                or trained != config["training"].get("rtc_training_max_delay")):
            raise ValueError("RTC capacity must match the recorded training contract")
        self.rtc_max_prefix_steps = min(trained, MAX_RTC_PREFIX)
        self.action_condition_time_mode = config["training"].get("action_condition_time_mode")
        if self.action_condition_time_mode not in (None, "fixed_action_grid_v1"):
            raise ValueError("Unsupported source action-condition time mode")
        if metadata["optimization_contract"].get("action_condition_time_mode") != self.action_condition_time_mode:
            raise ValueError("Source checkpoint/config action-condition time mode mismatch")
        self.observation_contract = make_deployment_observation_contract(config.get("data", {}))
        if self.observation_contract is not None:
            if (manifest.get("observation_contract") != self.observation_contract
                    or metadata.get("temporal_abi") != temporal
                    or any(temporal.get(key) != config["data"].get(key) for key in (
                        "real_observation_mode", "real_observation_contract", "real_action_label_mode", "real_action_projection",
                        "real_snapshot_max_age_s"))):
                raise ValueError("Source checkpoint/config/manifest observation contract mismatch")
            if (config["data"].get("real_action_label_mode") != "held_command_v1"
                    or self.action_condition_time_mode != "fixed_action_grid_v1"):
                raise ValueError("Collector snapshots require source-bound held-command labels and fixed action slots")
        elif manifest.get("observation_contract") is not None or temporal.get("real_observation_mode") is not None:
            raise ValueError("Manifest cannot relabel a legacy source as collector snapshots")


    def read_json(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def build_modules(self):
        manifest = self.read_json("inference_manifest.json")
        a, t = manifest["architecture"], manifest["temporal_abi"]
        if t["observed_visual_mode"] != "current_multiview":
            raise ValueError("This entry point requires the packaged current-multiview ABI")
        protocols = self.read_json("registries/action_protocol_registry.json")
        states = self.read_json("registries/state_schema_registry.json")
        dims = {key: int(value["dim"]) for key, value in states.items()}
        stats = {key: (torch.tensor(value["mean"], dtype=torch.float32),
                       torch.tensor(value["std"], dtype=torch.float32)) for key, value in states.items()}
        qwen_dim = self.read_json("feature_provenance.json")["feature_shape"][-1]
        if a.get("goal_intent_enabled", False):
            raise ValueError("Goal-intent deployment is retired; use its original source snapshot")
        history = PredictiveHistoryEncoder(qwen_dim=qwen_dim,
            state_history_count=t["state_history_count"], output_tokens=16,
            context_tokens=a["history_context_tokens"], hidden_dim=a["history_hidden_dim"],
            num_blocks=a["history_blocks"], num_heads=a["history_heads"], ff_dim=a["history_ff_dim"])
        transition = TemporalLatentTransition(protocols, context_dim=history.hidden_dim,
            latent_shape=tuple(a["latent_shape"]), group_ends_s=t["future_group_ends_s"],
            **{key: a["latent_transition"][key] for key in ("hidden_dim", "num_blocks", "num_heads", "ff_dim")})
        student = TemporalNativeActionExpert(protocols, dims, state_statistics=stats,
            qwen_dim=qwen_dim, context_tokens=a["expert_context_tokens"],
            latent_dim=a["latent_shape"][2], latent_groups=a["latent_shape"][0],
            latent_slots=a["latent_shape"][1], state_history_count=t["state_history_count"],
            action_steps=t["action_horizon"], hidden_dim=a["expert_hidden_dim"],
            num_blocks=a["expert_blocks"], num_heads=a["expert_heads"], ff_dim=a["expert_ff_dim"])
        modules = {"history_state_adapters": StateAdapterBank(dims, statistics=stats),
                   "history_encoder": history, "latent_transition": transition,
                   "normalizer": LatentNormalizer(tuple(a["latent_shape"])), "student": student}
        load_inference_weights(self.root, modules)
        for module in modules.values():
            module.eval()
        return modules

    def build_policy(self, qwen_model, device="cuda:0"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        # Relocation is allowed; changing model/processor bytes is not. Keep the
        # archived training config untouched and verify the selected local copy.
        qwen_model = Path(qwen_model).expanduser().resolve()
        if not qwen_model.is_dir():
            raise ValueError(f"Qwen model directory does not exist: {qwen_model}")
        device = torch.device(device)
        processor = AutoProcessor.from_pretrained(str(qwen_model), local_files_only=True)
        verify_qwen_provenance(qwen_model, self.read_json("feature_provenance.json"), processor=processor)
        modules = self.build_modules()
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(str(qwen_model),
            local_files_only=True, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa")
        expected = self.read_json("feature_provenance.json")["feature_shape"][-1]
        if qwen.config.text_config.hidden_size != expected:
            raise ValueError("Qwen hidden dimension differs from the packaged model")
        policy = PredictiveWorldPolicy(observation_encoder=FrozenQwenObservationEncoder(qwen),
            history_state_adapters=modules["history_state_adapters"],
            history_encoder=modules["history_encoder"], latent_transition=modules["latent_transition"],
            latent_normalizer=modules["normalizer"], student=modules["student"],
            action_solver_steps=self.read_json("inference_manifest.json")["architecture"]["action_solver_steps"])
        return policy.to(device).eval(), processor

    def prepare_observation(self, request, processor, *, base_dir=".", device):
        """Input times are genuine receive times relative to the observation anchor."""
        manifest = self.read_json("inference_manifest.json")
        t = manifest["temporal_abi"]
        def resolve(name):
            return Path(base_dir) / name
        prefix = None
        if self.observation_contract is not None and "state" not in request:
            raise ValueError("Collector snapshots require the source-bound wire state and snapshot timing")
        if "state" in request:
            values, valid, times, prefix = validate_request(
                request, state_count=t["state_history_count"], rtc_max_prefix_steps=self.rtc_max_prefix_steps,
                observation_contract=self.observation_contract)
        else:
            with np.load(resolve(request["state_npz"]), allow_pickle=False) as data:
                values = np.array(data["values"], dtype=np.float32)
                valid = np.array(data["valid"], dtype=bool)
                times = np.array(data["times"], dtype=np.float64)
        if values.shape != (t["state_history_count"], 41) or valid.shape != values.shape or times.shape != (len(values),):
            raise ValueError(f"State requires values/valid [{t['state_history_count']},41] and matching times")
        present = valid.any(axis=-1)
        queries = np.arange(1 - len(values), 1, dtype=np.float64) / t["state_query_hz"]
        age = queries - times
        max_age = self.read_json("original_run_config.json")["data"]["state_max_age_s"]
        if (not present[-1] or not np.isfinite(times[present]).all()
                or (age[present] < -1e-6).any() or (age[present] > max_age + 1e-6).any()
                or not np.isfinite(values[valid]).all()):
            raise ValueError(f"Valid states must be finite, causal and within the trained {max_age * 1000:g}ms age limit")
        values = np.where(valid, values, 0).astype(np.float32)
        times = np.where(present, times, 0).astype(np.float32)
        camera_items = request["images"]
        primary_camera = manifest["visual_keys"][0]
        if primary_camera not in camera_items or set(camera_items) - set(manifest["visual_keys"]):
            raise ValueError("Images require ego_view and only the three trained camera keys")
        images, content, primary = [], [], None
        provenance = self.read_json("feature_provenance.json")
        visual_age = self.read_json("original_run_config.json")["data"]["visual_max_age_s"]
        for camera, item in sorted(camera_items.items()):
            time = float(item["time"])
            if not np.isfinite(time) or time > 1e-6 or time < -visual_age - 1e-6:
                raise ValueError(f"Image times must be causal and within the trained {visual_age * 1000:g}ms age limit")
            source = (BytesIO(base64.b64decode(item["encoded"], validate=True))
                      if "encoded" in item else resolve(item["path"]))
            with Image.open(source) as raw:
                if raw.size != (640, 480):
                    raise ValueError("Real camera must match recorded 640x480 geometry")
                image = preprocess_endpoint_vlm_image(raw.convert("RGB"), image_preprocessing=provenance["image_preprocessing"])
            if camera == primary_camera:
                primary = len(images)
            images.append(image)
            content.extend(({"type": "text", "text": f"Camera: {camera}; observation time: {time:+.3f} seconds."},
                            {"type": "image", "image": image}))
        content.append({"type": "text", "text": request.get("task", manifest["task"])})
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        encoded = dict(processor(text=[text], images=images, padding=True, return_tensors="pt"))
        encoded["image_counts"] = torch.tensor([len(images)])
        encoded["primary_current_image_indices"] = torch.tensor([primary])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        protocol = manifest["action_protocol_id"]
        space = self.read_json("registries/action_protocol_registry.json")[protocol]["normalization_space"]
        observed = ObservedHistory(qwen_inputs=encoded,
            raw_states=torch.from_numpy(values).unsqueeze(0).to(device),
            state_valid=torch.from_numpy(valid).unsqueeze(0).to(device),
            state_times=torch.from_numpy(times).unsqueeze(0).to(device),
            state_schema_ids=(manifest["state_schema_id"],), action_protocol_ids=(protocol,),
            action_native_dims=torch.tensor([76], device=device), action_normalization_spaces=(space,))
        if prefix is not None and len(prefix):
            pv = torch.zeros((1, 30, 76), device=device, dtype=torch.float32)
            pm = torch.zeros_like(pv, dtype=torch.bool)
            pt = torch.zeros((1, 30), device=device, dtype=torch.float32)
            pv[0, :len(prefix)] = torch.as_tensor(prefix, device=device)
            pm[0, :len(prefix)] = True
            if self.action_condition_time_mode == "fixed_action_grid_v1":
                pt[0, :len(prefix)] = (torch.arange(1, len(prefix)+1, device=device, dtype=torch.float64) / 50).float()
            else:
                pt[0, :len(prefix)] = torch.arange(1, len(prefix)+1, device=device) / 50.0
            observed = replace(observed, committed_actions=NativeActionBatch(
                pv, pm, pt, (protocol,), torch.tensor([76], device=device), (space,)))
        observed.validate(batch_size=1, state_count=t["state_history_count"], action_horizon=t["action_horizon"])
        return observed

    def infer_request(self, policy, processor, request, *, base_dir=".", seed=0):
        device = next(policy.parameters()).device
        observed = self.prepare_observation(request, processor, base_dir=base_dir, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, 30, 76, device=device, generator=generator)
        actions = policy.sample(observed, action_noise=noise)[0, :, :76].float().cpu().numpy()
        if actions.shape != (30, 76) or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid native actions")
        return actions
