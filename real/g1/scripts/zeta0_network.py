"""Use the delivered trained networks within the local inference entry point."""
import base64
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
from policy_profile import runtime_kit
KIT = runtime_kit()


def import_runtime():
    expected = json.loads((KIT / 'COMPATIBILITY.json').read_text())['runtime_files_sha256']
    for relative, digest in expected.items():
        path = KIT / relative
        if not path.resolve().is_relative_to(KIT.resolve()):
            raise ValueError('Runtime source path escaped the verified kit')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Runtime source checksum mismatch: ' + relative)
    runtime_root = (KIT / 'runtime').resolve()
    for name, module in list(sys.modules.items()):
        if name == 'psi' or name.startswith('psi.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(runtime_root):
                raise RuntimeError('Another model runtime is already imported: ' + name)
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))


class LocalZetaPolicy(torch.nn.Module):
    def __init__(self, weights_dir, qwen_model, device, vlm_source=None):
        super().__init__()
        import_runtime()
        from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
        self.runtime = ResearchInferenceRuntime(weights_dir)
        self.policy, self.processor, self.provenance_verification = build_local_policy(
            self.runtime, qwen_model, device, vlm_source=vlm_source)
        # Keep the original checkpoint names for the local strict loader. These
        # attributes reference the exact modules used by policy.sample().
        self.history_encoder = self.policy.history_encoder
        self.history_state_adapters = self.policy.history_state_adapters
        self.latent_transition = self.policy.latent_transition
        self.normalizer = self.policy.latent_normalizer
        self.student = self.policy.student
        self.kit_root = KIT
        self.session = 'local/' + uuid.uuid4().hex
        self.sequence = 0
        self.sampling_seed = 0
        self.last_raw_actions = None

    @torch.inference_mode()
    def predict_request(self, request, *, seed=None):
        from psi.sonic_actions import controller_command_values
        manifest = self.runtime.read_json('inference_manifest.json')
        if request['task'] != manifest['task']:
            raise ValueError('Observation task differs from the trained task')
        effective_seed = self.sampling_seed + self.sequence if seed is None else seed
        raw = self.runtime.infer_request(self.policy, self.processor, request, seed=effective_seed)
        self.sequence += 1
        self.last_raw_actions = raw.copy()
        return controller_command_values(raw, projection=(self.runtime.observation_contract or {}).get('action_projection'))

    def predict(self, observation, task):
        """Preserve real local receive times; never invent zero-age sensor data."""
        from psi.deploy.real_sonic_protocol import FORMAT, PROTOCOL, STATE_SCHEMA
        timing = observation.get('observation.timing')
        if timing is None:
            raise ValueError('Live/local observations require genuine camera/body/hand timing metadata')
        anchor = float(timing['decision_monotonic'])
        body_time = float(timing['body']['received_monotonic']) - anchor
        hands = [float(t) - anchor for t in timing['hand_state']['channel_received_monotonic']]
        values = np.concatenate((observation['observation.state'], observation['observation.hand_q']))
        images = {}
        for short in ('ego_view', 'left_wrist', 'right_wrist'):
            output = BytesIO()
            Image.fromarray(observation['observation.images.' + short]).save(output, format='PNG')
            images['observation.images.' + short] = dict(
                encoded=base64.b64encode(output.getvalue()).decode('ascii'),
                time=float(timing['cameras'][short]['received_monotonic']) - anchor)
        request = dict(format=FORMAT, action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
            session_id=self.session, request_seq=self.sequence, queue_revision=0, anchor_time=anchor,
            task=task, state=dict(values=[values.tolist()], valid=[[True] * 41], times=[max(body_time, *hands)]),
            images=images, committed_prefix=[], snapshot_timing=dict(
                camera_anchor_time=float(timing['anchor_monotonic']) - anchor,
                body_time=body_time, hand_times=hands))
        return self.predict_request(request)

    def recorded_trajectory(self, path):
        module_name = '_local_zeta_recorded_trajectory'
        spec = importlib.util.spec_from_file_location(module_name, KIT / 'offline_replay.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RecordedTrajectory(path, self.runtime)


def build_policy(manifest, qwen_model, device, weights_dir, vlm_source=None):
    return LocalZetaPolicy(weights_dir, qwen_model, device, vlm_source=vlm_source)


def build_local_policy(runtime, qwen_model, device, vlm_source=None):
    """Keep the trained graph and files; audit the local CUDA build explicitly."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
    from psi.data.predictor_pretrain.runtime import fingerprint_model_directory
    from psi.models.predictive_world.observation import FrozenQwenObservationEncoder
    from psi.models.predictive_world.policy import PredictiveWorldPolicy
    from local_vla_compat import (
        processor_identity, require_preprocessing_runtime, validate_preprocessing_migration,
    )

    processor = AutoProcessor.from_pretrained(str(qwen_model), local_files_only=True)
    fingerprint = build_action_aligned_vlm_processor_fingerprint(processor)
    provenance = runtime.read_json('feature_provenance.json')
    if (provenance.get('image_preprocessing') != 'aspect_v1'
            or provenance.get('feature_layer') != 'post_merger_content_area_overlap_4x4_v1'
            or provenance.get('dtype') != 'float16' or provenance.get('feature_shape') != [16, 2048]):
        raise ValueError('The model visual-feature contract is unsupported')
    encoder_id = 'qwen3vl:' + fingerprint_model_directory(qwen_model)
    if vlm_source == 'psi0_tidy_scene_200_exact_encoder':
        from clean_table_vlm import verify
        verification = verify(provenance, fingerprint, encoder_id, KIT,
            evidence_directory=ROOT/'reports/tidy_scene_200-step15000-preparation')
    elif vlm_source == 'zeta0_tidy_scene_exact_encoder':
        from clean_table_vlm import verify
        verification = verify(provenance, fingerprint, encoder_id, KIT,
            evidence_directory=ROOT/'reports/tidy_scene-step15000-preparation')
    elif vlm_source == 'psi0_blow_dry_hair_exact_encoder':
        from clean_table_vlm import verify
        verification = verify(provenance, fingerprint, encoder_id, KIT,
            evidence_directory=ROOT/'reports/blow_dry_hair-20261029-model')
    elif vlm_source == 'psi0_place_chair_exact_encoder':
        from clean_table_vlm import verify
        verification = verify(provenance, fingerprint, encoder_id, KIT,
            evidence_directory=ROOT/'reports/place-chair-20261029-model')
    elif vlm_source in ('psi0_clean_table_exact_encoder', 'psi0_place_tape_on_table_exact_encoder'):
        from clean_table_vlm import verify
        # The tape checkpoint declares the identical encoder, processor and
        # visual-feature contract; verify against the same runtime-bound evidence.
        verification = verify(provenance, fingerprint, encoder_id, KIT)
    elif vlm_source is not None:
        from psi0_vlm_source import verify_psi0_vlm
        verification = verify_psi0_vlm(qwen_model, vlm_source, provenance, fingerprint, encoder_id)
    elif processor_identity(fingerprint) == provenance['processor_id']:
        if encoder_id != provenance['encoder_id']:
            raise ValueError('Qwen encoder file identity mismatch')
        verification = dict(mode='exact', encoder_id=encoder_id, processor_id=provenance['processor_id'])
    else:
        logs = ROOT / 'logs'
        names = ['zeta0-cuda-build-migration.json', 'zeta0-preprocessing-reference.json',
                 'zeta0-preprocessing-local.json']
        report, source, target = [json.loads((logs / name).read_text()) for name in names]
        for name, key in [(names[1], 'reference_report_sha256'), (names[2], 'target_report_sha256')]:
            if hashlib.sha256((logs / name).read_bytes()).hexdigest() != report.get(key):
                raise ValueError('Preprocessing comparison evidence was modified')
        digest = report.get('kit_compatibility_sha256')
        require_preprocessing_runtime(KIT, digest)
        verification = validate_preprocessing_migration(
            provenance, fingerprint, encoder_id, digest, report, source, target)
    modules = runtime.build_modules()
    target_device = torch.device(device)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(str(qwen_model), local_files_only=True,
        dtype=torch.bfloat16 if target_device.type == 'cuda' else torch.float32, attn_implementation='sdpa')
    expected_dim = provenance['feature_shape'][-1]
    if qwen.config.text_config.hidden_size != expected_dim:
        raise ValueError('Qwen hidden dimension differs from the trained model')
    policy = PredictiveWorldPolicy(observation_encoder=FrozenQwenObservationEncoder(qwen),
        history_state_adapters=modules['history_state_adapters'], history_encoder=modules['history_encoder'],
        latent_transition=modules['latent_transition'], latent_normalizer=modules['normalizer'],
        student=modules['student'], action_solver_steps=runtime.read_json('inference_manifest.json')['architecture']['action_solver_steps'])
    return policy.to(target_device).eval(), processor, verification
