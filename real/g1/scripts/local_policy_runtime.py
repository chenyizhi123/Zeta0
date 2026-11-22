"""Run a trained network inside the local observation/action pipeline.

Model definitions are regular local Python files, independent of any policy
server. They export build_policy(manifest, qwen_model, device), returning a
torch.nn.Module with the named inference submodules and a predict(observation,
task=...) method. Predictions use physical SONIC/BrainCo action coordinates.

Every packaged weight is loaded strictly into its named submodule. A definition
cannot silently omit a trained module or fall back to uninitialized parameters.
This runner saves predictions; it never opens a robot command publisher.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

from local_vla_adapter import CAMERAS, ROOT, LocalVLAAdapter, finite_array


class LocalPolicy:
    def __init__(self, weights_dir, qwen_model, definition, device='cuda', vlm_source=None):
        self.adapter = LocalVLAAdapter(weights_dir)
        self.definition = Path(definition).resolve()
        if not self.definition.is_file():
            raise FileNotFoundError('Model network definition is missing: ' + str(self.definition))
        manifest = self.adapter.manifest
        weights = self.adapter.weights_dir / manifest['weights_file']
        if weights.resolve().parent != self.adapter.weights_dir:
            raise ValueError('Weights must be inside the model directory')
        digest = hashlib.sha256()
        with weights.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get('weights_sha256'):
            raise ValueError('Model weight SHA256 differs from the model manifest')
        definition_digest = hashlib.sha256(self.definition.read_bytes()).hexdigest()
        module_name = 'local_policy_definition_' + definition_digest[:16]
        spec = importlib.util.spec_from_file_location(module_name, self.definition)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            factory = getattr(module, 'build_policy', None)
            if not callable(factory):
                raise TypeError('The local network file must export build_policy(manifest, qwen_model, device)')
            arguments = dict(manifest=copy.deepcopy(manifest), qwen_model=str(qwen_model), device=device)
            if 'weights_dir' in inspect.signature(factory).parameters:
                arguments['weights_dir'] = str(self.adapter.weights_dir)
            if vlm_source is not None:
                if 'vlm_source' not in inspect.signature(factory).parameters:
                    raise ValueError('The selected network does not support an explicit VLM source')
                arguments['vlm_source'] = vlm_source
            self.model = factory(**arguments)
            if not isinstance(self.model, torch.nn.Module) or not callable(getattr(self.model, 'predict', None)):
                raise TypeError('build_policy must return a torch.nn.Module with predict(observation, task=...)')
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        state = load_file(str(weights), device='cpu')
        if len(state) != manifest.get('tensor_count'):
            raise ValueError('Stored tensor count differs from the model manifest')
        consumed = set()
        names = manifest.get('module_names')
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or not all(isinstance(name, str) and name for name in names)):
            raise ValueError('Model inference module names are missing or invalid')
        for name in names:
            target = getattr(self.model, name, None)
            if not isinstance(target, torch.nn.Module):
                raise TypeError('Local network is missing trained module: ' + name)
            prefix = name + '.'
            selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not selected:
                raise ValueError('No packaged weights found for module: ' + name)
            target.load_state_dict(selected, strict=True)
            consumed.update(prefix + k for k in selected)
        if consumed != set(state):
            raise ValueError('Unused model weights: ' + ', '.join(sorted(set(state) - consumed)[:10]))
        self.model.to(device).eval()
        self.identity = dict(weights_sha256=digest.hexdigest(), qwen_model=str(Path(qwen_model).resolve()),
                             definition=str(self.definition),
                             definition_sha256=definition_digest, tensors_loaded=len(consumed), device=device)
        if hasattr(self.model, 'provenance_verification'):
            self.identity['provenance_verification'] = self.model.provenance_verification

    @torch.inference_mode()
    def predict(self, observation):
        # Make the input contract identical for stored observations and live
        # ObservationStream snapshots, and isolate the capture cache from models.
        body = finite_array(observation['observation.state'], (29,), 'body feedback')
        hands = finite_array(observation['observation.hand_q'], (12,), 'actual hand feedback')
        if np.any((hands < 0) | (hands > 1)):
            raise ValueError('Actual BrainCo feedback must be in 0..1')
        model_input = {'observation.state': body, 'observation.hand_q': hands}
        if 'observation.timing' in observation:
            model_input['observation.timing'] = copy.deepcopy(observation['observation.timing'])
        for camera in CAMERAS:
            key = 'observation.images.' + camera
            value = np.asarray(observation[key])
            if value.dtype != np.uint8 or value.shape != (480, 640, 3):
                raise ValueError(key + ' must be a 640x480 uint8 RGB image')
            model_input[key] = value.copy()
        prediction = self.model.predict(model_input, task=self.adapter.manifest['task'])
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().float().cpu().numpy()
        return self.adapter.action_chunk(prediction)


def main():
    parser = argparse.ArgumentParser(description='本地模型推理；保存预测，不启控或发送机器人动作')
    from policy_profile import TASKS
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--model-definition', type=Path, default=ROOT / 'scripts/zeta0_network.py',
                        help='默认使用已校验的 Zeta0 网络；也可指定本地 build_policy 定义')
    parser.add_argument('--model-paths', type=Path, default=None)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--observation', type=Path, help='离线观测 .npz，包含三路 RGB、body29 和 hand12')
    source.add_argument('--live', action='store_true', help='只读订阅现有本地 ObservationStream')
    source.add_argument('--trajectory', nargs='?', const='auto', help='回放真实轨迹；不填路径时选对应任务')
    parser.add_argument('--max-frames', type=int, default=3, help='轨迹中均匀抽取的帧数；0 表示全部')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0,
                        help='采样基础种子；轨迹回放按基础种子加原始帧号，与原版回放一致')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--execution-contract', type=Path,
                        help='可选：按已记录的本地执行契约同时生成 token/双手目标；仍不发送动作')
    parser.add_argument('--output', required=True, type=Path, help='保存 actions 和模型身份的 .npz')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    from policy_profile import model_paths
    paths = model_paths(args.task, args.model_paths)
    policy = LocalPolicy(paths['tasks'][args.task]['weights_dir'], paths['qwen_model'],
                         args.model_definition, device=args.device, vlm_source=paths.get('vlm_source'))
    if hasattr(policy.model, 'sampling_seed'):
        policy.model.sampling_seed = args.seed
    predictions, raw_predictions, durations, observations_timing = [], [], [], []
    trajectory_info = None
    if args.trajectory:
        if not callable(getattr(policy.model, 'recorded_trajectory', None)):
            parser.error('The selected network does not implement recorded trajectory loading')
        path = (policy.model.kit_root / 'trajectories' / args.task
                if args.trajectory == 'auto' else Path(args.trajectory))
        trajectory = policy.model.recorded_trajectory(path)
        try:
            indices = np.arange(trajectory.spec['frames'])
            if args.max_frames and len(indices) > args.max_frames:
                indices = indices[np.linspace(0, len(indices) - 1, args.max_frames, dtype=int)]
            for index in indices:
                request = trajectory.request(int(index))
                started = time.monotonic()
                prediction = policy.model.predict_request(request, seed=args.seed + int(index))
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.detach().float().cpu().numpy()
                predictions.append(policy.adapter.action_chunk(prediction))
                raw_predictions.append(policy.model.last_raw_actions.copy())
                durations.append(time.monotonic() - started)
                print(json.dumps(dict(event='local_model_prediction', task=args.task, frame=int(index),
                                      seconds=durations[-1])), file=sys.stderr, flush=True)
            trajectory_info = dict(directory=str(path), complete_frames=trajectory.spec['frames'],
                                   selected_frame_indices=indices.tolist(),
                                   action_noise_seeds=[args.seed + int(index) for index in indices],
                                   demonstration_actions_used_as_input=False)
        finally:
            trajectory.close()
    elif args.observation:
        with np.load(args.observation, allow_pickle=False) as source_file:
            observation = {key: source_file[key] for key in source_file.files}
        if 'timing_json' in observation:
            observation['observation.timing'] = json.loads(str(observation.pop('timing_json')))
        started = time.monotonic()
        predictions.append(policy.predict(observation))
        durations.append(time.monotonic() - started)
    else:
        from brainco_gripper_data import MissingGripperData
        from observation_stream import ObservationStream
        config = json.loads((ROOT / 'config/local.json').read_text())
        config['recording_cameras'] = 'all'
        stream = ObservationStream(config)
        try:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                try:
                    snapshot = stream.snapshot()
                except MissingGripperData:
                    time.sleep(.02)
                    continue
                started = time.monotonic()
                predictions.append(policy.predict(policy.adapter.observation(snapshot)))
                durations.append(time.monotonic() - started)
                observations_timing.append(stream.commit(snapshot))
        finally:
            stream.close()
    if not predictions:
        raise RuntimeError('No fresh local observations were available; no predictions were produced')
    metadata = dict(task=args.task, identity=policy.identity, inference_seconds=durations,
                    observation_timing=observations_timing, robot_commands_sent=False,
                    live_robot_observation=args.live, action_hz=policy.adapter.action_hz,
                    sampling_base_seed=args.seed)
    if trajectory_info is not None:
        metadata['trajectory'] = trajectory_info
    payload = dict(actions=np.stack(predictions))
    if raw_predictions:
        payload['raw_actions'] = np.stack(raw_predictions)
        metadata['prediction_space'] = 'source_bound_projected_controller_commands'
    if args.execution_contract:
        contract = json.loads(args.execution_contract.read_text())
        targets = [policy.adapter.local_targets(chunk, contract) for chunk in predictions]
        for key in ('motion_token', 'brainco_q', 'brainco_speed'):
            payload[key] = np.stack([target[key] for target in targets])
        metadata.update(deployment_contract=contract, proposals_only=True)
    with args.output.open('xb') as output:
        np.savez_compressed(output, **payload,
                            metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps(dict(output=str(args.output), predictions=len(predictions),
                          identity=policy.identity, inference_seconds=durations,
                          robot_commands_sent=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
