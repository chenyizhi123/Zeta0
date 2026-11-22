"""Offline model I/O adapter for the existing SONIC and BrainCo interfaces.

This module does not initialize a model, open sockets, or enable robot control.
Observations use actual six-channel BrainCo feedback, not the legacy two means
or the seven-channel Dex3 representation in the upstream VLA client.
"""
import argparse
import copy
import json
from pathlib import Path
import struct

import numpy as np

from training_actions import decode_model_action


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ('ego_view', 'left_wrist', 'right_wrist')
STATE_SCHEMA = 'real_g1_q29_brainco12_v1'
ACTION_PROTOCOL = 'real_sonic_motion64_brainco12_v1'


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return array.copy()


class LocalVLAAdapter:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir).resolve()
        self.manifest = json.loads((self.weights_dir / 'inference_manifest.json').read_text())
        if self.manifest.get('state_schema_id') != STATE_SCHEMA:
            raise ValueError('The model does not use the local G1/BrainCo state schema')
        if self.manifest.get('action_protocol_id') != ACTION_PROTOCOL:
            raise ValueError('The model does not use SONIC token64 + BrainCo position12')
        expected = ['observation.images.' + name for name in CAMERAS]
        if self.manifest.get('visual_keys') != expected:
            raise ValueError('The model camera order must be head, left wrist, right wrist')
        temporal = self.manifest['temporal_abi']
        self.horizon = temporal['action_horizon']
        self.action_hz = temporal['action_hz']
        if type(self.horizon) is not int or self.horizon <= 0 or (self.action_hz, self.horizon) not in ((50, 30), (30, 50)):
            raise ValueError('Unsupported model action grid; expected 50Hz/30 or 30Hz/50')
        self.state = self._registry('state_schema_registry.json', STATE_SCHEMA, 41)
        self.action = self._registry('action_protocol_registry.json', ACTION_PROTOCOL, 76)

    def _registry(self, filename, key, dim):
        registry = json.loads((self.weights_dir / 'registries' / filename).read_text())[key]
        if registry.get('dim') != dim:
            raise ValueError(f'{key} must have {dim} dimensions')
        mean = finite_array(registry.get('mean'), (dim,), key + ' mean')
        std = finite_array(registry.get('std'), (dim,), key + ' std')
        if np.any(std <= 0):
            raise ValueError(key + ' standard deviations must be positive')
        return dict(mean=mean, std=std)

    def observation(self, snapshot):
        """Convert an already time-validated local ObservationStream snapshot."""
        body = finite_array(snapshot['proprio']['body_q'], (29,), 'body feedback')
        hands = finite_array(snapshot['packet'].get('state_joints'), (2, 6), 'actual hand feedback')
        if np.any((hands < 0) | (hands > 1)):
            raise ValueError('Actual BrainCo hand positions must be normalized to 0..1')
        images = snapshot['image_message']['images']
        result = {'observation.state': body, 'observation.hand_q': hands.reshape(12).copy()}
        for name in CAMERAS:
            image = np.asarray(images[name])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(name + ' must be a 640x480 uint8 RGB image')
            result['observation.images.' + name] = image.copy()
        if 'timing' in snapshot:
            result['observation.timing'] = copy.deepcopy(snapshot['timing'])
        return result

    def state_vector(self, observation, *, normalized=False):
        """Keep body29, left fingers6, right fingers6 in their recorded order."""
        body = finite_array(observation['observation.state'], (29,), 'body feedback')
        hands = finite_array(observation['observation.hand_q'], (12,), 'actual hand feedback')
        if np.any((hands < 0) | (hands > 1)):
            raise ValueError('Actual BrainCo hand positions must be normalized to 0..1')
        state = np.concatenate((body, hands))
        return (state - self.state['mean']) / self.state['std'] if normalized else state

    def action_chunk(self, values, *, normalized=False):
        """Return physical token/position targets; normalization is explicit."""
        chunk = finite_array(values, (self.horizon, 76), 'action chunk')
        if normalized:
            chunk = chunk * self.action['std'] + self.action['mean']
        if not np.isfinite(chunk).all():
            raise ValueError('Denormalized action chunk is not finite')
        if np.any((chunk[:, 64:] < 0) | (chunk[:, 64:] > 1)):
            raise ValueError('Predicted BrainCo positions must be normalized to 0..1')
        projection = (self.manifest.get('observation_contract') or {}).get('action_projection', {})
        low, high = projection.get('motion_token_min'), projection.get('motion_token_max')
        if self.manifest.get('observation_contract') is None:
            low, high = (-0.75, 0.6875) if (self.action_hz, self.horizon) == (30, 50) else (-10/16, 10/16)
        if (not isinstance(low, (int, float)) or not isinstance(high, (int, float))
                or not np.isfinite([low, high]).all() or low >= high):
            raise ValueError('The recorded SONIC token bounds are missing or invalid')
        if np.any(chunk[:, :64] < low - 1e-6) or np.any(chunk[:, :64] > high + 1e-6):
            raise ValueError('Predicted SONIC tokens exceed the recorded task bounds')
        return chunk

    def local_targets(self, values, deployment_contract, *, normalized=False):
        """Reuse the local decoder and its recorded speed/mapping validation.

        The returned targets are proposals. The existing publisher must still
        apply FeedbackSlew starting from fresh actual feedback at execution time.
        No hand identity, PICO-valid flag, or enable flag is synthesized here.
        """
        chunk = self.action_chunk(values, normalized=normalized)
        spec = dict(space='sonic-hand-position', body_key='action.motion_token')
        targets = [decode_model_action(row, spec, deployment_contract) for row in chunk]
        return dict(
            motion_token=np.asarray([v['body_target'] for v in targets], dtype=np.float32),
            brainco_q=np.asarray([v['hand_q'] for v in targets], dtype=np.float32),
            brainco_speed=np.asarray([v['hand_speed'] for v in targets], dtype=np.float32),
            hand_target_rate_per_s=targets[0]['hand_target_rate_per_s'],
            grasp_config_sha256=targets[0]['grasp_config_sha256'],
            action_hz=self.action_hz,
        )

    def inspect(self):
        """Inspect stored tensors without claiming a network forward pass."""
        filename = self.manifest['weights_file']
        path = (self.weights_dir / filename).resolve()
        if path.parent != self.weights_dir:
            raise ValueError('Weights must be inside the model directory')
        with path.open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            if size > 32 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            tensors = json.loads(source.read(size))
        tensors.pop('__metadata__', None)
        if len(tensors) != self.manifest['tensor_count']:
            raise ValueError('Stored tensor count differs from the model manifest')
        return dict(
            weights_dir=str(self.weights_dir), task=self.manifest.get('task'),
            state_dimensions=41, action_dimensions=76, camera_order=list(CAMERAS),
            action_horizon=self.horizon, action_hz=self.action_hz,
            tensor_count=len(tensors), tensor_modules=sorted({k.split('.')[0] for k in tensors}),
            local_action_decoder='training_actions.decode_model_action',
            network_forward_verified=False, robot_commands_sent=False,
        )


def main():
    parser = argparse.ArgumentParser(description='本地 SONIC 模型接口检查与离线动作适配；不发送机器人指令')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--task', choices=('flatten_box', 'tidy_table_trash', 'place_chair'))
    source.add_argument('--weights-dir', type=Path)
    parser.add_argument('--model-paths', type=Path, default=ROOT / '.local/models/zeta0/model_paths.json')
    parser.add_argument('--actions', type=Path, help='模型预测的 horizon×76 数组，.npy 格式')
    parser.add_argument('--normalized-actions', action='store_true', help='显式按本任务 registry 反归一化')
    parser.add_argument('--execution-contract', type=Path, help='采集/训练时保存的 deployment_contract JSON')
    parser.add_argument('--output', type=Path, help='离线转换结果 .npz；不执行动作')
    args = parser.parse_args()
    if args.actions and (not args.execution_contract or not args.output):
        parser.error('--actions requires --execution-contract and --output')
    if not args.actions and (args.normalized_actions or args.execution_contract or args.output):
        parser.error('Action conversion options require --actions')
    weights = args.weights_dir
    if weights is None:
        paths = json.loads(args.model_paths.read_text())
        weights = Path(paths['tasks'][args.task]['weights_dir'])
    adapter = LocalVLAAdapter(weights)
    result = adapter.inspect()
    if args.actions:
        if args.output.exists():
            parser.error('Output already exists')
        contract = json.loads(args.execution_contract.read_text())
        targets = adapter.local_targets(np.load(args.actions, allow_pickle=False), contract,
                                        normalized=args.normalized_actions)
        metadata = {k: v for k, v in targets.items() if not isinstance(v, np.ndarray)}
        metadata.update(deployment_contract=contract, weights_dir=str(adapter.weights_dir),
                        proposals_only=True, robot_commands_sent=False)
        with args.output.open('xb') as stream:
            np.savez_compressed(stream, metadata_json=np.asarray(json.dumps(metadata)),
                                **{k: v for k, v in targets.items() if isinstance(v, np.ndarray)})
        result.update(output=str(args.output), frames=adapter.horizon,
                      execution_metadata=metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
