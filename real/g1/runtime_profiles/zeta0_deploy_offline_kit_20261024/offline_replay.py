"""Replay real recorded current observations without opening robot interfaces."""
from pathlib import Path
import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))

import av
import numpy as np
from PIL import Image

from psi.data.predictive_world.cache import file_sha256
from psi.deploy.real_sonic_protocol import FORMAT, validate_request
from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
from psi.real_observation_contract import validate_snapshot_timing
from psi.sonic_actions import controller_command_values


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing fixture file: {relative}')
    return path


class RecordedTrajectory:
    def __init__(self, root, runtime):
        self.root = Path(root).expanduser().resolve()
        self.spec = json.loads((self.root / 'trajectory.json').read_text())
        self.runtime = runtime
        self.manifest = runtime.read_json('inference_manifest.json')
        spec = self.spec
        if spec['format'] != 'zeta0.offline_trajectory.v1':
            raise ValueError('Unsupported trajectory format')
        if (spec['model_weights_sha256'] != self.manifest['weights_sha256']
                or spec['model_inference_manifest_sha256'] != file_sha256(runtime.root / 'inference_manifest.json')
                or spec['instruction'] != self.manifest['task']
                or spec['observation_contract'] != runtime.observation_contract):
            raise ValueError('Trajectory/model mismatch: select the matching task and 20000-step model package')
        if spec['cameras'] != self.manifest['visual_keys'] or spec['storage_fps'] != 50:
            raise ValueError('Trajectory camera/storage ABI differs from the model')
        for relative, digest in spec['files_sha256'].items():
            if file_sha256(contained(self.root, relative)) != digest:
                raise ValueError(f'Trajectory checksum mismatch: {relative}')
        with np.load(contained(self.root, spec['observations_file']), allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        n = spec['frames']
        shapes = dict(frame_indices=(n,), state_values=(n, 41), decision_times=(n,),
            elapsed_times=(n,), camera_times=(n, 3), camera_anchor_times=(n,), body_times=(n,),
            hand_times=(n, 2), video_frame_indices=(n, 3), sample_segments=(n,))
        if set(self.data) != set(shapes):
            raise ValueError('Unexpected observation fields')
        for key, shape in shapes.items():
            if self.data[key].shape != shape or not np.isfinite(self.data[key]).all():
                raise ValueError(f'Invalid trajectory array: {key}')
        if not np.array_equal(self.data['frame_indices'], np.arange(n)) or (np.diff(self.data['decision_times']) <= 0).any():
            raise ValueError('Frame addresses/decision times must be strictly increasing')
        self.videos = {}
        self.validate_all_timing()

    def validate_all_timing(self):
        contract = self.runtime.observation_contract
        for index, anchor in enumerate(self.data['decision_times']):
            validate_snapshot_timing(decision_time=float(anchor),
                camera_anchor_time=float(self.data['camera_anchor_times'][index]),
                camera_times={key: float(t) for key, t in zip(self.spec['cameras'], self.data['camera_times'][index])},
                body_time=float(self.data['body_times'][index]), hand_times=self.data['hand_times'][index].tolist(),
                contract=contract['collector'], snapshot_max_age_s=contract.get('real_snapshot_max_age_s'))

    def image(self, camera, frame_index):
        if camera not in self.videos:
            container = av.open(str(contained(self.root, self.spec['video_paths'][camera])))
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if (stream.width, stream.height) != (640, 480) or stream.frames != self.spec['frames']:
                container.close()
                raise ValueError('Video geometry/frame count differs from the complete recorded episode')
            self.videos[camera] = (container, stream)
        container, stream = self.videos[camera]
        fps = self.spec['storage_fps']
        container.seek(int(frame_index / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            actual = round(float(frame.pts * frame.time_base) * fps)
            if actual < frame_index:
                continue
            if actual != frame_index:
                raise ValueError(f'Could not address recorded frame exactly: {camera}/{frame_index}')
            image = Image.fromarray(frame.to_ndarray(format='rgb24'))
            output = BytesIO()
            image.save(output, format='PNG', compress_level=1)
            return base64.b64encode(output.getvalue()).decode('ascii')
        raise ValueError(f'Missing recorded video frame: {camera}/{frame_index}')

    def request(self, index):
        anchor = float(self.data['decision_times'][index])
        body_time = float(self.data['body_times'][index]) - anchor
        hands = (self.data['hand_times'][index] - anchor).tolist()
        images = {camera: dict(encoded=self.image(camera, int(self.data['video_frame_indices'][index, i])),
            time=float(self.data['camera_times'][index, i]) - anchor) for i, camera in enumerate(self.spec['cameras'])}
        request = dict(format=FORMAT, session_id='offline/' + self.spec['task_key'], request_seq=int(index),
            queue_revision=0, anchor_time=anchor, task=self.manifest['task'],
            action_protocol_id=self.manifest['action_protocol_id'], state_schema_id=self.manifest['state_schema_id'],
            state=dict(values=[self.data['state_values'][index].tolist()], valid=[[True]*41], times=[max(body_time, *hands)]),
            images=images, snapshot_timing=dict(camera_anchor_time=float(self.data['camera_anchor_times'][index])-anchor,
                body_time=body_time, hand_times=hands), committed_prefix=[])
        validate_request(request, state_count=1, rtc_max_prefix_steps=self.runtime.rtc_max_prefix_steps,
            observation_contract=self.runtime.observation_contract)
        return request

    def close(self):
        for container, _ in self.videos.values():
            container.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--trajectory-dir', required=True)
    parser.add_argument('--qwen-model', help='Existing matching local Qwen3-VL; required for inference')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-frames', type=int, default=3, help='Evenly sample this many anchors; 0 means all selected anchors')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check-only', action='store_true', help='Check all timing and selected real inputs; no model inference')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if args.max_frames < 0 or args.stride < 1:
        parser.error('--max-frames must be nonnegative; --stride must be positive')
    if not args.check_only and not args.qwen_model:
        parser.error('Inference requires --qwen-model')
    runtime = ResearchInferenceRuntime(args.weights_dir)
    trajectory = RecordedTrajectory(args.trajectory_dir, runtime)
    try:
        anchors = np.arange(0, trajectory.spec['frames'], args.stride)
        if args.max_frames and len(anchors) > args.max_frames:
            anchors = anchors[np.linspace(0, len(anchors)-1, args.max_frames, dtype=int)]
        policy, processor = None, None
        if args.check_only and args.qwen_model:
            from transformers import AutoProcessor
            from psi.data.predictive_world.provenance import verify_qwen_provenance
            processor = AutoProcessor.from_pretrained(args.qwen_model, local_files_only=True)
            verify_qwen_provenance(args.qwen_model, runtime.read_json('feature_provenance.json'), processor=processor)
        elif not args.check_only:
            policy, processor = runtime.build_policy(args.qwen_model, device=args.device)
        actions, input_shapes = [], []
        for index in anchors.tolist():
            request = trajectory.request(index)
            if args.check_only:
                if processor is not None:
                    observed = runtime.prepare_observation(request, processor, device='cpu')
                    input_shapes.append({k: list(v.shape) for k, v in observed.qwen_inputs.items()})
            else:
                actions.append(runtime.infer_request(policy, processor, request, seed=args.seed + index))
            print(json.dumps(dict(event='offline_input_checked' if args.check_only else 'offline_action_generated',
                task=trajectory.spec['task_key'], frame=index)), flush=True)
        report = dict(task=trajectory.spec['task_key'], instruction=trajectory.spec['instruction'],
            runtime_origin=str(Path(sys.modules['psi'].__file__).resolve()),
            weights_sha256=trajectory.manifest['weights_sha256'],
            trajectory_sha256=file_sha256(trajectory.root / 'trajectory.json'),
            complete_episode_frames=trajectory.spec['frames'], checked_timing_frames=trajectory.spec['frames'],
            selected_frame_indices=anchors.tolist(), state_history_count=1, committed_prefix_steps=0,
            seed=args.seed, device=None if args.check_only else args.device,
            mode='input_validation' if args.check_only else 'offline_open_loop_replay',
            actual_qwen_inputs_prepared=bool(input_shapes), qwen_input_shapes=input_shapes,
            demonstration_actions_used_as_input=False, robot_execution=False,
            full_vlm_policy_inference_run=not args.check_only)
        if not args.check_only or args.output_dir:
            output = Path(args.output_dir or ('offline_results/' + trajectory.spec['task_key'] + '/' +
                datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).expanduser().resolve()
            if output.exists():
                raise FileExistsError(f'Select a new output directory: {output}')
            output.mkdir(parents=True)
            if actions:
                raw = np.stack(actions)
                if raw.shape != (len(anchors), 30, 76) or not np.isfinite(raw).all():
                    raise ValueError('Expected finite [anchors,30,76] native actions')
                offsets = np.arange(1, 31, dtype=np.float64)/50
                np.savez_compressed(output / 'predictions.npz', frame_indices=anchors,
                    decision_times=trajectory.data['decision_times'][anchors], action_offsets_s=offsets,
                    action_absolute_times=trajectory.data['decision_times'][anchors, None] + offsets[None],
                    raw_actions=raw, controller_actions=controller_command_values(raw.reshape(-1, 76),
                        projection=runtime.observation_contract['action_projection']).reshape(raw.shape))
                report['action_shape'] = list(raw.shape)
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            report['output_dir'] = str(output)
        print(json.dumps(dict(event='offline_replay_complete', **report)), flush=True)
    finally:
        trajectory.close()


if __name__ == '__main__':
    main()
