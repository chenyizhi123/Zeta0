"""Continuous BrainCo gripper labels for LeRobot v2.1; no robot writes."""
import copy
import math
import time
import uuid

import numpy as np

ENDPOINT = 'tcp://127.0.0.1:5563'
LEGACY_TELEOP_KEYS = ('teleop.left_hand_joints', 'teleop.right_hand_joints')


class MissingGripperData(ValueError):
    pass


def closure(q):
    """Mean of six normalized feedback/command joints, 0=open, 1=closed."""
    if len(q) != 6 or any(not math.isfinite(float(x)) or not 0 <= x <= 1 for x in q):
        raise MissingGripperData('Expected six finite normalized hand positions')
    return float(sum(q)/6)


class GripperCache:
    def __init__(self, max_age=.2):
        self.max_age = max_age
        self.samples = {}
        self.command_speeds = {}
        self.source_session = uuid.uuid4().hex
        self.receive_sequence = {}

    def update(self, kind, side, q, now, speeds=None):
        if kind not in ('state', 'action') or side not in ('left', 'right'):
            raise ValueError('Unknown hand data channel')
        mean = closure(q)
        if kind == 'action':
            if speeds is not None and (len(speeds)!=6 or any(not math.isfinite(float(v)) or not 0<=v<=1 for v in speeds)):
                raise MissingGripperData('Invalid normalized actual DDS command speeds')
            self.command_speeds[side] = [float(v) for v in speeds] if speeds is not None else None
        self.samples[kind, side] = (mean, now, [float(x) for x in q])
        self.receive_sequence[kind,side]=self.receive_sequence.get((kind,side),0)+1

    def packet(self, now):
        result = {'valid': True, 'timestamp_monotonic': now, 'source_session':self.source_session}
        ages = []
        for kind in ('state', 'action'):
            values = []
            joints = []
            for side in ('left', 'right'):
                if (kind, side) not in self.samples:
                    raise MissingGripperData('Waiting for both hand states and actual commands')
                value, stamp, raw = self.samples[kind, side]
                age = now-stamp
                if not math.isfinite(age) or not 0 <= age <= self.max_age:
                    raise MissingGripperData('Hand state/command is stale')
                values.append(value)
                joints.append(raw)
                ages.append(age)
            result[kind] = values
            result[kind+'_joints'] = joints
        result['receive_ages'] = ages  # state L/R, command L/R
        result['action_speed_joints'] = [self.command_speeds.get(side) for side in ('left','right')]
        if any(value is None for value in result['action_speed_joints']):
            result['action_speed_joints'] = None
        result['action_receive_monotonic']=[self.samples['action',side][1] for side in ('left','right')]
        result['action_receive_sequence']=[self.receive_sequence['action',side] for side in ('left','right')]
        return result


def packet_features(packet, now, max_age=.2):
    """Validate telemetry; never substitute requested targets for actual feedback."""
    try:
        age = now-packet['timestamp_monotonic']
        ages = packet['receive_ages']
        if packet['valid'] is not True or not math.isfinite(age) or not 0 <= age <= max_age:
            raise MissingGripperData('Gripper telemetry unavailable or stale')
        if len(ages) != 4 or any(not math.isfinite(a) or not 0 <= a+age <= max_age for a in ages):
            raise MissingGripperData('Underlying hand messages are stale')
        result = {}
        for kind, key in [('state','observation.gripper'), ('action','action.gripper')]:
            q = packet[kind]
            if len(q) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in q):
                raise MissingGripperData('Invalid gripper scalars')
            result[key] = np.asarray(q, dtype=np.float32)
        result['observation.gripper_receive_age'] = np.asarray([a+age for a in ages], dtype=np.float32)
        return result
    except (KeyError, TypeError) as error:
        raise MissingGripperData('Malformed gripper telemetry') from error


def body_indices(model):
    hands = set(model.get_joint_group_indices('left_hand')) | set(model.get_joint_group_indices('right_hand'))
    return [i for i in range(model.num_joints) if i not in hands]


def adapt_schema(features, modalities, model):
    features, modalities = copy.deepcopy(features), copy.deepcopy(modalities)
    keep = body_indices(model)
    for key in ('observation.state','action.wbc'):
        features[key]['shape'] = (len(keep),)
        features[key]['names'] = [model.joint_names[i] for i in keep]
    for key in LEGACY_TELEOP_KEYS:
        features.pop(key, None)
    for key in ('observation.gripper','action.gripper'):
        features[key] = {'dtype':'float32', 'shape':(2,), 'names':['left_gripper','right_gripper']}
    features['observation.gripper_receive_age'] = {
        'dtype':'float32', 'shape':(4,),
        'names':['left_state_age','right_state_age','left_command_age','right_command_age']}
    remap = {old:new for new, old in enumerate(keep)}
    for section in ('state','action'):
        for key in ('left_hand','right_hand','left_hand_joints','right_hand_joints'):
            modalities[section].pop(key, None)
        for group in ('left_leg','right_leg','waist','left_arm','right_arm'):
            if group in modalities[section]:
                indices = sorted(model.get_joint_group_indices(group))
                modalities[section][group].update(start=remap[indices[0]], end=remap[indices[-1]]+1)
        for i, side in enumerate(('left','right')):
            modalities[section][side+'_gripper'] = {
                'start':i, 'end':i+1,
                'original_key':'observation.gripper' if section=='state' else 'action.gripper'}
    return features, modalities


def adapt_frame(frame, model, packet, now):
    frame = dict(frame)
    keep = body_indices(model)
    for key in ('observation.state','action.wbc'):
        frame[key] = np.asarray(frame[key])[keep].copy()
    for key in LEGACY_TELEOP_KEYS:
        frame.pop(key, None)
    frame.update(packet_features(packet, now))
    return frame


class GripperSubscriber:
    def __init__(self, endpoint=ENDPOINT):
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.connect(endpoint)
        self.latest = {}

    def packet(self):
        if self.socket.poll(0):
            try:
                self.latest = self.socket.recv_json()
            except (ValueError, TypeError):
                self.latest = {}
        packet_features(self.latest, time.monotonic())
        return self.latest

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def install_gripper_collector(sonic):
    """Adapt the existing collector without modifying its camera/body control."""
    original = sonic.GrootDataCollector

    class GripperCollector(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gripper_source = GripperSubscriber()
            original_add = self.data_exporter.add_frame

            def add(frame):
                original_add(adapt_frame(frame, self.robot_model,
                                         self.gripper_source.packet(), time.monotonic()))
            self.data_exporter.add_frame = add

        def _add_data_frame_sonic(self, t_start):
            try:
                return super()._add_data_frame_sonic(t_start)
            except MissingGripperData as error:
                # Do not fill missing labels with zero or silently compress time gaps.
                # Only the current recording is reset; robot control is unaffected.
                if self.data_exporter.episode_buffer.get('size', 0):
                    self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._print_and_say('Recording reset: '+str(error), say=False)
                return False

        def save_and_cleanup(self):
            try:
                super().save_and_cleanup()
            finally:
                self.gripper_source.close()

    sonic.GrootDataCollector = GripperCollector
