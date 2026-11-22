"""Revo2 DDS bridge. Default: preview only, no DDS initialization or writes.

Publishing requires verified hardware config, fresh PICO input, running body
policy debug packets, and fresh six-channel feedback from both hands.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import zmq
from brainco_controls import TOPICS, grasp_command_speeds, validate_grasp_config, hand_execution_policy
from recording_runtime import write_runtime_manifest, decode_runtime_identity

ROOT = Path(__file__).resolve().parents[1]
HEADER_SIZE = 1280
DTYPES = {'f32':'<f4', 'f64':'<f8', 'i32':'<i4', 'i64':'<i8', 'bool':'?'}


def decode_manager(raw):
    prefix = b'manager_state'
    if not raw.startswith(prefix) or len(raw) < len(prefix)+HEADER_SIZE:
        raise ValueError('Invalid manager packet')
    offset = len(prefix)+HEADER_SIZE
    header = json.loads(raw[len(prefix):offset].rstrip(b'\0 '))
    result = {}
    for field in header['fields']:
        dtype = np.dtype(DTYPES[field['dtype']])
        count = math.prod(field['shape'])
        if count < 0 or offset+count*dtype.itemsize > len(raw):
            raise ValueError('Invalid field length')
        result[field['name']] = np.frombuffer(raw, dtype, count, offset).tolist()
        offset += count*dtype.itemsize
    if offset != len(raw):
        raise ValueError('Unexpected trailing bytes')
    return result


def valid_positions(q):
    return len(q) == 6 and all(math.isfinite(x) and 0 <= x <= 1 for x in q)


def blocking_reasons(latest, manager_age, debug_age, feedback_ages, *, expected_config_hash=None):
    targets = [latest.get('brainco_left_q', []), latest.get('brainco_right_q', [])]
    reasons=[]
    if expected_config_hash is not None:
        identity = decode_runtime_identity(latest)
        if identity is None:
            reasons.append('manager_runtime_identity_missing')
        elif identity['grasp_config_sha256'] != expected_config_hash:
            reasons.append('manager_bridge_grasp_config_mismatch')
    if latest.get('stream_mode', [0])[0] not in (1,2,3,4,5):reasons.append('policy_not_active')
    if latest.get('brainco_input_valid') != [True]:reasons.append('pico_input_invalid')
    if latest.get('brainco_hand_enabled') != [True]:reasons.append('fingers_not_enabled')
    if latest.get('upper_body_motion_enabled') != [True]:reasons.append('body_mode_not_active')
    ages=[('manager_stale',manager_age),('body_debug_stale',debug_age)]
    if len(feedback_ages)!=2:reasons.append('both_hand_feedback_required')
    else:ages.extend(zip(('left_hand_feedback_stale','right_hand_feedback_stale'),feedback_ages))
    for reason,age in ages:
        if not math.isfinite(age) or not 0<=age<=.5:reasons.append(reason)
    if not all(valid_positions(q) for q in targets):reasons.append('invalid_hand_targets')
    return reasons


def ready_to_publish(latest, manager_age, debug_age, feedback_ages, **kwargs):
    return not blocking_reasons(latest,manager_age,debug_age,feedback_ages,**kwargs)


class FeedbackSlew:
    """Start at actual feedback; bound subsequent normalized target changes."""
    def __init__(self, rate=.8):
        if not math.isfinite(rate) or not 0 < rate <= 10:
            raise ValueError('Target slew rate must be in (0,10]')
        self.rate = rate
        self.reset()

    def reset(self):
        self.previous = None
        self.last_time = None

    def update(self, target, feedback, now):
        if not valid_positions(target) or not valid_positions(feedback):
            raise ValueError('Expected six normalized positions')
        if self.previous is None:
            self.previous = list(feedback)
            self.last_time = now
            return self.previous[:]
        delta = self.rate*min(.05, max(0, now-self.last_time))
        self.previous = [old+max(-delta, min(delta, goal-old))
                         for goal, old in zip(target, self.previous)]
        self.last_time = now
        return self.previous[:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--duration', type=float, default=10, help='0 runs until interrupted')
    parser.add_argument('--speed', type=float, help='Override the configured normalized speed ceiling')
    from robot_network import LOCAL_CONFIG, local_dds_config
    parser.add_argument('--config', type=Path, default=LOCAL_CONFIG)
    args = parser.parse_args()
    if args.speed is not None and (not math.isfinite(args.speed) or not 0 < args.speed <= 1):
        parser.error('speed must be in (0,1]')
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be nonnegative')
    readers = writers = []
    hardware = json.loads((ROOT/'config/hand-hardware.json').read_text())
    grasp_config = hardware.get('grasp_input')
    if grasp_config is not None:
        validate_grasp_config(grasp_config)
    execution = hand_execution_policy(grasp_config)
    args.speed = execution['speed_ceiling'] if args.speed is None else args.speed
    runtime_manifest = write_runtime_manifest(ROOT/'logs/brainco-bridge-runtime.json',
        'brainco_bridge',grasp_config,publish=args.publish,speed_ceiling=args.speed,
        target_rate_per_s=execution['target_rate_per_s'])
    if args.publish:
        if hardware.get('model') != 'Revo2' or hardware.get('service_verified') is not True:
            parser.error('Revo2 hand services have not yet been verified; preview only')
        from cyclonedds.domain import Domain, DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader
        from cyclonedds.pub import DataWriter
        from cyclonedds.core import Qos, Policy
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorCmd_, MotorStates_
        cfg = local_dds_config(args.config)
        domain = Domain(0, cfg); participant = DomainParticipant(0)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        readers = [DataReader(participant, Topic(participant, topic.replace('/cmd','/state'), MotorStates_), qos=qos) for topic in TOPICS]
        writers = [DataWriter(participant, Topic(participant, topic, MotorCmds_)) for topic in TOPICS]
    context = zmq.Context()
    manager = context.socket(zmq.SUB); debug = context.socket(zmq.SUB)
    manager.setsockopt(zmq.CONFLATE, 1); manager.setsockopt(zmq.SUBSCRIBE, b'manager_state')
    manager.connect('tcp://127.0.0.1:5556')
    debug.setsockopt(zmq.CONFLATE, 1); debug.setsockopt(zmq.SUBSCRIBE, b'g1_debug')
    debug.connect('tcp://127.0.0.1:5557')
    latest = {}; manager_time = debug_time = float('-inf')
    feedback = [None, None]; feedback_time = [float('-inf')]*2
    feedback_currents = [None, None]
    limiters = [FeedbackSlew(execution['target_rate_per_s']) for _ in range(2)]
    was_active = False
    freeze_pending = False
    frozen_feedback = None
    command_speeds = [None, None]
    published_q = [None, None]
    command_source = 'none'
    started = last_print = time.monotonic()
    try:
        while not args.duration or time.monotonic()-started < args.duration:
            now = time.monotonic()
            if manager.poll(0):
                try:
                    latest = decode_manager(manager.recv()); manager_time = now
                except (ValueError, KeyError, TypeError):
                    latest = {}; manager_time = float('-inf')
            if debug.poll(0):
                debug.recv(); debug_time = now
            for i, reader in enumerate(readers):
                for state in reader.take(100):
                    # DDS disposal notifications contain no motor telemetry.
                    if not hasattr(state, "states"):
                        continue
                    q = [motor.q for motor in state.states]
                    if valid_positions(q):
                        feedback[i] = q; feedback_time[i] = now
                        # The hand server places normalized SDK currents here,
                        # not torque in Nm or measured fingertip force.
                        current = [motor.tau_est for motor in state.states]
                        feedback_currents[i] = (current if len(current) == 6 and
                            all(math.isfinite(v) for v in current) else None)
            targets = [latest.get('brainco_left_q', []), latest.get('brainco_right_q', [])]
            blockers = blocking_reasons(latest,now-manager_time,now-debug_time,[now-t for t in feedback_time],
                                        expected_config_hash=runtime_manifest['grasp_config_sha256'])
            active = not blockers
            if was_active and not active:
                freeze_pending = True
            hand_feedback_fresh = all(now-t<=.5 for t in feedback_time) and all(q is not None for q in feedback)
            if freeze_pending and hand_feedback_fresh:
                frozen_feedback = [q[:] for q in feedback]
                freeze_pending = False
            if args.publish and active:
                frozen_feedback = None
                command_source = 'active_target'
                for i, writer in enumerate(writers):
                    previous = limiters[i].previous
                    q = limiters[i].update(targets[i], feedback[i], now)
                    speeds = grasp_command_speeds(grasp_config, args.speed, q, previous)
                    command_speeds[i] = speeds
                    msg = MotorCmds_([MotorCmd_(0, value, speed, 0, 0, 0, [0,0,0]) for value,speed in zip(q,speeds)])
                    writer.write(msg)
                    published_q[i] = q[:]
            else:
                for limiter in limiters: limiter.reset()
                command_speeds = [None, None]
                published_q = [None, None]
                command_source = 'none'
                # After an enabled session stops, hold the measured pose rather
                # than letting the previous closure target finish. Startup sends
                # nothing: this path exists only after prior active actuation.
                if args.publish and frozen_feedback is not None and hand_feedback_fresh:
                    command_source = 'frozen_feedback'
                    for i,(writer, q) in enumerate(zip(writers,frozen_feedback)):
                        writer.write(MotorCmds_([MotorCmd_(0,value,args.speed,0,0,0,[0,0,0]) for value in q]))
                        command_speeds[i] = [args.speed]*6
                        published_q[i] = q[:]
            was_active = active
            if now-last_print >= .5:
                status={'preview_only':not args.publish, 'publish_ready':bool(args.publish and active),
                                  'stream_mode':latest.get('stream_mode'), 'targets':targets,
                                  'hand_enabled':latest.get('brainco_hand_enabled') == [True],
                                  'triggers':latest.get('brainco_triggers'),
                                  'stick_clicks':latest.get('brainco_stick_clicks'),
                                  'grasp_profile':latest.get('brainco_grasp_profile'),
                                  'grasp_mapping_revision':latest.get('brainco_grasp_revision'),
                                  'command_speed_normalized':command_speeds,
                                  'published_q':published_q,
                                  'command_source':command_source,
                                  'speed_ceiling':args.speed,
                                  'target_rate_per_s':execution['target_rate_per_s'],
                                  'speed_policy_revision':(grasp_config or {}).get('speed_policy_revision',0),
                                  'grasp_config_sha256':runtime_manifest['grasp_config_sha256'],
                                  'runtime_session':runtime_manifest['session'],
                                  'feedback_q':feedback,
                                  'feedback_current_normalized':feedback_currents,
                                  'target_minus_feedback':[
                                      [a-b for a,b in zip(target,q)] if q is not None and valid_positions(target) else None
                                      for target,q in zip(targets,feedback)],
                                  'enable_armed':latest.get('brainco_hand_enable_armed'),
                                  'blocking_reasons':blockers,
                                  'requires_unlock':False,
                                  'body_debug_fresh':now-debug_time<=.5,
                                  'hands_feedback_fresh':[now-t<=.5 for t in feedback_time],
                                  'updated_at':time.time()}
                print(json.dumps(status), flush=True)
                temporary=ROOT/'logs/brainco-bridge-status.tmp'
                temporary.write_text(json.dumps(status,indent=2));temporary.replace(ROOT/'logs/brainco-bridge-status.json')
                last_print = now
            time.sleep(.02)
    finally:
        manager.close(); debug.close(); context.term()


if __name__ == '__main__': main()
