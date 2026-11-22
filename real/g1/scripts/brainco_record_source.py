"""Read-only DDS-to-ZMQ telemetry for normalized BrainCo gripper recording."""
import argparse
import json
from pathlib import Path
import time

import zmq
from brainco_gripper_data import ENDPOINT, GripperCache, MissingGripperData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=0)
    from robot_network import LOCAL_CONFIG, local_dds_config
    parser.add_argument('--config', type=Path, default=LOCAL_CONFIG)
    args = parser.parse_args()
    if args.duration < 0:
        parser.error('duration must be nonnegative')
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.core import Qos, Policy
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
    cfg = local_dds_config(args.config)
    domain = Domain(0, cfg)
    participant = DomainParticipant(0)
    qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
    readers = []
    for side in ('left','right'):
        for kind, suffix, dtype, attr in [('state','state',MotorStates_,'states'),
                                         ('action','cmd',MotorCmds_,'cmds')]:
            reader = DataReader(participant, Topic(participant, f'rt/brainco/{side}/{suffix}', dtype), qos=qos)
            readers.append((kind,side,attr,reader))
    context = zmq.Context()
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.SNDHWM, 1)
    output.bind(ENDPOINT)
    cache = GripperCache()
    started = last_print = time.monotonic()
    try:
        while not args.duration or time.monotonic()-started < args.duration:
            now = time.monotonic()
            for kind, side, attr, reader in readers:
                for msg in reader.take(20):
                    try:
                        motors=getattr(msg, attr)
                        cache.update(kind, side, [m.q for m in motors], now,
                                     speeds=[m.dq for m in motors] if kind=='action' else None)
                    except (MissingGripperData, AttributeError):
                        cache.samples.pop((kind,side), None)
            try:
                packet = cache.packet(now)
            except MissingGripperData as error:
                packet = {'valid':False, 'timestamp_monotonic':now, 'reason':str(error)}
            output.send_json(packet)
            if now-last_print >= 1:
                print(json.dumps(packet), flush=True)
                last_print = now
            time.sleep(.02)
    finally:
        output.close(linger=0)
        context.term()


if __name__ == '__main__':
    main()
