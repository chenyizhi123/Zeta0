#!/usr/bin/env python3
"""Read-only observation entry for checking or feeding a future visual policy."""
import argparse
import json
from pathlib import Path
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from brainco_gripper_data import MissingGripperData
from observation_stream import ObservationStream

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='只读观察接口；不启动 SONIC，不发送动作')
    parser.add_argument('--config',type=Path,default=ROOT/'config/local.json')
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--cameras',choices=('all','head'),default='all',help='与采集一致的相机集合；head 直接读取顶部 RGB')
    parser.add_argument('--publish',help='可选 PUB 地址，例如 tcp://127.0.0.1:5564；输出与采集同规则的模型输入')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    config = json.loads(args.config.read_text())
    config['recording_cameras'] = args.cameras
    stream = ObservationStream(config)
    context = zmq.Context() if args.publish else None
    publisher = context.socket(zmq.PUB) if context else None
    summaries = []
    reasons = {}
    start = time.monotonic()
    try:
        if publisher:
            publisher.setsockopt(zmq.LINGER,0)
            publisher.setsockopt(zmq.SNDHWM,2)
            publisher.bind(args.publish)
        while time.monotonic()-start < args.seconds:
            tick = time.monotonic()
            try:
                snapshot = stream.snapshot()
                if publisher:
                    publisher.send(msgpack.packb(dict(model_input=snapshot['model_input'],
                        timing=snapshot['timing']),default=msgpack_numpy.encode,use_bin_type=True))
                timing = stream.commit(snapshot)
                summaries.append(dict(skew_ms=timing['camera_skew_ms'],
                    body_offset_ms=timing['body']['offset_ms'],
                    hand_offset_ms=timing['hand_state']['offset_ms'],
                    decision_age_ms=(timing['decision_monotonic']-timing['anchor_monotonic'])*1000,
                    repeated=[timing['cameras'][k]['repeated'] for k in stream.assembler.camera_keys]))
            except MissingGripperData as error:
                reasons[str(error)] = reasons.get(str(error),0)+1
            time.sleep(max(0,1/config['dataset_frequency']-(time.monotonic()-tick)))
    finally:
        stream.close()
        if publisher:
            publisher.close()
            context.term()
    result = dict(accepted=len(summaries),rejections=reasons,robot_commands_sent=False,
                  hardware_synchronized=False,observation_contract=stream.assembler.contract,
                  samples=summaries)
    if args.report:
        args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False,indent=2))
    return 0 if summaries else 1


if __name__ == '__main__':
    raise SystemExit(main())
