#!/usr/bin/env python3
"""Pair RGB streams under the common collection/inference timing contract."""
import argparse
import json
from pathlib import Path
import signal
import time

import msgpack
import zmq

from observation_sync import CameraPairer


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=root/'config/local.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    pairer = CameraPairer(config)
    context = zmq.Context()
    poller = zmq.Poller()
    states = {}
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM,3)
    publisher.setsockopt(zmq.LINGER,0)
    publisher.bind(f"tcp://127.0.0.1:{config['camera_output_port']}")
    running = True
    def stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    last_status = 0
    started = time.monotonic()
    latest_payload = None
    (root/'logs').mkdir(parents=True,exist_ok=True)
    try:
        for name, source in config['camera_sources'].items():
            sock = context.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE,b'')
            sock.setsockopt(zmq.RCVHWM,12)
            sock.setsockopt(zmq.LINGER,0)
            sock.setsockopt(zmq.HEARTBEAT_IVL,1000)
            sock.setsockopt(zmq.HEARTBEAT_TIMEOUT,3000)
            sock.connect(f"tcp://{source['host']}:{source['port']}")
            poller.register(sock,zmq.POLLIN)
            states[sock] = (name,source)
        while running:
            for sock in dict(poller.poll(2)):
                name, source = states[sock]
                # Drain bounded bursts so accumulated network packets do not
                # remain artificially old while healthy sources are flowing.
                for _ in range(128):
                    try:
                        raw = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    mono, wall = time.monotonic(),time.time()
                    try:
                        payload = msgpack.unpackb(raw,raw=False)
                        if not isinstance(payload,dict):
                            raise ValueError('Camera payload must be a map')
                        pairer.push(name,source['key'],payload,mono,wall)
                    except (ValueError,TypeError,KeyError):
                        pairer.source_metrics[name].invalid_messages += 1
            now = time.monotonic()
            for payload in pairer.ready(now):
                try:
                    publisher.send(msgpack.packb(payload,use_bin_type=True),flags=zmq.NOBLOCK)
                    latest_payload = payload
                except zmq.Again:
                    pairer.publisher_drops += 1
            # Keep reporting even when no matched groups can be produced.
            if now-last_status >= 1:
                status = dict(updated_at=time.time(),frames=pairer.sequence,
                    rejected_groups=pairer.rejected,fps=pairer.sequence/(now-started),
                    fps_scope='composed_groups_per_subscriber_receive_time_not_hardware_fps',
                    keys=[] if latest_payload is None else list(latest_payload['images']),
                    latest_skew_ms=None if latest_payload is None else latest_payload['sync']['skew_ms'],
                    timestamp_kind='workstation_receive_time',hardware_synchronized=False,
                    observation_contract=pairer.contract,composer_session=pairer.session,
                    diagnostics=pairer.metrics(now))
                temporary = root/'logs/camera-composer-status.tmp'
                temporary.write_text(json.dumps(status,indent=2,allow_nan=False))
                temporary.replace(root/'logs/camera-composer-status.json')
                last_status = now
    finally:
        for sock in states:
            sock.close()
        publisher.close()
        context.term()


if __name__ == '__main__':
    main()
