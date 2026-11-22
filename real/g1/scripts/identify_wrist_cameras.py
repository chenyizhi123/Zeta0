#!/usr/bin/env python3
"""Display isolated wrist streams so the operator can confirm their placement."""
from collections import deque
import json
from pathlib import Path
import time

import cv2
import msgpack
import numpy as np
import zmq


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/local.json').read_text())
    sources = config['camera_sources']
    output = root / 'logs'
    output.mkdir(exist_ok=True)
    title = 'G1 wrist identification - A / B - Q to close'
    context = zmq.Context()
    streams = []
    for label, key, serial, port in (
            ('A', 'camera_a', sources['video6']['usb_serial'], 15556),
            ('B', 'camera_b', sources['video8']['usb_serial'], 15557)):
        if not serial or serial.startswith('REPLACE_'):
            raise ValueError('Set verified wrist USB serials before identification')
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect('tcp://' + config['robot_host'] + ':' + str(port))
        streams.append(dict(label=label, key=key, serial=serial, socket=socket,
                            image=None, last_frame=0, frames=0, arrivals=deque(maxlen=60)))
    cv2.setNumThreads(1)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 550)
    last_saved = 0
    try:
        while True:
            for state in streams:
                if state['socket'].poll(1):
                    payload = msgpack.unpackb(state['socket'].recv(), raw=False)
                    if payload.get('camera', {}).get('serial') != state['serial']:
                        continue
                    frame = cv2.imdecode(np.frombuffer(payload['images'][state['key']], np.uint8),
                                         cv2.IMREAD_COLOR)
                    if frame is not None:
                        state.update(image=frame, last_frame=time.monotonic())
                        state['arrivals'].append(state['last_frame'])
                        state['frames'] += 1
            now = time.monotonic()
            panels, statuses = [], []
            for state in streams:
                panel = np.zeros((550, 640, 3), np.uint8)
                if state['image'] is not None:
                    panel[70:] = cv2.resize(state['image'], (640, 480))
                live = now - state['last_frame'] < 2
                arrivals = state['arrivals']
                fps = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                       if len(arrivals) > 1 else 0)
                color = (0, 255, 0) if live else (0, 80, 255)
                cv2.putText(panel, state['label'] + '  SN ' + state['serial'],
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2, cv2.LINE_AA)
                status = ('LIVE %.1f FPS' % fps) if live else 'WAITING / STALE'
                cv2.putText(panel, status + '  |  Cover one camera to identify',
                            (12, 56), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                panels.append(panel)
                statuses.append(dict(label=state['label'], serial=state['serial'], live=live,
                                     frames=state['frames'], fps=fps))
            canvas = np.hstack(panels)
            cv2.imshow(title, canvas)
            if now - last_saved > 1:
                cv2.imwrite(str(output / 'wrist-identification.jpg'), canvas)
                (output / 'wrist-identification-status.json').write_text(json.dumps(
                    dict(updated_at=time.time(), cameras=statuses), indent=2))
                last_saved = now
            if cv2.waitKey(10) & 0xff in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        for state in streams:
            state['socket'].close()
        context.term()


if __name__ == '__main__':
    main()
