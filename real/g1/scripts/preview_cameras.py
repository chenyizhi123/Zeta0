#!/usr/bin/env python3
"""Show the three labeled G1 camera views in a single live window."""
from collections import deque
import json
from pathlib import Path
import time

import cv2
import msgpack
import numpy as np
import zmq

root = Path(__file__).resolve().parents[1]
config = json.loads((root/'config/local.json').read_text())
title = 'chenyz G1 - Head + Left Wrist + Right Wrist'
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
socket.setsockopt(zmq.CONFLATE, 1)
socket.setsockopt(zmq.LINGER, 0)
socket.connect('tcp://127.0.0.1:%d' % config['camera_output_port'])
cv2.setNumThreads(1)
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, 1920, 480)
names = [('ego_view','Head D455'),('left_wrist','Left Wrist D415'),('right_wrist','Right Wrist D415')]
arrivals = deque(maxlen=60)
last_frame = 0
last_saved = 0
frames = 0
payload = None
canvas = np.zeros((480,1920,3),np.uint8)
try:
    while True:
        if socket.poll(10):
            payload = msgpack.unpackb(socket.recv(),raw=False)
            now = time.monotonic()
            arrivals.append(now)
            last_frame = now
            frames += 1
            panels = []
            for key,label in names:
                if key not in payload['images']:
                    continue
                image = cv2.imdecode(np.frombuffer(payload['images'][key],np.uint8),cv2.IMREAD_COLOR)
                panel = cv2.resize(image,(640,480))
                cv2.rectangle(panel,(0,0),(640,36),(0,0,0),-1)
                cv2.putText(panel,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),1,cv2.LINE_AA)
                panels.append(panel)
            if len(panels)==3:
                canvas = np.hstack(panels)
                if now-last_saved>=2:
                    cv2.imwrite(str(root/'reports/evidence/three-camera-preview.jpg'),canvas)
                    fps=(len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else 0
                    (root/'logs/camera-preview-status.json').write_text(json.dumps(dict(
                        updated_at=time.time(), frames=frames, fps=fps, keys=list(payload['images']),
                        software_reception_skew_ms=payload['sync']['skew_ms']),indent=2))
                    last_saved = now
        display = canvas.copy()
        if time.monotonic()-last_frame>2:
            cv2.putText(display,'Waiting for all three live cameras...',(20,65),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(0,80,255),1,cv2.LINE_AA)
        cv2.imshow(title,display)
        if cv2.waitKey(1)&0xff in (ord('q'),ord('Q'),27):
            break
        if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
            break
finally:
    cv2.destroyAllWindows()
    socket.close()
    context.term()
