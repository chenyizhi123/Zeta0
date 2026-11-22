#!/usr/bin/env python3
"""Publish one verified wrist RGB camera for recording and visual inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import uuid

import cv2
import msgpack
import zmq
from wrist_camera import (wrist_camera_settings, wrist_identity_matches, wrist_capture_matches,
                          resolve_wrist_usb_identity)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/dev/video0')
parser.add_argument('--source-key', help='Fixed images key; defaults to the resolved video node name')
parser.add_argument('--expected-usb-path', help='Require this USB path unless a stable USB serial is configured')
parser.add_argument('--expected-usb-serial', help='Bind this exact camera serial across USB port changes')
parser.add_argument('--expected-usb-vendor', default='1c45')
parser.add_argument('--expected-usb-product', default='6200')
parser.add_argument('--expected-usb-interface')
parser.add_argument('--pixel-format', choices=['MJPG', 'YUYV'], default='MJPG')
parser.add_argument('--bind', required=True, help='Local robot camera bind endpoint')
parser.add_argument('--status', default='/tmp/sonic-wrist-status.json')
parser.add_argument('--wait-seconds', type=float, default=0)
args = parser.parse_args()
deadline = time.monotonic() + args.wait_seconds
while True:
    node = Path(args.device).resolve().name
    parent = (Path('/sys/class/video4linux') / node / 'device').resolve()
    usb = next((p for p in [parent, *parent.parents] if (p / 'idVendor').exists()), None)
    if usb is not None and Path(args.device).exists():
        break
    if time.monotonic() >= deadline:
        raise RuntimeError('USB camera identity unavailable')
    time.sleep(1)
identity = {key: (usb / key).read_text().strip() for key in ('idVendor', 'idProduct', 'product', 'serial') if (usb / key).exists()}
identity['usb_path'] = usb.name
if args.expected_usb_serial is None and args.expected_usb_path and identity['usb_path'] != args.expected_usb_path:
    raise RuntimeError('USB camera path mismatch: expected ' + args.expected_usb_path + ', got ' + identity['usb_path'])
profile = dict(usb_path=args.expected_usb_path or usb.name,
    usb_vendor_id=args.expected_usb_vendor, usb_product_id=args.expected_usb_product,
    pixel_format=args.pixel_format)
if args.expected_usb_interface is not None:
    profile['usb_interface'] = args.expected_usb_interface
if args.expected_usb_serial is not None:
    profile['usb_serial'] = args.expected_usb_serial
settings = wrist_camera_settings(profile)
identity = resolve_wrist_usb_identity(profile, identity, usb)
if not wrist_identity_matches(profile, identity):
    raise RuntimeError('USB camera identity mismatch: selected device is not the configured wrist camera')
interface = next((p.name.split(':', 1)[1] for p in [parent, *parent.parents]
                  if p.parent == usb and ':' in p.name), None)
source_key = args.source_key or node
# A manual startup and the recovery watchdog may discover a missing publisher
# together. Keep a per-stream lock for the publisher's whole lifetime.
publisher_lock = open('/tmp/sonic-wrist-'+source_key+'-publisher.lock', 'a')
fcntl.flock(publisher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
query = bytearray(104)
with open(args.device, 'rb', buffering=0) as device:
    fcntl.ioctl(device, 0x80685600, query, True)
capabilities = struct.unpack_from('I', query, 84)[0]
if capabilities & 0x80000000:
    capabilities = struct.unpack_from('I', query, 88)[0]
index = int((Path('/sys/class/video4linux') / node / 'index').read_text().strip())
if not wrist_capture_matches(profile, identity, interface, index, capabilities):
    raise RuntimeError('The selected node is not the verified wrist RGB interface')
identity['usb_interface'] = interface
owner = subprocess.run(['fuser', args.device], capture_output=True)
if owner.returncode == 0:
    raise RuntimeError('The camera is already in use: ' + owner.stdout.decode())
if owner.returncode != 1:
    raise RuntimeError('Could not check camera ownership')

running = True
def stop(signum, frame):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
cv2.setNumThreads(1)
context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.setsockopt(zmq.SNDHWM, 2)
publisher.setsockopt(zmq.LINGER, 0)
publisher.bind(args.bind)
camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
try:
    if not camera.isOpened():
        raise RuntimeError('Could not open the camera')
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings['pixel_format']))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    # One V4L2 buffer halves delivery on these cameras; three sustain 30 FPS.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) != 640 or int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 480:
        raise RuntimeError('Camera did not accept the 640x480 RGB observation shape')
    sequence = 0
    failures = 0
    started = time.monotonic()
    last_status = started
    source_session = uuid.uuid4().hex
    while running:
        ok, frame = camera.read()
        capture_wall, capture_mono = time.time(), time.monotonic()
        if not ok:
            failures += 1
            if failures >= 10:
                raise RuntimeError('Camera stopped returning frames')
            time.sleep(0.1)
            continue
        failures = 0
        if frame.shape != (480, 640, 3):
            raise RuntimeError('Unexpected wrist RGB image shape: '+str(frame.shape))
        sequence += 1
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        payload = {'images': {source_key: encoded.tobytes()}, 'sequence': sequence, 'camera': identity,
                   'source_session':source_session, 'timestamps':{source_key:capture_wall},
                   'timestamp_kind':'robot_wall_capture_read_complete',
                   'capture':dict(read_complete_wall=capture_wall,read_complete_monotonic=capture_mono,
                                  hardware_exposure_timestamp=None)}
        publisher.send(msgpack.packb(payload, use_bin_type=True))
        now = time.monotonic()
        if now - last_status >= 1:
            status = dict(device=args.device, source_key=source_key, usb=identity, frames=sequence,
                          width=frame.shape[1], height=frame.shape[0],
                          fps=sequence / (now - started), pid=os.getpid(),
                          buffer_size=camera.get(cv2.CAP_PROP_BUFFERSIZE))
            status.update(pixel_format=settings['pixel_format'], configured_fps=camera.get(cv2.CAP_PROP_FPS),
                          source_session=source_session, last_frame_wall=capture_wall)
            temporary = args.status + '.tmp'
            Path(temporary).write_text(json.dumps(status, indent=2))
            os.replace(temporary, args.status)
            last_status = now
finally:
    camera.release()
    publisher.close()
    context.term()
