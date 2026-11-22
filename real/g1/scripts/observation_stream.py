"""Read-only live adapter: one causal snapshot feeds collection/inference."""
import base64
import threading
import time

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from brainco_gripper_data import MissingGripperData
from observation_sync import CameraPairer, ObservationAssembler
from collection_metrics import RollingStats
from raw_telemetry_archive import RawTelemetryArchive


class ObservationStream:
    def __init__(self, config):
        self.config = config
        self.assembler = ObservationAssembler(config)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.decode_id = None
        self.decoded = None
        self.parse_ms = {k:RollingStats() for k in ('camera','body','gripper')}
        self.decode_ms = RollingStats()
        self.archive = None
        self.last_archive_status = None
        self.thread = threading.Thread(target=self._receive,name='observation-receiver',daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or self.error:
            self.close()
            raise RuntimeError('Observation receiver failed: '+str(self.error))

    def _receive(self):
        context = zmq.Context()
        sockets = {}
        poller = zmq.Poller()
        endpoints = dict(body=f"tcp://127.0.0.1:{self.config.get('state_port',5557)}",
                         gripper=self.config.get('gripper_endpoint','tcp://127.0.0.1:5563'))
        head_source = None
        pairer = None
        if self.assembler.camera_keys == ('ego_view',):
            head_source = self.config['camera_sources']['ego_view']
            endpoints['camera'] = f"tcp://{head_source['host']}:{head_source['port']}"
            pairer = CameraPairer(self.config)
        else:
            endpoints['camera'] = f"tcp://127.0.0.1:{self.config['camera_output_port']}"
        try:
            for kind, endpoint in endpoints.items():
                sock = context.socket(zmq.SUB)
                sock.setsockopt(zmq.SUBSCRIBE,b'g1_debug' if kind == 'body' else b'')
                sock.setsockopt(zmq.RCVHWM,128 if kind == 'body' else 8)
                sock.setsockopt(zmq.LINGER,0)
                sock.connect(endpoint)
                poller.register(sock,zmq.POLLIN)
                sockets[sock] = kind
            self.ready.set()
            while not self.stop.is_set():
                for sock in dict(poller.poll(2)):
                    for _ in range(128):
                        try:
                            raw = sock.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        mono, wall = time.monotonic(), time.time()
                        kind = sockets[sock]
                        parse_started = time.monotonic()
                        try:
                            if kind == 'gripper':
                                import json
                                payload = json.loads(raw)
                            else:
                                if kind == 'body':
                                    raw = raw[len(b'g1_debug'):]
                                payload = msgpack.unpackb(raw,raw=False,object_hook=msgpack_numpy.decode)
                            if not isinstance(payload,dict):
                                with self.lock:
                                    self.assembler.ingress_metrics[kind].invalid_messages += 1
                                continue
                            with self.lock:
                                if kind == 'camera' and pairer is not None:
                                    # Normalize only the selected raw head stream
                                    # into the same receive-time contract. No wrist
                                    # source or three-view composer is involved.
                                    pairer.push('ego_view',head_source['key'],payload,mono,wall)
                                    for group in pairer.ready(mono):
                                        self.assembler.push(kind,group,mono,wall)
                                else:
                                    self.assembler.push(kind,payload,mono,wall)
                                self.parse_ms[kind].observe((time.monotonic()-parse_started)*1000)
                                if self.archive is not None:
                                    self.archive.append(kind,payload,mono,wall)
                        except (ValueError,TypeError,KeyError):
                            with self.lock:
                                self.assembler.ingress_metrics[kind].invalid_messages += 1
                            continue
        except BaseException as error:
            self.error = error
            self.ready.set()
        finally:
            for sock in sockets:
                sock.close(linger=0)
            context.term()

    def snapshot(self):
        if self.error:
            raise MissingGripperData('观察接收线程失败: '+str(self.error))
        with self.lock:
            if self.archive is not None:
                status = self.archive.status()
                if status['error'] or status['dropped']:
                    raise MissingGripperData('原始遥测归档不完整: '+str(status))
            snapshot = self.assembler.snapshot()
        proprio = snapshot['proprio']
        for key, size in (('body_q',29),('last_action',29),('left_hand_q',7),('right_hand_q',7),
                          ('last_left_hand_action',7),('last_right_hand_action',7),
                          ('base_quat',4),('token_state',64)):
            value = np.asarray(proprio.get(key,[]))
            if value.shape != (size,) or not np.isfinite(value).all():
                raise MissingGripperData('身体状态/动作无效: '+key)
        message = snapshot['image_message']
        decode_id = (message['composer_session'],message['sequence'])
        if decode_id != self.decode_id:
            decode_started = time.monotonic()
            images = {}
            for key in self.assembler.camera_keys:
                value = message['images'][key]
                if isinstance(value,str):
                    value = base64.b64decode(value)
                if isinstance(value,(bytes,bytearray)):
                    bgr = cv2.imdecode(np.frombuffer(value,np.uint8),cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise MissingGripperData('图像解码失败: '+key)
                    value = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                if (not isinstance(value,np.ndarray) or value.dtype != np.uint8 or
                    list(value.shape) != self.assembler.contract['image_shape']):
                    raise MissingGripperData('RGB 图像尺寸/类型不符合观察接口: '+key)
                images[key] = value
            self.decode_id, self.decoded = decode_id, images
            with self.lock:
                self.decode_ms.observe((time.monotonic()-decode_started)*1000)
        # A policy may normalize/crop inputs in place. Never let it mutate the
        # cached frame used by subsequent decisions or recording consumers.
        images = {k:v.copy() for k,v in self.decoded.items()}
        snapshot['image_message'] = {**message,'images':images}
        snapshot['model_input'] = {
            'observation.state':np.asarray(proprio['body_q'],dtype=np.float32).copy(),
            'observation.gripper':np.asarray(snapshot['packet']['state'],dtype=np.float32),
            **{'observation.images.'+k:images[k] for k in self.assembler.camera_keys},
        }
        return snapshot

    def commit(self, snapshot):
        with self.lock:
            return self.assembler.commit(snapshot)

    def reset_episode(self):
        with self.lock:
            self.assembler.reset_episode()

    def metrics(self):
        with self.lock:
            return {**self.assembler.metrics(),
                'parse_ms':{k:v.summary() for k,v in self.parse_ms.items()},
                'rgb_decode_ms':self.decode_ms.summary(),
                'receiver_error':None if self.error is None else str(self.error),
                'archive':self.archive.status() if self.archive is not None else self.last_archive_status}

    def start_archive(self, path):
        with self.lock:
            if self.archive is not None:
                raise RuntimeError('A telemetry archive is already active')
            self.archive = RawTelemetryArchive(path,
                max_queue=self.config.get('raw_archive_queue_size',1024))
            self.last_archive_status = None
            return self.archive.status()

    def append_archive(self, channel, payload, mono=None, wall=None):
        with self.lock:
            return False if self.archive is None else self.archive.append(channel,payload,mono,wall)

    def stop_archive(self):
        with self.lock:
            archive, self.archive = self.archive, None
        if archive is not None:
            self.last_archive_status = archive.close()
        return self.last_archive_status

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Observation receiver did not stop')
        self.stop_archive()
