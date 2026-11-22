"""Camera-only subprocess recording one three-camera clip per accepted M command.

The control thread enqueues small mode messages. JPEG decoding, video encoding,
filesystem work and camera subscriptions belong exclusively to the subprocess.
"""
import base64
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time

CAMERAS = ('head', 'left_wrist', 'right_wrist')


def camera_sources(config):
    reverse = {label: key for key, label in config['wrist_mapping'].items()}
    keys = ('ego_view', reverse['left_wrist'], reverse['right_wrist'])
    return {name: dict(config['camera_sources'][key]) for name, key in zip(CAMERAS, keys)}


class VideoSegment:
    """30fps common timeline; missing images are explicit black frames."""
    def __init__(self, directory, event, fps=30):
        import cv2
        self.directory, self.event, self.fps = Path(directory), event, fps
        self.start = event['monotonic']
        self.frames = 0
        self.missing = {name: 0 for name in CAMERAS}
        self.writers = {}
        self.timeline = None
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            for name in CAMERAS:
                path = self.directory/(name+'.partial.mp4')
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640,480))
                self.writers[name] = writer
                if not writer.isOpened():
                    raise RuntimeError('Cannot open video encoder: '+str(path))
            self.timeline = (self.directory/'frames.jsonl').open('w', buffering=65536)
            self._metadata('recording')
        except BaseException:
            for writer in self.writers.values():writer.release()
            if self.timeline:self.timeline.close()
            raise

    def _metadata(self, status, **extra):
        value = dict(status=status, trigger=self.event, fps=self.fps, frames=self.frames,
            duration_s=self.frames/self.fps, missing_frames=self.missing,
            recording='independent camera streams; not exact inference-input snapshots', **extra)
        path = self.directory/'metadata.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        temp.replace(path)

    def write_until(self, end, latest):
        import cv2
        import numpy as np
        # Frame i represents [start+i/fps, start+(i+1)/fps). Do not accelerate
        # playback if a camera pauses; pad the common timeline explicitly.
        due = max(0, math.ceil((end-self.start)*self.fps-1e-8))
        if due-self.frames > self.fps:
            raise RuntimeError('Video encoder fell over one second behind; recording stopped')
        while self.frames < due:
            stamp = self.start+self.frames/self.fps
            receipts, freshness = {}, {}
            for name, writer in self.writers.items():
                history = latest.get(name, ())
                sample = next((s for s in reversed(history) if s[0] <= stamp+1e-9), None)
                fresh = sample is not None and stamp-sample[0] <= .2
                receipts[name] = sample[0] if sample is not None else None
                freshness[name] = fresh
                if fresh:
                    image = sample[1]
                else:
                    image = np.zeros((480,640,3), np.uint8)
                    cv2.putText(image, name+' - NO FRESH FRAME', (15,240),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
                    self.missing[name] += 1
                writer.write(image)
            self.timeline.write(json.dumps(dict(frame_index=self.frames, scheduled_monotonic=stamp,
                received_monotonic=receipts, fresh=freshness))+'\n')
            self.frames += 1

    def close(self, reason, error=None):
        for writer in self.writers.values():writer.release()
        if self.timeline:self.timeline.close()
        for name in self.writers:
            path = self.directory/(name+'.partial.mp4')
            if path.exists():path.rename(self.directory/(name+'.mp4'))
        self._metadata('error' if error else 'finished', stop_reason=reason, error=error)


def record_worker(sources, directory, commands, statuses, abort, parent_pid):
    import cv2
    import msgpack
    import numpy as np
    import zmq
    from collections import deque
    cv2.setNumThreads(1)
    signal.signal(signal.SIGINT, lambda *_: abort.set())
    signal.signal(signal.SIGTERM, lambda *_: abort.set())
    context, poller = zmq.Context(), zmq.Poller()
    sockets, latest, identities = {}, {}, {}
    segment = None
    alive = True

    def report(event, **fields):
        try:statuses.put_nowait(dict(event=event, **fields))
        except queue.Full:pass

    def finish(reason, at=None, error=None):
        nonlocal segment
        if segment is None:return
        old, segment = segment, None
        try:
            if at is not None:old.write_until(at, latest)
        except Exception as exc:error = repr(exc)
        try:old.close(reason, error)
        except Exception as exc:error = repr(exc)
        report('model_video_error' if error else 'model_video_saved', directory=str(old.directory),
            frames=old.frames, missing_frames=old.missing, reason=reason, error=error)

    try:
        for name, source in sources.items():
            sock = context.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE,b'')
            sock.setsockopt(zmq.CONFLATE,1)
            sock.setsockopt(zmq.LINGER,0)
            sock.connect(f"tcp://{source['host']}:{source['port']}")
            poller.register(sock, zmq.POLLIN)
            sockets[sock] = name
        report('model_video_ready', directory=directory)
        while alive and not abort.is_set() and os.getppid() == parent_pid:
            # Process mode transitions before recording additional frames.
            for _ in range(64):
                try:command=commands.get_nowait()
                except queue.Empty:break
                finish(command['kind'], command['monotonic'])
                if command['kind']=='start':
                    try:
                        target=Path(directory)/f"segment-{command['segment']:04d}"
                        segment=VideoSegment(target, command)
                        report('model_video_started', directory=str(target), segment=command['segment'])
                    except Exception as exc:
                        report('model_video_error',directory=directory,error=repr(exc))
                elif command['kind']=='close':
                    alive=False
                    break
            if not alive:break
            for sock in dict(poller.poll(5)):
                name=sockets[sock];source=sources[name]
                try:
                    payload=msgpack.unpackb(sock.recv(),raw=False)
                    received=time.monotonic()
                    if source.get('usb_serial') and payload.get('camera',{}).get('serial')!=source['usb_serial']:
                        continue
                    raw=payload['images'][source['key']]
                    if isinstance(raw,str):raw=base64.b64decode(raw)
                    identity=payload.get('frame_ids',{}).get(source['key'],
                        payload.get('timestamps',{}).get(source['key']))
                    if identity is None:identity=hashlib.sha256(raw).hexdigest()
                    identity=(payload.get('source_session'),identity)
                    if identities.get(name)==identity:continue
                    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape!=(480,640,3):continue
                    identities[name]=identity
                    latest.setdefault(name,deque(maxlen=12)).append((received,image))
                except (KeyError,TypeError,ValueError,msgpack.UnpackException):continue
            if segment is not None:
                try:segment.write_until(time.monotonic(),latest)
                except Exception as exc:
                    finish('encoder_error',error=repr(exc))
    except Exception as exc:
        report('model_video_error',directory=directory,error=repr(exc))
        finish('worker_error',error=repr(exc))
    finally:
        finish('worker_exit',time.monotonic())
        for sock in sockets:sock.close(linger=0)
        context.term()


class ModelVideoRecorder:
    def __init__(self, config, directory, audit=None):
        self.directory, self.audit = Path(directory), audit
        self.segment, self.active, self.failed = 0, False, False
        self.closed = False
        self.ready=threading.Event()
        context=mp.get_context('spawn')
        self.commands=context.Queue(maxsize=64)
        self.statuses=context.Queue(maxsize=128)
        self.abort=context.Event()
        self.process=context.Process(target=record_worker,
            args=(camera_sources(config),str(self.directory),self.commands,self.statuses,self.abort,os.getpid()),
            name='model-camera-recorder',daemon=True)
        self.process.start()
        self.reader_stop=threading.Event()
        self.reader=threading.Thread(target=self._read_status,daemon=True,name='video-status')
        self.reader.start()

    def _report(self, event):
        value=dict(event)
        kind=value.pop('event')
        if kind=='model_video_ready':self.ready.set()
        if self.audit:self.audit.record(kind,**value)
        if kind=='model_video_started':print('三路录像开始：'+value['directory'],flush=True)
        elif kind=='model_video_saved':print('三路录像已保存：'+value['directory'],flush=True)
        elif kind=='model_video_error':print('三路录像异常：'+str(value.get('error')),flush=True)

    def _read_status(self):
        while not self.reader_stop.is_set():
            try:self._report(self.statuses.get(timeout=.1))
            except queue.Empty:
                if self.process.exitcode is not None:
                    if not self.closed:
                        self.failed=True
                        self._report(dict(event='model_video_error',error='camera recorder exited'))
                    break

    def on_event(self, event):
        if self.closed or self.failed or event.get('event')!='mode':return
        start=event.get('key')=='m'
        if not start and not self.active:return
        if start:self.segment+=1
        message=dict(kind='start' if start else 'stop',segment=self.segment,monotonic=time.monotonic(),
            wall_time=time.time(),generation=event.get('generation'),session_id=event.get('session_id'))
        try:self.commands.put_nowait(message)
        except queue.Full:
            self.failed=True
            self.abort.set()
            self._report(dict(event='model_video_error',error='recorder command queue full'))
        else:self.active=start

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.commands.put(dict(kind='close',monotonic=time.monotonic()),timeout=.2)
        except queue.Full:self.abort.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.abort.set()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()  # Only our camera encoder child; never a robot service.
            self.process.join(timeout=1)
            self._report(dict(event='model_video_error',error='encoder did not finalize; partial MP4 may need recovery'))
        self.reader_stop.set()
        self.reader.join(timeout=.5)
        self.commands.cancel_join_thread()
        self.commands.close()
        self.statuses.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
