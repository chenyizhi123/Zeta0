"""Bounded background JSONL diagnostics; never block the robot's control loop."""
import base64
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time


def audit_path(root,role):
    import os
    return Path(root)/'logs/model-audit'/f'{role}-{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}.jsonl'


class RuntimeAudit:
    def __init__(self,path,metadata=None,capacity=4096,image_interval_s=None):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=capacity)
        self.stopped=threading.Event()
        self.dropped=0
        self.error=None
        self.stream=self.path.open('a',buffering=65536)
        self.thread=threading.Thread(target=self._write,daemon=True,name='model-audit-writer')
        self.thread.start()
        self.images=None if image_interval_s is None else ObservationImages(self,image_interval_s)
        self.record('audit_start',schema='chenyz.model_audit.v1',metadata=metadata or {})

    def record(self,event,**fields):
        if self.stopped.is_set():
            self.dropped+=1
            return
        item=dict(fields,event=event,monotonic=time.monotonic(),wall_time=time.time())
        try:self.queue.put_nowait(item)
        except queue.Full:self.dropped+=1

    @staticmethod
    def encode(value):
        if hasattr(value,'tolist'):return value.tolist()
        if hasattr(value,'item'):return value.item()
        raise TypeError('Unsupported audit value: '+type(value).__name__)

    def _write(self):
        flushed=time.monotonic()
        try:
            while not self.stopped.is_set() or not self.queue.empty():
                try:item=self.queue.get(timeout=.1)
                except queue.Empty:continue
                if self.error is not None:
                    self.dropped+=1
                    continue
                try:
                    self.stream.write(json.dumps(item,default=self.encode,ensure_ascii=False,
                                                 allow_nan=False,separators=(',',':'))+'\n')
                    if time.monotonic()-flushed>=1.:
                        self.stream.flush();flushed=time.monotonic()
                except (OSError,TypeError,ValueError) as error:
                    self.error=repr(error)
                    self.dropped+=1
        finally:
            try:
                self.stream.write(json.dumps(dict(event='audit_end',monotonic=time.monotonic(),
                    dropped=self.dropped,error=self.error))+'\n')
                self.stream.close()
            except OSError:pass

    def close(self):
        if self.images:self.images.close()
        self.stopped.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive() or self.error:
            print('模型诊断日志未完整写入：'+str(self.path)+'；'+str(self.error),file=sys.stderr,flush=True)

    def __enter__(self):return self
    def __exit__(self,kind,value,traceback):
        if kind is not None:
            self.record('client_exit_reason',exception_type=kind.__name__,reason=str(value),
                        explicit_operator_exit=kind is KeyboardInterrupt)
        self.close()


class ObservationImages:
    """Save existing HTTP PNGs off-thread, with at most two pending snapshots."""
    def __init__(self,audit,interval_s):
        if not 0<interval_s<3600:raise ValueError('Invalid diagnostic image interval')
        self.audit,self.interval=audit,interval_s
        self.path=audit.path.with_suffix('')
        self.path=self.path.with_name(self.path.name+'-images')
        self.path.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=2)
        self.stopped=threading.Event()
        self.last=-float('inf')
        self.last_session=None
        self.thread=threading.Thread(target=self._write,daemon=True,name='model-observation-writer')
        self.thread.start()

    def submit(self,request):
        now=time.monotonic()
        if self.stopped.is_set() or (request['session_id']==self.last_session and now-self.last<self.interval):return
        # Called only by the HTTP worker after codecs, with immutable PNG strings.
        item=dict(session_id=request['session_id'],request_seq=request['request_seq'],
                  anchor_time=request['anchor_time'],images={key:dict(value) for key,value in request['images'].items()})
        try:self.queue.put_nowait(item)
        except queue.Full:
            self.audit.record('observation_snapshot_skipped',session_id=item['session_id'],request_seq=item['request_seq'],reason='writer busy')
        self.last=now
        self.last_session=request['session_id']

    def _save(self,item):
        safe=lambda value:re.sub(r'[^a-zA-Z0-9_-]','_',str(value))
        destination=self.path/(safe(item['session_id'])+'-'+str(item['request_seq']))
        destination.mkdir(exist_ok=True)
        files={}
        for camera,frame in item['images'].items():
            target=destination/(safe(camera)+'.png')
            # Preserve the exact already-encoded inference input, no resizing or
            # color conversions and no codec work on the control or HTTP thread.
            target.write_bytes(base64.b64decode(frame['encoded'],validate=True))
            files[camera]=str(target)
        metadata={key:item[key] for key in ('session_id','request_seq','anchor_time')}
        metadata.update(image_times={key:value['time'] for key,value in item['images'].items()},files=files)
        (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        self.audit.record('observation_snapshot',**metadata)

    def _write(self):
        while not self.stopped.is_set() or not self.queue.empty():
            try:item=self.queue.get(timeout=.1)
            except queue.Empty:continue
            try:self._save(item)
            except Exception as error:
                self.audit.record('observation_snapshot_error',session_id=item['session_id'],
                                  request_seq=item['request_seq'],reason=repr(error))

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.audit.record('observation_snapshot_error',reason='snapshot writer did not finish before shutdown')


def request_fields(request):
    return dict(session_id=request.get('session_id'),request_seq=request.get('request_seq'),
        queue_revision=request.get('queue_revision'),anchor_time=request.get('anchor_time'),
        task=request.get('task'),rtc_prefix_steps=len(request.get('committed_prefix',[])),
        committed_prefix=request.get('committed_prefix',[]),state=request.get('state'),
        snapshot_timing=request.get('snapshot_timing'),
        image_times={key:value.get('time') for key,value in request.get('images',{}).items()})


def audited_http_client(endpoint,audit):
    from psi.deploy.real_sonic_client import HttpClient
    class Client(HttpClient):
        def call(self,path,payload=None):
            started=time.monotonic()
            if path=='/infer':
                # HttpClient.infer calls this after image encoding: this is the
                # exact HTTP payload's prefix, not a configured RTC capacity.
                audit.record('http_request',**request_fields(payload))
                if audit.images:audit.images.submit(payload)
            try:result=super().call(path,payload)
            except Exception as error:
                audit.record('http_error',path=path,reason=repr(error),elapsed_s=time.monotonic()-started)
                raise
            if path=='/infer':
                audit.record('http_response',session_id=result.get('session_id'),
                    request_seq=result.get('request_seq'),rtc_prefix_steps=len(payload.get('committed_prefix',[])),
                    actions=result.get('actions'),inference_seconds=result.get('inference_seconds'),
                    elapsed_s=time.monotonic()-started,anchor_time=result.get('anchor_time'))
            return result
    return Client(endpoint)
