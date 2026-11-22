"""Optional bounded metadata JSONL journal, independent of robot commands.

This is NOT a full sensor archive: camera RGB and unselected image payloads are
omitted. It supports inspecting/replaying received body, gripper and optional
PICO telemetry into offline consumers. It cannot replay physical robot effects.
"""
import copy
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

SCHEMA_VERSION = 1
SCOPE = dict(body='decoded_received_payload', gripper='decoded_received_payload',
             pico='only_if_appended_by_collector', camera='metadata_only_no_rgb',
             full_sensor_replay=False, physical_effects_replay=False)


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('Unsupported raw telemetry value: '+type(value).__name__)


def metadata_payload(channel, payload):
    if channel != 'camera':
        return copy.deepcopy(payload)
    result = {key:copy.deepcopy(value) for key,value in payload.items() if key != 'images'}
    result['image_payloads_omitted'] = True
    result['image_payload_summary'] = {
        key:(dict(serialized_chars=len(value)) if isinstance(value,str) else dict(encoded_bytes=len(value)))
        if isinstance(value,(str,bytes,bytearray)) else
            dict(shape=list(getattr(value,'shape',())),dtype=str(getattr(value,'dtype','unknown')))
        for key,value in payload.get('images',{}).items()}
    return result


class RawTelemetryArchive:
    def __init__(self, path, max_queue=1024):
        if not isinstance(max_queue,int) or max_queue < 1:
            raise ValueError('Archive queue must be bounded and positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.output = self.path.open('x',encoding='utf-8')
        self.queue = Queue(maxsize=max_queue)
        self.lock = threading.Lock()
        self.accepted = self.written = self.dropped = 0
        self.channels = {}
        self.error = None
        self.closed = False
        self.stop = threading.Event()
        try:
            self._write(dict(type='header',version=SCHEMA_VERSION,scope=dict(SCOPE),
                             started_wall=time.time(),started_monotonic=time.monotonic()))
        except BaseException:
            self.output.close()
            raise
        self.thread = threading.Thread(target=self._write_loop,name='telemetry-archive',daemon=True)
        self.thread.start()

    def _write(self, value):
        self.output.write(json.dumps(value,default=_json_default,allow_nan=False,separators=(',',':'))+'\n')

    def append(self, channel, payload, mono=None, wall=None):
        if not isinstance(channel,str) or not channel:
            raise ValueError('Archive channel required')
        mono = time.monotonic() if mono is None else float(mono)
        wall = time.time() if wall is None else float(wall)
        if not math.isfinite(mono) or not math.isfinite(wall):
            raise ValueError('Archive receive timestamps must be finite')
        value = dict(type='packet',version=SCHEMA_VERSION,channel=channel,
                     received_monotonic=mono,received_wall=wall,
                     payload=metadata_payload(channel,payload))
        with self.lock:
            if self.closed:
                raise RuntimeError('Archive is closed')
            if self.error:
                return False
            value['archive_sequence'] = self.accepted
            try:
                self.queue.put_nowait(value)
            except Full:
                self.dropped += 1
                return False
            self.accepted += 1
            self.channels[channel] = self.channels.get(channel,0)+1
        return True

    def _write_loop(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    value = self.queue.get(timeout=.05)
                except Empty:
                    continue
                self._write(value)
                with self.lock:
                    self.written += 1
                self.queue.task_done()
            self._write(dict(type='trailer',**self.status()))
            self.output.flush()
        except BaseException as error:
            with self.lock:
                self.error = type(error).__name__+': '+str(error)
        finally:
            self.output.close()

    def status(self):
        with self.lock:
            return dict(path=str(self.path),version=SCHEMA_VERSION,scope=dict(SCOPE),
                accepted=self.accepted,written=self.written,dropped=self.dropped,
                pending=self.queue.qsize(),channels=dict(self.channels),error=self.error,
                closed=self.closed,complete=self.closed and self.error is None and
                    not self.dropped and self.written == self.accepted)

    def close(self, timeout=3):
        with self.lock:
            self.closed = True
        self.stop.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError('Raw telemetry writer did not stop within timeout')
        return self.status()


def iter_archive(path, channels=None):
    """Yield decoded packet events in journal order without starting any network.

Receive clocks are preserved, not converted to robot clock or video time. A
journal with missing trailer, writer failure or dropped entries is rejected.
"""
    allowed = None if channels is None else set(channels)
    path = Path(path)
    # First validate terminal completeness before exposing events to consumers.
    with path.open(encoding='utf-8') as stream:
        first = stream.readline()
        try:
            header = json.loads(first)
        except ValueError as error:
            raise ValueError('Invalid telemetry archive header') from error
        if header.get('type') != 'header' or header.get('version') != SCHEMA_VERSION:
            raise ValueError('Unsupported telemetry archive')
        trailer = None
        packets = 0
        for line in stream:
            value = json.loads(line)
            if value.get('type') == 'packet':
                packets += 1
                trailer = None
            elif value.get('type') == 'trailer':
                trailer = value
            else:
                raise ValueError('Unknown telemetry archive record')
        if (trailer is None or not trailer.get('complete') or trailer.get('dropped') or
                trailer.get('error') or trailer.get('written') != packets):
            raise ValueError('Incomplete telemetry archive; inspect diagnostics instead of replaying')
    with path.open(encoding='utf-8') as stream:
        next(stream)
        for line in stream:
            value = json.loads(line)
            if value.get('type') == 'packet' and (allowed is None or value['channel'] in allowed):
                yield value


def replay_archive(path, consumer, channels=None):
    """Feed events to an offline callback; no scheduling, DDS or robot replay."""
    count = 0
    for event in iter_archive(path,channels):
        consumer(event)
        count += 1
    return count
