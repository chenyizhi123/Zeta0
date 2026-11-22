"""Retired bounded video writer, retained for historical dataset tooling.

The exporter adapter reserves capacity for the complete camera bundle before
any queue receives it. Encoder failure invalidates the episode; it never drops
one camera independently. Only a joined, flushed writer is a saved video.
"""
from contextlib import ExitStack
from pathlib import Path
import queue
import threading
import time


class VideoBackpressureError(RuntimeError):
    """No complete RGB bundle could be accepted before the admission deadline."""


class EpisodeVideoWriter:
    def __init__(self, output_path, width, height, fps, codec='h264', buffer_size=50,
                 admission_timeout_s=.25, stop_timeout_s=30.):
        import av
        if buffer_size < 1 or admission_timeout_s <= 0 or stop_timeout_s <= 0:
            raise ValueError('Invalid video queue or timeout')
        self.av = av
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.container = av.open(str(self.output_path), mode='w')
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width, self.stream.height = width, height
        self.queue = queue.Queue(maxsize=buffer_size)
        self._slots = threading.BoundedSemaphore(buffer_size)
        self._lock = threading.RLock()
        self.admission_timeout_s = float(admission_timeout_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.error = None
        self.closed = False
        self._stop_requested = False
        self._capture = None
        self._metrics = dict(accepted_frames=0, encoded_frames=0, queue_high_watermark=0,
            admission_wait_s=0., admission_timeouts=0, encoding_s=0.,
            encoding_max_s=0., flush_s=0., worker_failed=False)
        self.worker = threading.Thread(target=self._run, daemon=True,
                                       name='episode-video-'+self.output_path.stem)
        self.worker.start()

    def _run(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is not None:
                    self._slots.release()
                try:
                    if frame is None:
                        if self.error is None:
                            began = time.monotonic()
                            for packet in self.stream.encode():
                                self.container.mux(packet)
                            with self._lock:
                                self._metrics['flush_s'] += time.monotonic()-began
                        return
                    if self.error is None:
                        began = time.monotonic()
                        video_frame = self.av.VideoFrame.from_ndarray(frame, format='rgb24')
                        for packet in self.stream.encode(video_frame):
                            self.container.mux(packet)
                        elapsed = time.monotonic()-began
                        with self._lock:
                            self._metrics['encoded_frames'] += 1
                            self._metrics['encoding_s'] += elapsed
                            self._metrics['encoding_max_s'] = max(
                                self._metrics['encoding_max_s'], elapsed)
                except BaseException as error:
                    with self._lock:
                        self.error = error
                        self._metrics['worker_failed'] = True
                finally:
                    self.queue.task_done()
        finally:
            try:
                self.container.close()
            except BaseException as error:
                with self._lock:
                    self.error = self.error or error
                    self._metrics['worker_failed'] = True

    def _validate(self, frame):
        if self.closed or self.error or not self.worker.is_alive():
            raise RuntimeError('视频编码器不可用') from self.error
        if frame.shape != (self.height, self.width, 3):
            raise ValueError('相机画面尺寸与数据集不同')

    def add_frame(self, frame):
        self._validate(frame)
        if self._capture is not None:
            if self._capture:
                raise RuntimeError('同一数据行重复提交相机画面')
            self._capture.append(frame.copy())
            return
        submit_frame_bundle({str(self.output_path): self},
                            {str(self.output_path): frame}, self.admission_timeout_s)

    def metrics_snapshot(self):
        with self._lock:
            return dict(self._metrics, queue_depth=self.queue.qsize(),
                        queue_capacity=self.queue.maxsize, closed=self.closed,
                        worker_alive=self.worker.is_alive())

    def stop(self):
        deadline = time.monotonic()+self.stop_timeout_s
        with self._lock:
            self.closed = True
            enqueue_stop = not self._stop_requested
            self._stop_requested = True
        if enqueue_stop:
            try:
                self.queue.put(None, timeout=max(.001, deadline-time.monotonic()))
            except queue.Full as error:
                with self._lock:
                    self._stop_requested = False
                raise VideoBackpressureError('视频队列未能在停止时限内排空；未提交本条') from error
        self.worker.join(max(0., deadline-time.monotonic()))
        if self.worker.is_alive():
            # Never close/mux a container concurrently with an encoding worker.
            raise VideoBackpressureError('视频编码线程停止超时；未提交本条')
        if self.error:
            raise RuntimeError('视频编码失败') from self.error
        if self._metrics['accepted_frames'] != self._metrics['encoded_frames']:
            raise RuntimeError('视频接纳帧数与编码帧数不一致')
        return str(self.output_path)

    def cancel(self):
        try:
            self.stop()
        finally:
            if not self.worker.is_alive():
                self.output_path.unlink(missing_ok=True)


def submit_frame_bundle(writers, frames, timeout_s=.25, copy_frames=True):
    """Reserve every queue first. Saturation rejects the whole row, never one view."""
    if set(writers) != set(frames) or not writers:
        raise ValueError('RGB bundle does not match all video writers')
    copies = {}
    for key, writer in writers.items():
        writer._validate(frames[key])
        copies[key] = frames[key].copy() if copy_frames else frames[key]
    began = time.monotonic()
    deadline = began+timeout_s
    reserved = []
    try:
        for key, writer in writers.items():
            if not writer._slots.acquire(timeout=max(0., deadline-time.monotonic())):
                with writer._lock:
                    writer._metrics['admission_timeouts'] += 1
                raise VideoBackpressureError('RGB 编码队列拥堵；整帧未接纳')
            reserved.append(writer)
        with ExitStack() as stack:
            for writer in sorted(writers.values(), key=id):
                stack.enter_context(writer._lock)
            for key, writer in writers.items():
                writer._validate(copies[key])
            for key, writer in writers.items():
                writer.queue.put_nowait(copies[key])
                writer._metrics['accepted_frames'] += 1
                writer._metrics['queue_high_watermark'] = max(
                    writer._metrics['queue_high_watermark'], writer.queue.qsize())
                writer._metrics['admission_wait_s'] += time.monotonic()-began
        reserved.clear()
    finally:
        for writer in reserved:
            writer._slots.release()


def install_atomic_frame_admission(exporter, timeout_s=.25):
    """Adapt upstream per-view add_frame calls to one atomic camera bundle."""
    if getattr(exporter, '_chenyz_atomic_video_admission', False):
        return exporter
    original = exporter.add_frame

    def add_frame(frame):
        writers = exporter.video_writers
        if any(not isinstance(writer, EpisodeVideoWriter) for writer in writers.values()):
            raise TypeError('Atomic admission requires EpisodeVideoWriter for every view')
        buffer = exporter.episode_buffer
        old_size = buffer['size']
        lengths = {key: len(value) for key, value in buffer.items() if isinstance(value, list)}
        try:
            for writer in writers.values():
                writer._capture = []
            original(frame)
            if any(len(writer._capture) != 1 for writer in writers.values()):
                raise ValueError('数据行必须包含每路 RGB 的一张画面')
            captures = {key: writer._capture[0] for key, writer in writers.items()}
            # Captures already own their arrays; avoid copying three RGB views
            # a second time solely to reserve queue slots.
            submit_frame_bundle(writers, captures, timeout_s, copy_frames=False)
        except BaseException:
            for key, length in lengths.items():
                del buffer[key][length:]
            buffer['size'] = old_size
            raise
        finally:
            for writer in writers.values():
                writer._capture = None

    exporter.add_frame = add_frame
    exporter._chenyz_atomic_video_admission = True
    return exporter
