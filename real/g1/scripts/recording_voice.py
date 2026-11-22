"""Asynchronous G1 speech retaining outcomes and the latest ordinary status."""
from collections import deque
from pathlib import Path
import subprocess
import threading


class RecordingVoice:
    def __init__(self, local, enabled=True):
        python = Path(local['sonic_root']) / '.venv_sim/bin/python' if local.get('sonic_root') else None
        self.command = ([str(python), str(Path(__file__).with_name('g1_say.py')),
                         '--interface', local['robot_interface']]
                        if enabled and python and python.is_file() and local.get('robot_interface') else None)
        self.pending = deque()
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.worker = None
        if self.command:
            self.worker = threading.Thread(target=self._run, name='recording-voice', daemon=True)
            self.worker.start()
        elif enabled:
            print('G1 语音提示不可用：缺少 SDK Python 环境；录制仍可正常使用。', flush=True)

    def say(self, text, *, preserve=False):
        if not self.command:
            return
        with self.condition:
            if self.stopped.is_set():
                return
            # Outcomes stay in order; replace only obsolete ordinary statuses.
            # Enqueueing never waits for audio or delays frame capture.
            self.pending = deque(item for item in self.pending if item[1])
            self.pending.append((text, preserve))
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.stopped.is_set())
                if not self.pending:
                    return
                text, _ = self.pending.popleft()
            try:
                subprocess.run([*self.command, text], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               check=True, timeout=12)
            except (OSError, subprocess.SubprocessError) as error:
                print('G1 语音播放失败，录制继续：' + str(error), flush=True)

    def close(self):
        with self.condition:
            self.stopped.set()
            remaining=len(self.pending)
            self.condition.notify_all()
        if self.worker:
            # Allow each queued RPC its existing 12-second timeout, including
            # the call already in flight. This confirms sending, not playback.
            self.worker.join(timeout=12*(remaining+1)+1)
            if self.worker.is_alive():
                print('部分语音尚未发送完成，请以采集终端记录为准。',flush=True)


def announce(collector, text):
    voice = getattr(collector, 'recording_voice', None)
    if voice is not None:
        voice.say(text)
