"""Progress and G1 speaker output around the unchanged official recording loop."""
from contextlib import contextmanager
from functools import wraps
import math
import re
import threading
import time

from recording_voice import RecordingVoice


def task_speech_label(task, prompt):
    normalized=' '.join(prompt.strip().rstrip('.。').lower().split())
    names={
        'tidy up the table and put the trash into the bin':'收拾桌面',
        'place the chair in front of the table':'摆放凳子',
        'place the doll on the table':'把玩偶放到桌上',
    }
    return names.get(normalized,prompt.strip().rstrip('.。') or task)


class RecordingSpeech:
    def __init__(self, voice, task_label=''):
        self.voice = voice
        self.saved_reset_notice=False

    def say(self, text, blocking=False):
        match = re.fullmatch(r'Recording episode (\d+)', text) or re.fullmatch(r'第(\d+)条开始录制。',text)
        if match:
            self.saved_reset_notice=False
            translated = f'第{int(match[1])}条开始录制。'
        elif (match:=re.match(r'第(\d+)条已保存',text)):
            translated=f'第{int(match[1])}条已保存，请复位。'
            self.saved_reset_notice=True
        elif text.startswith('正在保存'):
            translated='保存中。'
        elif text.startswith('请人工复位') or text=='Reset the environment':
            if self.saved_reset_notice:
                self.saved_reset_notice=False
                return  # The successful-save message already asks for reset.
            translated='请复位，确认后再开始。'
        elif (match:=re.match(r'第(\d+)条已就绪',text)):
            self.saved_reset_notice=False
            translated=f'第{int(match[1])}条就绪。'
        elif text.startswith('本段已丢弃') or text=='Re-record episode':
            self.saved_reset_notice=False
            translated='已丢弃，请复位。' if '请人工复位' in text else '已丢弃。'
        else:
            translated = {
                'Stop recording': '采集结束，正在收尾。',
                'Exiting': '采集程序已退出。',
            }.get(text, text)
        preserve = (translated.startswith(('保存中','已丢弃','请复位','当前采集任务','开始录制',
                    '保存未完成','录制中断','丢弃失败','采集异常','本次采集结束',
                    '退出收尾失败','已到设定时长')) or
                    re.match(r'第\d+条(?:已保存|开始录制)', translated) is not None)
        if preserve:
            self.voice.say(translated, preserve=True)
        else:
            self.voice.say(translated)


@contextmanager
def recording_feedback(local, *, voice_enabled=True, progress_enabled=True, task_label=''):
    """Delegate all recording to LeRobot; observe its actual episode buffer.

    The wrapper adds no frames, timers controlling an episode, save operations,
    key handlers, image selection or robot commands. Official say() is routed
    to the robot speaker instead of the workstation's spd-say process.
    """
    from lerobot.scripts import lerobot_record
    from lerobot.utils import utils

    original_loop, original_say = lerobot_record.record_loop, utils.say
    voice = RecordingVoice(local, enabled=voice_enabled)
    speech = RecordingSpeech(voice,task_label=task_label)

    @wraps(original_loop)
    def observed_loop(*args, **kwargs):
        # LeRobot's safe_stop_image_writer decorator exposes (*args, **kwargs),
        # so signature introspection cannot recover the wrapped loop fields.
        dataset = kwargs.get('dataset', args[6] if len(args) > 6 else None)
        limit = kwargs.get('control_time_s', args[8] if len(args) > 8 else None)
        episode = dataset.num_episodes if dataset is not None else None
        started = time.monotonic()
        stopped = threading.Event()
        previous = [started, 0]

        def report():
            now = time.monotonic()
            elapsed = now - started
            duration = f'{elapsed:.1f}s' + (f' / {limit:g}s' if limit is not None and math.isfinite(limit) else '')
            if dataset is None:
                text = f'复位中：{duration}；按 → 结束复位。'
            else:
                frames = dataset.writer.episode_buffer['size']
                delta = now - previous[0]
                hz = (frames - previous[1]) / delta if delta > 0 else 0
                text = f'录制中 episode {episode}：{frames} 帧 | {hz:.1f}Hz | {duration}'
                previous[:] = [now, frames]
            print(text, flush=True)

        def monitor():
            while not stopped.wait(1):
                try:
                    report()
                except (OSError, KeyError, AttributeError):
                    return  # An optional status display never stops recording.

        thread = None
        if progress_enabled:
            thread = threading.Thread(target=monitor, name='lerobot-progress', daemon=True)
            thread.start()
        completed = False
        try:
            result = original_loop(*args, **kwargs)
            completed = True
            return result
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=2)
            if progress_enabled and completed:
                try:
                    report()
                    if dataset is not None:
                        print(f'episode {episode} 录制阶段结束；后续复位、保存或重录由 LeRobot 处理。',flush=True)
                except (OSError, KeyError, AttributeError):
                    pass

    try:
        if voice_enabled:
            utils.say = speech.say
        if progress_enabled:
            lerobot_record.record_loop = observed_loop
        if task_label:
            # Task descriptions may be long or unsupported by the speaker's
            # language. Keep them in the terminal, out of the speech queue.
            utils.log_say(f'当前采集任务：{task_label}。',False)
        yield
    finally:
        lerobot_record.record_loop = original_loop
        utils.say = original_say
        voice.close()
