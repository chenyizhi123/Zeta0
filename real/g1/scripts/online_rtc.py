"""LeRobot latency scheduling adapted to the checkpoint's absolute action slots.

The policy uses hard committed prefixes and fixed_action_grid_v1, so replacing
its sampler or retiming actions with LeRobot's generic ActionQueue would change
the trained ABI. Reuse the official latency tracker; keep the source-bound wire
validation, projection, deadlines and stale-reply rejection below it.
"""
import importlib.util
import math

import numpy as np

from online_runtime import ROOT, import_runtime

import_runtime()
from psi.deploy.real_sonic_protocol import AbsoluteActionQueue, MAX_RTC_PREFIX, PERIOD


# Import this dependency-free official module directly: importing rtc/__init__
# would also load Torch and all policy processors into the control environment.
_path = ROOT/'external_dependencies/lerobot/src/lerobot/policies/rtc/latency_tracker.py'
if not _path.is_file():
    _path = ROOT/'vendor/lerobot/latency_tracker.py'
_spec = importlib.util.spec_from_file_location('_chenyz_lerobot_latency', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
LatencyTracker = _module.LatencyTracker
RTC_SCHEDULER = 'lerobot_latency_adaptive_v1'


class RTCBudgetExceeded(ValueError):
    pass


class AdaptiveRTCTiming:
    """Historical max end-to-end latency, as in RTCInferenceEngine._rtc_loop.

The first usable chunk is warmup: it has no commitments to protect. The first
conditioned request uses the full trained capacity. Subsequent requests use
ceil(observed complete latency / period) + two publication periods, capped by
the trained capacity. A delay above that capacity cannot be hidden by clipping.
"""
    def __init__(self, capacity, margin_steps=2, *, action_hz=50):
        if type(capacity) is not int or not 1 <= capacity <= MAX_RTC_PREFIX:
            raise ValueError('Invalid trained RTC prefix capacity')
        if type(margin_steps) is not int or margin_steps < 0:
            raise ValueError('Invalid RTC scheduling margin')
        if action_hz not in (30, 50):
            raise ValueError('Unsupported RTC action frequency')
        self.period = 1. / action_hz
        self.capacity, self.margin_steps = capacity, margin_steps
        self.latency = LatencyTracker()

    def reset(self):
        self.latency.reset()

    def observe(self, seconds, *, bootstrap=False):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid end-to-end RTC latency')
        if not bootstrap:
            self.latency.add(seconds)

    def plan(self):
        seconds = self.latency.max() or 0.
        delay = math.ceil(seconds/self.period - 1e-9) if seconds else 0
        if delay > self.capacity:
            raise RTCBudgetExceeded(
                f'完整推理延迟 {seconds:.3f}s 超过模型 RTC 上限 {self.capacity*self.period:.3f}s；需重新按 f / m。')
        prefix = min(self.capacity, delay+self.margin_steps) if delay else self.capacity
        return dict(scheduler=RTC_SCHEDULER, predicted_delay_steps=delay,
                    rtc_prefix_steps=prefix, margin_steps=self.margin_steps,
                    latency_max_s=seconds, latency_samples=len(self.latency))


class RTCActionQueue(AbsoluteActionQueue):
    """Retain raw and projected actions, like LeRobot's original/processed queues.

Only the projected command queue may supply this model's prefix. These methods
have one owner (the control thread); HTTP workers cannot merge or pop actions.
"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_commands = {}
        self.consumed_steps = 0
        self.skipped_steps = 0
        self.last_merge = None

    def reset(self, session):
        super().reset(session)
        self.raw_commands.clear()
        self.consumed_steps = self.skipped_steps = 0
        self.last_merge = None

    def install(self, request, response, now):
        count = super().install(request, response, now)
        first = self.key(request['anchor_time'])+1
        prefix = len(request.get('committed_prefix', []))
        raw = np.asarray(response['actions'], dtype=np.float32)
        self.raw_commands = {
            slot: (self.raw_commands[slot] if slot < first+prefix else raw[slot-first].copy())
            for slot in self.commands
        }
        projected = np.asarray([self.commands[slot] for slot in sorted(self.commands)], dtype=np.float32)
        self.last_merge = dict(first_slot=first, prefix_steps=prefix, installed_steps=count,
                               first_pending_slot=min(self.commands), last_pending_slot=max(self.commands),
                               raw_actions=raw.copy(), projected_pending_actions=projected)
        return count

    def pop(self, scheduled_time):
        action = super().pop(scheduled_time)
        self.raw_commands = {slot: value for slot, value in self.raw_commands.items() if slot in self.commands}
        if action is not None:
            self.consumed_steps += 1
        return action

    def skip_to(self, scheduled_time, *, missed_current=False):
        skipped = super().skip_to(scheduled_time, missed_current=missed_current)
        self.raw_commands = {slot: value for slot, value in self.raw_commands.items() if slot in self.commands}
        self.skipped_steps += skipped
        return skipped
