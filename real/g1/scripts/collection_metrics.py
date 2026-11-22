"""Bounded, receive-clock collection diagnostics; no timing acceptance policy.

A sequence gap is an observed discontinuity, not proof of a hardware drop.
Unique rates count frame-identity transitions at this subscriber/recorder; they
are not camera exposure rates and do not exclude every replay of an older ID.
"""
from collections import deque
import math


class RollingStats:
    def __init__(self, size=512):
        self.values = deque(maxlen=size)
        self.count = 0
        self.total = 0.
        self.minimum = self.maximum = self.last = None

    def observe(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Metric must be finite')
        self.values.append(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.last = value

    def summary(self):
        ordered = sorted(self.values)
        return dict(count=self.count, mean=None if not self.count else self.total/self.count,
                    min=self.minimum, max=self.maximum, last=self.last,
                    rolling_p95=None if not ordered else ordered[math.ceil(.95*len(ordered))-1],
                    rolling_samples=len(ordered))


def _sequence(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class SourceCounter:
    """Message arrivals and identity/sequence continuity at one receive boundary."""
    def __init__(self):
        self.received = self.identity_transitions = self.repeated_identities = 0
        self.sequence_gap_slots = self.sequence_regressions = self.session_changes = 0
        self.receive_clock_regressions = self.history_evictions = self.invalid_messages = 0
        self.sequence_duplicates = 0
        self.first_mono = self.last_mono = self.last_identity = None
        self.last_sequence = self.last_session = None
        self.intervals_ms = RollingStats()

    def observe(self, mono, identity=None, sequence=None, session=None):
        if not math.isfinite(mono):
            raise ValueError('Receive timestamp must be finite')
        flags = []
        previous_count = self.received
        self.received += 1
        if self.first_mono is None:
            self.first_mono = mono
        if self.last_mono is not None:
            if mono < self.last_mono:
                self.receive_clock_regressions += 1
                flags.append('receive_clock_regression')
            else:
                self.intervals_ms.observe((mono-self.last_mono)*1000)
        session_changed = (previous_count > 0 and self.last_session is not None and
                           session is not None and session != self.last_session)
        if session_changed:
            self.session_changes += 1
            flags.append('source_session_changed')
        sequence = _sequence(sequence)
        if session == self.last_session and sequence is not None and self.last_sequence is not None:
            delta = sequence-self.last_sequence
            if delta < 0:
                self.sequence_regressions += 1
                flags.append('source_sequence_regressed')
            elif delta == 0:
                self.sequence_duplicates += 1
            elif delta > 1:
                self.sequence_gap_slots += delta-1
        if identity is not None:
            if previous_count and identity == self.last_identity and not session_changed:
                self.repeated_identities += 1
            else:
                self.identity_transitions += 1
        self.last_mono, self.last_identity = mono, identity
        self.last_sequence, self.last_session = sequence, session
        return flags

    def summary(self, now=None):
        span = 0. if self.first_mono is None else max(0., self.last_mono-self.first_mono)
        return dict(received=self.received, identity_transitions=self.identity_transitions,
                    repeated_identities=self.repeated_identities,
                    effective_receive_hz=None if span <= 0 else (self.received-1)/span,
                    effective_unique_hz=None if span <= 0 or not self.identity_transitions else
                        max(0, self.identity_transitions-1)/span,
                    sequence_gap_slots=self.sequence_gap_slots,
                    sequence_regressions=self.sequence_regressions,
                    sequence_duplicates=self.sequence_duplicates, session_changes=self.session_changes,
                    receive_clock_regressions=self.receive_clock_regressions,
                    history_evictions=self.history_evictions, invalid_messages=self.invalid_messages,
                    latest_receive_age_ms=None if now is None or self.last_mono is None else
                        max(0., (now-self.last_mono)*1000),
                    receive_intervals_ms=self.intervals_ms.summary(),
                    rate_scope='subscriber_receive_clock_not_hardware_exposure_fps',
                    gap_scope='unobserved_sequence_slots_not_confirmed_hardware_drops')


class EpisodeMetrics:
    """Observe only committed samples; readiness probes do not affect quality."""
    CAMERAS = ('ego_view', 'left_wrist', 'right_wrist')

    def __init__(self, cameras=None):
        self.CAMERAS = tuple(cameras) if cameras is not None else self.CAMERAS
        if not self.CAMERAS or len(set(self.CAMERAS)) != len(self.CAMERAS):
            raise ValueError('Invalid committed camera set')
        self.frames = 0
        self.first_decision = self.last_decision = None
        self.last_ids = {}
        self.unique = {k:0 for k in self.CAMERAS}
        self.repeated = {k:0 for k in self.CAMERAS}
        self.camera_age_ms = {k:RollingStats() for k in self.CAMERAS}
        self.skew_ms = RollingStats()
        self.body_offset_ms = RollingStats()
        self.hand_offset_ms = RollingStats()
        self.intervals_ms = RollingStats()
        self.loop_ms = RollingStats()
        self.loop_overruns = 0
        self.segment_starts = deque(maxlen=256)
        self.segment_count = 0
        self.last_segment = None

    def observe(self, timing):
        decision = float(timing['decision_monotonic'])
        if not math.isfinite(decision) or (self.last_decision is not None and decision <= self.last_decision):
            raise ValueError('Committed metric decisions must increase')
        if self.first_decision is None:
            self.first_decision = decision
        if self.last_decision is not None:
            self.intervals_ms.observe((decision-self.last_decision)*1000)
        for key in self.CAMERAS:
            frame = timing['cameras'][key]
            frame_id = frame['frame_id']
            repeated = key in self.last_ids and self.last_ids[key] == frame_id
            self.repeated[key] += int(repeated)
            self.unique[key] += int(not repeated)
            self.last_ids[key] = frame_id
            self.camera_age_ms[key].observe(frame['age_ms'])
        self.skew_ms.observe(timing['camera_skew_ms'])
        self.body_offset_ms.observe(timing['body']['offset_ms'])
        self.hand_offset_ms.observe(timing['hand_state']['offset_ms'])
        segment = timing.get('sample_segment', 0)
        if segment != self.last_segment:
            self.segment_starts.append(dict(frame_index=self.frames, sample_segment=segment,
                decision_monotonic=decision, reasons=timing.get('segment_reasons', [])))
            self.segment_count += 1
            self.last_segment = segment
        self.frames += 1
        self.last_decision = decision

    def observe_loop(self, duration_s, target_period_s):
        if not math.isfinite(target_period_s) or target_period_s <= 0:
            raise ValueError('Target loop period must be positive')
        if duration_s < 0:
            raise ValueError('Loop duration must not be negative')
        self.loop_ms.observe(duration_s*1000)
        self.loop_overruns += int(duration_s > target_period_s)

    def summary(self):
        span = 0. if self.first_decision is None else self.last_decision-self.first_decision
        return dict(version=1, frames=self.frames, duration_monotonic_s=span,
            effective_sample_hz=None if span <= 0 else max(0,self.frames-1)/span,
            cameras={k:dict(identity_transitions=self.unique[k], repeated_frames=self.repeated[k],
                repeated_fraction=None if not self.frames else self.repeated[k]/self.frames,
                effective_unique_hz=None if span <= 0 else max(0,self.unique[k]-1)/span,
                age_ms=self.camera_age_ms[k].summary()) for k in self.CAMERAS},
            camera_skew_ms=self.skew_ms.summary(), body_offset_ms=self.body_offset_ms.summary(),
            hand_offset_ms=self.hand_offset_ms.summary(), sample_intervals_ms=self.intervals_ms.summary(),
            loop_ms=self.loop_ms.summary(), loop_overruns=self.loop_overruns,
            segment_count=self.segment_count,segment_starts=list(self.segment_starts),
            segment_history_truncated=self.segment_count > len(self.segment_starts),
            unique_rate_scope='committed_frame_identity_transitions_not_hardware_exposure_fps')
