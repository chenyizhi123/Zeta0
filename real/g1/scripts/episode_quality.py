"""Episode lifecycle and training admission. No robot or network APIs.

Storage validation never implies task completion. Fixed-fps archives containing
time gaps remain useful diagnostic records but are excluded from demonstrations.
"""
import copy
import math


QUALITY_VERSION = 2


class EpisodeLifecycle:
    def __init__(self, task_success=None):
        if task_success not in (None, True, False):
            raise ValueError('Task success must be true, false, or unknown')
        self.task_success = task_success
        self.end_reason = None
        self.gaps = []
        self.segments = []
        self.runtime_capture = None
        self.success_label_source = 'unknown'
        self.runtime_sample_mismatches = []
        self.freeze_reason = None
        self.sampling_period_s = None
        self.first_decision_monotonic = None
        self.max_nominal_timeline_error_s = 0.
        self.max_nominal_timeline_error_frame = None
        self._segment = None
        self._last_time = None

    def frame(self, index, decision_monotonic=None, expected_period=None):
        if decision_monotonic is not None and not math.isfinite(decision_monotonic):
            raise ValueError('Invalid frame decision time')
        if expected_period is not None:
            if not math.isfinite(expected_period) or expected_period <= 0:
                raise ValueError('Invalid sampling period')
            self.sampling_period_s=expected_period
        if decision_monotonic is not None:
            if self.first_decision_monotonic is None:
                self.first_decision_monotonic=decision_monotonic
            if expected_period is not None:
                error=abs(decision_monotonic-self.first_decision_monotonic-index*expected_period)
                if error > self.max_nominal_timeline_error_s:
                    self.max_nominal_timeline_error_s=error
                    self.max_nominal_timeline_error_frame=index
        if (decision_monotonic is not None and self._last_time is not None and
            expected_period is not None and
            decision_monotonic - self._last_time > 1.75 * expected_period and
            self._segment is not None):
            self.gap(dict(reason='sampling_interval_overrun', after_frames=index,
                duration_s=decision_monotonic-self._last_time,
                recovered=True, kind='sampling_interval',
                previous_decision_monotonic=self._last_time,
                next_decision_monotonic=decision_monotonic))
        if self._segment is None:
            self._segment = dict(segment_id=len(self.segments), start_frame=index,
                end_frame=index, decision_start_monotonic=decision_monotonic,
                decision_end_monotonic=decision_monotonic)
            self.segments.append(self._segment)
        else:
            self._segment.update(end_frame=index, decision_end_monotonic=decision_monotonic)
        self._last_time = decision_monotonic

    def gap(self, value):
        self.gaps.append(copy.deepcopy(value))
        self._segment = None

    def end(self, command, frozen=False, unresolved_gap=False):
        if command in ('discard', 'rerecord', 'exit'):
            self.end_reason = 'operator_' + command
        elif command == 'save':
            self.end_reason = ('operator_save_after_video_backpressure' if frozen and
                self.freeze_reason == 'video_backpressure' else
                'operator_save_after_data_timeout' if frozen else
                'operator_save_during_data_gap' if unresolved_gap else 'operator_save')
        else:
            raise ValueError('Unknown lifecycle decision: '+str(command))
        return self.end_reason

    def snapshot(self):
        return dict(end_reason=self.end_reason, task_success=self.task_success,
            segments=copy.deepcopy(self.segments), recording_gaps=copy.deepcopy(self.gaps),
            runtime_capture=copy.deepcopy(self.runtime_capture),
            success_label_source=self.success_label_source,
            runtime_sample_mismatches=list(self.runtime_sample_mismatches),
            freeze_reason=self.freeze_reason,
            sampling_period_s=self.sampling_period_s,
            max_nominal_timeline_error_s=self.max_nominal_timeline_error_s,
            max_nominal_timeline_error_frame=self.max_nominal_timeline_error_frame)


def build_episode_quality(lifecycle, storage_valid, runtime_capture=None, additional_exclusions=()):
    """Conservative admission; an operator success label cannot override faults."""
    value = lifecycle.snapshot() if isinstance(lifecycle, EpisodeLifecycle) else lifecycle
    gaps = value.get('recording_gaps', [])
    end_reason = value.get('end_reason')
    unresolved = any(not gap.get('recovered', False) for gap in gaps)
    truncated = unresolved or end_reason in (
        'operator_save_during_data_gap', 'operator_save_after_data_timeout',
        'operator_save_after_video_backpressure')
    runtime = runtime_capture if runtime_capture is not None else value.get('runtime_capture')
    exclusions = []
    if not storage_valid:
        exclusions.append('storage_invalid')
    if value.get('task_success') is None:
        exclusions.append('task_success_unknown')
    elif value.get('task_success') is False:
        exclusions.append('task_failed')
    if gaps:
        exclusions.append('time_gap_present')
    period=value.get('sampling_period_s')
    nominal_error=value.get('max_nominal_timeline_error_s',0.)
    if period is not None and nominal_error > period:
        exclusions.append('nominal_timeline_drift')
    if truncated:
        exclusions.append('data_interrupted')
    if not runtime or not runtime.get('runtime_verified', False):
        exclusions.append('runtime_configuration_unverified')
    if value.get('runtime_sample_mismatches'):
        exclusions.append('runtime_configuration_changed_or_unmatched')
    if end_reason != 'operator_save':
        exclusions.append('non_normal_end')
    exclusions.extend(reason for reason in additional_exclusions if reason not in exclusions)
    return dict(quality_version=QUALITY_VERSION, storage_valid=bool(storage_valid),
        task_success=value.get('task_success'), training_eligible=not exclusions,
        quality_exclusions=exclusions, end_reason=end_reason,
        segments=copy.deepcopy(value.get('segments', [])),
        recording_gaps=copy.deepcopy(gaps),
        runtime_sample_mismatches=list(value.get('runtime_sample_mismatches', [])),
        freeze_reason=value.get('freeze_reason'),
        nominal_timeline=dict(max_error_s=nominal_error,
            max_error_frame=value.get('max_nominal_timeline_error_frame'),
            exclusion_threshold_s=period,
            measured_as='abs(decision - first_decision - frame_index / fps)',
            interval_segment_threshold_periods=1.75),
        truncated_by_data_interruption=truncated,
        timeline_compressed_by_gaps=bool(gaps),
        training_time_contract='fixed-fps archives with any gap are excluded; '
            'real sample times and contiguous segments are retained for diagnostics',
        success_label_source=value.get('success_label_source', 'operator' if
            value.get('task_success') is not None else 'unknown'))
