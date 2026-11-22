"""Recognize complete PICO episode gestures without importing any SDK.

A gesture is committed only when all four face buttons are released. The
matching left Grip may be released before, with, or after the face button.
"""
import math
import time
from collections.abc import Mapping
from numbers import Real


def _plain(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def _number(value):
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError('expected one numeric value')
        value = _plain(value[0])
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('expected a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('expected a finite number')
    return value


def _buttons(value):
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('expected four A/B/X/Y button values')
    result = []
    for button in value:
        button = _plain(button)
        if isinstance(button, bool):
            result.append(button)
        elif isinstance(button, Real) and button in (0, 1):
            result.append(bool(button))
        else:
            raise ValueError('expected boolean A/B/X/Y button values')
    return tuple(result)


class PicoEpisodeGesture:
    """Protocol 2 release gestures; ``feed`` returns an event or ``None``.

    A fresh, fully released sample must arm the receiver before each new
    connection/reset. A single A/B press may wait for Grip to be squeezed, but
    once qualified, the candidate stays latched regardless of Grip release order.
    Any other face button cancels the entire cycle until all ABXY are released.
    ``diagnostics`` is a JSON-compatible dictionary snapshot.
    """

    def __init__(self, max_age_s=0.5):
        if not isinstance(max_age_s, Real) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self.max_age_s = float(max_age_s)
        self._armed = False
        self._cycle_button = None
        self._candidate = None
        self._cancelled = False
        self._last_stamp = None
        self._last_signature = None
        self._last_received_at = None
        self._reset_floor = None
        self.diagnostics = dict(status='waiting_release', armed=False, candidate=None,
            cycle_button=None, reason='initial_release_required', last_event=None,
            last_emitted_event=None, last_cancel_reason=None, protocol=None,
            stamp_age_s=None, input_age_ms=None, emitted_events=0,
            cancelled_cycles=0, invalid_samples=0, legacy_samples=0, resets=0,
            reset_floor_monotonic=None)

    def _state(self, status, reason):
        self.diagnostics.update(status=status, reason=reason, armed=self._armed,
            candidate=self._candidate, cycle_button=self._cycle_button)

    def _clear_cycle(self):
        self._cycle_button = self._candidate = None
        self._cancelled = False

    def _interrupt(self, status, reason):
        if self._candidate is not None:
            self.diagnostics['cancelled_cycles'] += 1
            self.diagnostics['last_cancel_reason'] = reason
        self._armed = False
        self._clear_cycle()
        self._state(status, reason)

    def _cancel(self, reason):
        if not self._cancelled:
            self.diagnostics['cancelled_cycles'] += 1
        self._candidate = None
        self._cancelled = True
        self.diagnostics['last_cancel_reason'] = reason
        self._state('cancelled', reason)

    def reset(self, now=None):
        """Interrupt any candidate and require release; retain replay protection."""
        now = _number(time.monotonic() if now is None else now)
        self._reset_floor = now if self._reset_floor is None else max(now, self._reset_floor)
        self.diagnostics['resets'] += 1
        self.diagnostics['reset_floor_monotonic'] = self._reset_floor
        self.diagnostics['last_event'] = None
        self._interrupt('waiting_release', 'reset_release_required')

    def feed(self, manager, now):
        self.diagnostics['last_event'] = None
        self.diagnostics['stamp_age_s'] = self.diagnostics['input_age_ms'] = None
        try:
            protocol = _number(manager.get('recording_controls_protocol')) if isinstance(manager, Mapping) else None
        except ValueError:
            protocol = None
        self.diagnostics['protocol'] = protocol
        if protocol != 2:
            self.diagnostics['legacy_samples'] += 1
            self._interrupt('legacy', 'protocol_2_required')
            return None
        try:
            now = _number(now)
            stamp = _number(manager['recording_controls_monotonic'])
            grip = _number(manager['recording_left_grip'])
            input_age_ms = _number(manager['pico_input_age_ms'])
            buttons = _buttons(manager['controller_buttons_abxy'])
            if not 0 <= grip <= 1:
                raise ValueError('left Grip must be in [0, 1]')
        except (KeyError, ValueError, TypeError):
            self.diagnostics['invalid_samples'] += 1
            self._interrupt('malformed', 'invalid_control_fields')
            return None
        age = now - stamp
        self.diagnostics.update(stamp_age_s=age, input_age_ms=input_age_ms)
        if not 0 <= age <= self.max_age_s or not 0 <= input_age_ms <= self.max_age_s * 1000:
            self.diagnostics['invalid_samples'] += 1
            self._interrupt('stale', 'control_or_tracking_age_invalid')
            return None
        if self._reset_floor is not None and stamp <= self._reset_floor:
            self._interrupt('waiting_release', 'sample_predates_reset')
            return None
        signature = (buttons, grip, input_age_ms)
        if self._last_stamp is not None and stamp <= self._last_stamp:
            if stamp == self._last_stamp and signature == self._last_signature:
                self._state('duplicate', 'sample_already_processed')
            else:
                self.diagnostics['invalid_samples'] += 1
                self._interrupt('malformed', 'control_timestamp_not_increasing')
            return None
        if ((self._last_received_at is not None and now - self._last_received_at > self.max_age_s) or
            (self._last_stamp is not None and stamp - self._last_stamp > self.max_age_s)):
            self._interrupt('waiting_release', 'control_stream_gap')
        self._last_stamp, self._last_signature = stamp, signature
        self._last_received_at = now

        # Release comes first: simultaneous face/Grip release is a valid end.
        if not any(buttons):
            event = self._candidate if self._armed and not self._cancelled else None
            self._clear_cycle()
            self._armed = True
            self._state('armed', 'all_face_buttons_released')
            if event is not None:
                self.diagnostics['emitted_events'] += 1
                self.diagnostics.update(last_event=event, last_emitted_event=event)
            return event
        if not self._armed:
            self._state('waiting_release', 'held_buttons_need_release')
            return None
        if self._cancelled:
            self._state('cancelled', self.diagnostics['last_cancel_reason'])
            return None
        pressed = [index for index, down in enumerate(buttons) if down]
        if self._cycle_button is None:
            if len(pressed) != 1 or pressed[0] not in (0, 1):
                self._cancel('ambiguous_or_mode_buttons')
                return None
            self._cycle_button = pressed[0]
        elif pressed != [self._cycle_button]:
            self._cancel('other_face_button_in_cycle')
            return None
        if grip > 0.5:
            self._candidate = 'toggle_save' if self._cycle_button == 0 else 'discard'
        if self._candidate is not None:
            self._state('holding', 'waiting_for_face_release')
        else:
            self._state('waiting_grip', 'single_face_button_without_grip')
        return None
