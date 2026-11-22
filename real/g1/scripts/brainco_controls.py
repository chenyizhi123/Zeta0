"""Revo2 normalized target mapping and selectable grasp inputs.

Protocol reference: unitreerobotics/brainco_hand_service. Its DDS interface
uses six normalized q values, not radians or raw serial SDK 0..1000 values.
This module neither opens serial ports nor publishes DDS messages.
"""
import math
import copy
import uuid
from dex3_controls import TriggerClosureRatios
from recording_runtime import config_hash

CHANNELS = ('thumb', 'thumb_aux', 'index', 'middle', 'ring', 'pinky')
TOPICS = ('rt/brainco/left/cmd', 'rt/brainco/right/cmd')

# Endpoint values follow BrainCo's revo2_tactile_grasp/grasp_controller.py
# grasp_with_speed defaults, converted from SDK 0..1000 to DDS 0..1,
# with both pinkies held open to avoid the observed reopening problem.
# This adapter uses POSITION control, not the SDK's tactile/current algorithm.
DEFAULT_GRASP_CONFIG = {
    'mapping_revision': 2,
    'trigger_full': .90,
    'trigger_hysteresis': .015,
    # Official grasp_with_speed uses [speed,2*speed,speed,2*speed,speed,speed].
    # Normalize that pattern to a ceiling so no channel exceeds the chosen speed.
    'closing_speed_ratios': [.5, 1, .5, 1, .5, .5],
    'default_profile': 'power',
    # Fractions multiply the official endpoint; these are chenyz tuning values,
    # not vendor-validated object-specific grasps. AUX opposes before wrapping.
    'closure_curve': [
        {'stroke': 0., 'fractions': [0, 0, 0, 0, 0, 0]},
        {'stroke': .25, 'fractions': [0, 1, .08, .08, .08, .08]},
        {'stroke': .65, 'fractions': [.25, 1, .75, .75, .80, .80]},
        {'stroke': 1., 'fractions': [1, 1, 1, 1, 1, 1]},
    ],
    'profiles': [
        {'id': 0, 'name': 'tripod', 'closed_pose': [.5, .75, .85, 1, 0, 0]},
        {'id': 1, 'name': 'power', 'closed_pose': [.5, .75, .85, 1, 1, 0]},
    ],
}


def grasp_command_speeds(config, ceiling=.3, target=None, previous=None):
    """Reuse official finger coordination; release uses the existing speed ceiling.

    This controls speed only. It neither sets current nor estimates contact.
    target/previous are published position targets, not tactile measurements.
    """
    if not math.isfinite(ceiling) or not 0 < ceiling <= 1:
        raise ValueError('Normalized command speed must be in (0,1]')
    ratios = (config or {}).get('closing_speed_ratios', [1]*6)
    if len(ratios) != 6 or any(not math.isfinite(v) or not 0 < v <= 1 for v in ratios):
        raise ValueError('Closing speed ratios require six values in (0,1]')
    for pose in (target, previous):
        if pose is not None and (len(pose) != 6 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in pose)):
            raise ValueError('Speed coordination requires normalized position targets')
    return [ceiling if target is not None and previous is not None and
            target[i] < previous[i]-1e-6 else ceiling*ratio
            for i, ratio in enumerate(ratios)]


def hand_execution_policy(config):
    """Resolve execution limits; older captured configs retain their old limits."""
    policy = (config or {}).get('execution_policy', {})
    limits = dict(trigger_rate_per_s=.8, target_rate_per_s=.8, speed_ceiling=.3)
    if not isinstance(policy, dict) or set(policy)-set(limits):
        raise ValueError('Unknown hand execution policy field')
    for name, fallback in limits.items():
        value = policy.get(name, fallback)
        maximum = 1 if name == 'speed_ceiling' else 10
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError(f'{name} must be in (0,{maximum}]')
        limits[name] = float(value)
    return limits


def validate_grasp_config(config):
    profiles = config['profiles']
    if len(profiles) != 2:
        raise ValueError('Exactly two grasp profiles are required')
    for profile, expected in zip(profiles, ((0, 'tripod'), (1, 'power'))):
        if (profile['id'], profile['name']) != expected:
            raise ValueError('Stable grasp IDs must be 0=tripod, 1=power')
        pose = profile['closed_pose']
        if len(pose) != 6 or any(not math.isfinite(q) or not 0 <= q <= 1 for q in pose):
            raise ValueError('Grasp endpoints require six normalized values in [0,1]')
    names = [p['name'] for p in profiles]
    if config['default_profile'] not in names:
        raise ValueError('Unknown default grasp profile')
    grasp_command_speeds(config)
    hand_execution_policy(config)
    full = config.get('trigger_full', .98)
    hysteresis = config.get('trigger_hysteresis', 0.)
    if not math.isfinite(full) or not .5 <= full <= 1:
        raise ValueError('trigger_full must be in [0.5,1]')
    if not math.isfinite(hysteresis) or not 0 <= hysteresis <= .05:
        raise ValueError('trigger_hysteresis must be in [0,0.05]')
    revision = config.get('mapping_revision', 1)
    if type(revision) is not int or revision not in (1, 2):
        raise ValueError('Unsupported grasp mapping revision')
    curve = config.get('closure_curve')
    if curve is not None:
        if len(curve) < 2:
            raise ValueError('Closure curve requires at least two knots')
        previous_stroke, previous_fractions = -1., [0]*6
        for knot in curve:
            stroke, fractions = knot['stroke'], knot['fractions']
            if (not math.isfinite(stroke) or not 0 <= stroke <= 1 or
                    stroke <= previous_stroke or len(fractions) != 6 or
                    any(not math.isfinite(v) or not lo <= v <= 1
                        for lo, v in zip(previous_fractions, fractions))):
                raise ValueError('Closure curve must have increasing strokes and monotone fractions in [0,1]')
            previous_stroke, previous_fractions = stroke, fractions
        if (curve[0]['stroke'] != 0 or any(curve[0]['fractions']) or
                curve[-1]['stroke'] != 1 or any(v != 1 for v in curve[-1]['fractions'])):
            raise ValueError('Closure curve must begin fully open and end at the configured endpoint')
    return tuple(tuple(p['closed_pose']) for p in profiles), names.index(config['default_profile'])


class BrainCoRevo2Hands(TriggerClosureRatios):
    def __init__(self, *, clock, rate=None,
                 open_pose=(0, 0, 0, 0, 0, 0),
                 closed_pose=(1, 1, 1, 1, 1, 1), grasp_config=None):
        for pose in (open_pose, closed_pose):
            if len(pose) != 6 or any(not math.isfinite(q) or not 0 <= q <= 1 for q in pose):
                raise ValueError('Revo2 DDS poses require six normalized values in [0,1]')
        self.open_pose = tuple(open_pose)
        self.closed_pose = tuple(closed_pose)
        self.profiles = None
        self.grasp_config = copy.deepcopy(grasp_config)
        grasp_config = self.grasp_config
        self.config_sha256 = config_hash(grasp_config)
        self.runtime_session = uuid.uuid4().hex
        self.curve = None
        self.mapping_revision = 0  # No selectable-grasp configuration.
        self.trigger_full = .98
        self.trigger_hysteresis = 0.
        self.input_targets = [0., 0.]
        self.profile_ids = [-1, -1]  # -1 identifies the legacy all-finger mapping.
        if grasp_config is not None:
            if any(self.open_pose):
                raise ValueError('Selectable grasps must share the fully open zero pose')
            self.profiles, default = validate_grasp_config(grasp_config)
            self.profile_ids = [default, default]
            self.curve = grasp_config.get('closure_curve')
            self.mapping_revision = grasp_config.get('mapping_revision', 1)
            self.trigger_full = grasp_config.get('trigger_full', .98)
            self.trigger_hysteresis = grasp_config.get('trigger_hysteresis', 0.)
        self.click_armed = [False, False]
        self.selection_status = ['idle', 'idle']
        super().__init__(clock=clock, rate=hand_execution_policy(grasp_config)['trigger_rate_per_s']
                         if rate is None else rate)

    def update(self, left_trigger, right_trigger):
        if self.profiles is None:
            return super().update(left_trigger, right_trigger)
        now = self.clock()
        dt = 0. if self.last_update is None else min(.05, max(0., now-self.last_update))
        self.last_update = now
        for side, value in enumerate((left_trigger, right_trigger)):
            value = float(value)
            if not math.isfinite(value):
                continue
            target = min(1., max(0., (value-.02)/(self.trigger_full-.02)))
            # Always honor release/full press; suppress only small interior noise.
            if target in (0., 1.) or abs(target-self.input_targets[side]) > self.trigger_hysteresis:
                self.input_targets[side] = target
            difference = self.input_targets[side]-self.ratios[side]
            self.ratios[side] += max(-self.rate*dt, min(self.rate*dt, difference))
        return self.positions()

    def fractions(self, stroke):
        if self.curve is None:
            return [stroke]*6
        for low, high in zip(self.curve, self.curve[1:]):
            if stroke <= high['stroke']:
                alpha = (stroke-low['stroke'])/(high['stroke']-low['stroke'])
                return [lo+alpha*(hi-lo) for lo, hi in zip(low['fractions'], high['fractions'])]
        return list(self.curve[-1]['fractions'])

    def select_from_clicks(self, clicks, triggers, enabled):
        """One selection per fresh released->pressed edge; never switch mid-grasp."""
        self.selection_status = ['idle', 'idle']
        if not enabled:
            self.click_armed = [False, False]
            return
        for side, (click, trigger) in enumerate(zip(clicks, triggers)):
            if not click:
                self.click_armed[side] = True
            elif self.click_armed[side]:
                self.click_armed[side] = False
                if self.profiles is None:
                    continue
                if trigger > .02 or self.ratios[side] > 1e-6:
                    self.selection_status[side] = 'release_trigger_and_wait_for_open'
                    continue  # Rejected clicks are consumed, never applied later.
                self.profile_ids[side] = 1-self.profile_ids[side]
                self.selection_status[side] = 'changed'

    def positions(self):
        # Both hands share normalized closure direction; no Dex3 sign mirroring.
        endpoints = (self.closed_pose,)*2 if self.profiles is None else tuple(
            self.profiles[i] for i in self.profile_ids)
        return tuple([lo + fraction*(hi-lo) for lo, hi, fraction in
                      zip(self.open_pose, endpoint, self.fractions(ratio))]
                     for ratio, endpoint in zip(self.ratios, endpoints))

    def command_fields(self, speed=None):
        speed = hand_execution_policy(self.grasp_config)['speed_ceiling'] if speed is None else speed
        return tuple({'topic': topic, 'channels': CHANNELS,
                      'q': q, 'dq': grasp_command_speeds(self.grasp_config, speed)}
                     for topic, q in zip(TOPICS, self.positions()))


def install_brainco_hands(pico, reader_getter, clock, grasp_config=None):
    """Keep legacy seven-slot body packets neutral; export Revo2 separately."""
    import numpy as np
    controller = BrainCoRevo2Hands(clock=clock, grasp_config=grasp_config)
    original = pico.build_planner_message
    original_axes = pico.get_axis_clicks
    def routed_axes(reader=None):
        # Clicks select grasp profiles only; axes still control locomotion.
        return False, False

    def neutral_pose(*args):
        return np.zeros((1, 7), dtype=np.float32), np.zeros((1, 7), dtype=np.float32)

    def neutral_planner(*args, **kwargs):
        kwargs['left_hand_position'] = [0.0]*7
        kwargs['right_hand_position'] = [0.0]*7
        return original(*args, **kwargs)

    def status_fields(mode):
        reader = reader_getter()
        sample = reader.get_latest() if reader is not None else None
        age = clock()-sample['timestamp_monotonic'] if sample is not None else float('inf')
        valid = math.isfinite(age) and 0 <= age <= .5
        left = right = 0.
        if valid:
            _, left, right, _, _ = pico.get_controller_inputs(reader)
            valid = math.isfinite(float(left)) and math.isfinite(float(right))
        clicks = original_axes(reader) if valid else (False, False)
        # Direct trigger control in every active mode; no stick-click unlock.
        # OFF/initialization and stale tracking still publish no hand commands.
        enabled = valid and mode in (1,2,3,4,5)
        controller.select_from_clicks(clicks, (left,right), enabled)
        if enabled:
            controller.update(left, right)
        else:
            controller.hold()
        left_q, right_q = controller.positions()
        return {'brainco_left_q': np.asarray(left_q, dtype=np.float32),
                'brainco_right_q': np.asarray(right_q, dtype=np.float32),
                'brainco_input_valid': np.asarray([valid], dtype=bool),
                'brainco_triggers': np.asarray([left,right], dtype=np.float32),
                'brainco_stick_clicks': np.asarray(clicks, dtype=bool),
                'brainco_grasp_profile': np.asarray(controller.profile_ids, dtype=np.int32),
                'brainco_grasp_closure': np.asarray(controller.ratios, dtype=np.float32),
                'brainco_grasp_revision': np.asarray([controller.mapping_revision], dtype=np.int32),
                'hc': np.asarray(list(bytes.fromhex(controller.config_sha256+controller.runtime_session)),dtype=np.int32),
                'brainco_hand_enable_armed': np.asarray([enabled], dtype=bool),
                'brainco_requires_unlock': np.asarray([False], dtype=bool),
                'brainco_hand_enabled': np.asarray([enabled], dtype=bool),
                # Preserve the original body policy; active input enables tracking.
                'upper_body_motion_enabled': np.asarray([valid and mode in (1,2,3,4,5)], dtype=bool)}

    pico.compute_hand_joints_from_inputs = neutral_pose
    pico.build_planner_message = neutral_planner
    pico.get_axis_clicks = routed_axes
    return controller, status_fields
