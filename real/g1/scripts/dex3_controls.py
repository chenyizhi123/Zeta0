"""Continuous Dex3 trigger targets; no robot connection or motor writes.

Closed poses match SONIC's G1 hand IK middle-finger grasp preset. All seven
targets remain inside the Dex3 limits used by the deployment driver.
"""
import math
import time


LEFT_CLOSED = (0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5)
RIGHT_CLOSED = tuple(-q for q in LEFT_CLOSED)


class TriggerClosureRatios:
    def __init__(self, clock=time.monotonic, rate=0.8):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Hand closing rate must be finite and positive')
        self.clock = clock
        self.rate = rate
        self.ratios = [0.0, 0.0]
        self.last_update = None

    def update(self, left_trigger, right_trigger):
        now = self.clock()
        dt = 0.0 if self.last_update is None else min(0.05, max(0.0, now-self.last_update))
        self.last_update = now
        for i, value in enumerate((left_trigger, right_trigger)):
            value = float(value)
            if not math.isfinite(value):
                continue  # Retain the last target if the input is invalid.
            # A small release deadzone keeps q=0 stable when the trigger is up.
            target = min(1.0, max(0.0, (value-0.02)/0.96))
            difference = target-self.ratios[i]
            self.ratios[i] += max(-self.rate*dt, min(self.rate*dt, difference))
        return self.positions()

    def positions(self):
        return self.ratios[:]

    def hold(self):
        # Restart timing on recovery so stale time cannot cause a target jump.
        self.last_update = None
        return self.positions()


class ContinuousDex3Hands(TriggerClosureRatios):
    def positions(self):
        return ([q*self.ratios[0] for q in LEFT_CLOSED],
                [q*self.ratios[1] for q in RIGHT_CLOSED])


def install_continuous_hands(pico, reader_getter, clock=time.monotonic):
    """Patch only hand targets in SONIC's pose and all planner modes."""
    import numpy as np

    controller = ContinuousDex3Hands(clock=clock)
    original_planner_builder = pico.build_planner_message

    def fresh_reader():
        reader = reader_getter()
        sample = reader.get_latest() if reader is not None else None
        if sample is None:
            return None
        age = clock()-sample['timestamp_monotonic']
        return reader if math.isfinite(age) and 0 <= age <= 0.5 else None

    def targets():
        reader = fresh_reader()
        if reader is None:
            return controller.hold()
        _, left_trigger, right_trigger, _, _ = pico.get_controller_inputs(reader)
        return controller.update(left_trigger, right_trigger)

    def compute_hands(left_solver, right_solver, left_trigger, left_grip,
                      right_trigger, right_grip):
        # Grips retain their existing recording shortcuts; they do not close fingers.
        left, right = (controller.update(left_trigger, right_trigger)
                       if fresh_reader() is not None else controller.hold())
        return np.asarray(left, dtype=np.float32).reshape(1, 7), np.asarray(right, dtype=np.float32).reshape(1, 7)

    def build_planner(*args, **kwargs):
        left, right = targets()
        kwargs['left_hand_position'] = left
        kwargs['right_hand_position'] = right
        return original_planner_builder(*args, **kwargs)

    pico.compute_hand_joints_from_inputs = compute_hands
    pico.build_planner_message = build_planner
    return controller
