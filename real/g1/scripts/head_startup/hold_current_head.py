#!/usr/bin/python3
"""Enable head torque at measured positions without returning to a saved pose."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from boot_head import StartupHead, verified_config
from head_check import CONFIG, DEVICE, ROOT, ReadOnlyHeadBus
from head_serial_lock import exclusive_head_bus

ENABLE_GOAL_TOLERANCE_TICKS = 20  # About 1.76 degrees; only at the torque handoff.


class CurrentPoseHold(StartupHead):
    """Latch the measured pose once; verify torque without a return trajectory."""
    verifying_torque = False

    def inspect(self, servo):
        state = super().inspect(servo)
        if not hasattr(self, 'last_feedback'):self.last_feedback = {}
        self.last_feedback[servo] = dict(state)
        return state

    def feedback_accepted(self, servo, tick):
        # Current-pose holding captures a fresh target below. The initial read
        # is not a fixed reference that the unpowered head must stay beside.
        return 0 <= tick <= 4095

    def run_hold(self):
        initial = self.states()
        torque = [s['torque_enable'] for s in initial]
        if torque not in ([0,0], [1,1]):
            raise RuntimeError('Only one head axis is enabled; manual inspection required')
        if any(self.read(i,10,1)&12 or self.read(i,98,1)!=0 for i in (0,1)):
            raise RuntimeError('Unexpected automatic-torque/profile mode or watchdog')
        current = self.states()
        if [s['torque_enable'] for s in current] != torque:
            raise RuntimeError('Head enable state changed during startup inspection')
        if torque == [1,1] and any(abs(s['goal_position']-s['present_position'])*360/4096>3 for s in current):
            raise RuntimeError('Active head targets do not match its measured pose')
        damping_changes = []
        for i, state in enumerate(current):
            if state['position_d_gain'] != 300:
                self.write(i,80,2,300)
                damping_changes.append(dict(id=i,previous=state['position_d_gain'],applied=300))
        # Refresh after inspection/damping instead of returning to the earlier
        # sample or waiting for a stationary window while torque is off.
        current = self.states()
        if [s['torque_enable'] for s in current] != torque:
            raise RuntimeError('Head enable state changed during startup inspection')
        targets = [s['present_position'] for s in current]
        self.saved_target_ticks = targets[:]
        enable_goal_adjustment = [0, 0]
        if torque == [0,0]:
            for i in (0,1):self.capture_idle_goal(i, targets[i])
            current = self.states()
            if any(s['torque_enable']!=0 or abs(s['goal_position']-s['present_position'])>1 for s in current):
                raise RuntimeError('Head idle goal does not match its measured pose')
            # An idle bus can recapture by one tick. Keep the actual latched goal,
            # rather than commanding a return to the earlier encoder sample.
            targets = [s['goal_position'] for s in current]
            self.saved_target_ticks = targets[:]
            for i in (0,1):self.write(i,64,1,1)
            # Idle goals can follow encoder feedback until torque engages.
            # Accept a small startup handoff, once, with matching feedback.
            # Never write a corrective target or follow later target changes.
            self.verifying_torque = True
            current = self.states()
            if any(s['torque_enable'] != 1 for s in current):
                raise RuntimeError('Head hold was lost during torque handoff')
            for i, state in enumerate(current):
                delta = state['goal_position'] - targets[i]
                if delta and (abs(delta) > ENABLE_GOAL_TOLERANCE_TICKS or
                              not 0 <= state['goal_position'] <= 4095 or
                              abs(state['goal_position'] - state['present_position']) > ENABLE_GOAL_TOLERANCE_TICKS):
                    raise RuntimeError('Head goal changed beyond startup handoff tolerance')
                enable_goal_adjustment[i] = delta
            targets = [s['goal_position'] for s in current]
            self.saved_target_ticks = targets[:]
        else:
            for i in (0,1):
                if current[i]['goal_position'] != targets[i]:self.write(i,116,4,targets[i])
        self.verifying_torque = True
        observed = []
        until = self.clock.monotonic()+5
        while self.clock.monotonic()<until:
            current = self.states()  # Bus/model/mode/current/temperature/fault checks remain.
            if any(s['torque_enable']!=1 for s in current):raise RuntimeError('Head hold was lost')
            if any(current[i]['goal_position']!=targets[i] for i in (0,1)):
                raise RuntimeError('Latched head goal changed during torque verification')
            observed.append(current)
            self.clock.sleep(.1)
        errors = [observed[-1][i]['present_position']-targets[i] for i in (0,1)]
        maximum = [max(abs(s[i]['present_position']-targets[i]) for s in observed) for i in (0,1)]
        return dict(initial=initial, final=observed[-1], target_ticks=targets, holding=True,
            pre_enable_movement_check_enforced=False,
            torque_verified=True, position_error_check_enforced=False,
            position_error_ticks=errors, position_error_deg=[v*360/4096 for v in errors],
            max_position_error_ticks=maximum, position_warning=any(v>5 for v in maximum),
            enable_goal_adjustment_ticks=enable_goal_adjustment,
            damping_changes=damping_changes,
            cold_capture_attempts=self.cold_capture_attempts)


def hold_current(device, cfg, *, sdk=None, clock=None):
    with ReadOnlyHeadBus(device, sdk=sdk) as bus:
        ticks=[bus.inspect(i)['present_position'] for i in (0,1)]
    options=dict(target_ticks=ticks, direct=True, startup_policy='hold_near_reference', sdk=sdk)
    if clock is not None:options['clock']=clock
    with CurrentPoseHold(device,cfg,**options) as head:
        try:result=head.run_hold()
        except Exception as error:
            # Preserve available feedback; do not send a corrective trajectory or
            # disable torque merely because verification failed.
            error.head_state=dict(target_ticks=head.saved_target_ticks,
                final=[head.last_feedback[i] for i in (0,1) if i in getattr(head,'last_feedback',{})],
                last_write_result=getattr(head,'last_write_result',None))
            raise
    result.update(phase='holding_current',hold_current=True,hold_capture_ticks=result['target_ticks'],
                  initial_capture_ticks=ticks,
                  automatic_recalibration=False,saved_reference_unchanged=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-path',type=Path,
                        default=Path('/run/sonic-g1-head-current-hold/status.json'))
    args=parser.parse_args()
    def record(value):
        value['timestamp']=time.time()
        args.state_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2))
        temporary.replace(args.state_path)
        print(json.dumps(value),flush=True)
    try:
        if not DEVICE.exists():raise RuntimeError('Dedicated head adapter not connected')
        with exclusive_head_bus(DEVICE,ROOT):
            cfg=verified_config(CONFIG,
                                ROOT/'autocheck-reference.json')
            record(hold_current(DEVICE,cfg))
    except BaseException as error:
        record(dict(phase='failed',error=str(error),automatic_retry=False,
                    diagnostics=getattr(error,'head_state',None)))
        raise


if __name__=='__main__':main()
