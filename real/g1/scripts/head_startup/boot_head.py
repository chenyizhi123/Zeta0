#!/usr/bin/python3
"""Boot hold at the saved operator reference; no recalibration."""
import argparse
import fcntl
import json
import signal
import subprocess
import time
from pathlib import Path

import yaml
from head_check import CONFIG, DEVICE, ROOT, ReadOnlyHeadBus, signed
from head_tolerance import TARGET_TOLERANCE_TICKS, TARGET_TOLERANCE_DEG
from head_serial_lock import exclusive_head_bus


def verified_config(path, reference_path):
    cfg = yaml.safe_load(Path(path).read_text())
    reference = json.loads(Path(reference_path).read_text())['config']
    if not isinstance(cfg,dict) or any(type(cfg.get(k)) is not type(v) or cfg.get(k)!=v
                                      for k,v in reference.items()):
        raise RuntimeError('Calibration differs from the saved operator record')
    if cfg.get('has_calibrate') != 1 or cfg['direction'] != [1,-1]:
        raise RuntimeError('Saved calibration is not usable')
    return cfg


def verified_startup_target(path, cfg, with_policy=False):
    saved = json.loads(Path(path).read_text())
    if (saved.get('source') != 'user_manually_adjusted_pose'
            or saved.get('operator_confirmed_module_pose') is not True
            or saved.get('calibration_config') != cfg):
        raise RuntimeError('Startup reference does not match the operator record and calibration')
    ticks = saved.get('target_ticks')
    if not isinstance(ticks,list) or len(ticks)!=2 or any(type(t) is not int for t in ticks):
        raise RuntimeError('Invalid saved startup encoder targets')
    policy = saved.get('startup_policy','calibrated_return')
    if policy not in ('calibrated_return','hold_near_reference','operator_endpoint_return'):
        raise RuntimeError('Unknown head startup policy')
    if policy != 'calibrated_return' and saved.get('automatic_recalibration') is not False:
        raise RuntimeError('Local head hold must not change calibration')
    starts=saved.get('operator_verified_start_ticks')
    if starts is not None:
        if (policy != 'operator_endpoint_return' or saved.get('operator_verified_start_source') != 'onsite_manual_return_trial'
                or not isinstance(starts,list) or len(starts)!=2 or any(type(t) is not int for t in starts)):
            raise RuntimeError('Invalid operator-confirmed return starting positions')
    return (ticks,policy,starts) if with_policy else ticks


class StartupHead(ReadOnlyHeadBus):
    def __init__(self, device, cfg, sdk=None, clock=time, target_ticks=None, direct=False,
                 startup_policy='calibrated_return', operator_start_ticks=None,
                 operator_position_margin_ticks=0):
        super().__init__(device,sdk=sdk)
        self.cfg, self.clock = cfg, clock
        self.direct = direct
        if startup_policy not in ('calibrated_return','hold_near_reference','operator_endpoint_return'):
            raise RuntimeError('Unknown head startup policy')
        self.startup_policy = startup_policy
        if type(operator_position_margin_ticks) is not int or not 0 <= operator_position_margin_ticks <= 20:
            raise ValueError('Operator position margin must be 0 to 20 ticks')
        self.operator_position_margin_ticks = operator_position_margin_ticks
        self.operator_start_ticks = operator_start_ticks
        if operator_start_ticks is not None:
            if (startup_policy != 'operator_endpoint_return' or not isinstance(operator_start_ticks,list)
                    or len(operator_start_ticks)!=2 or any(type(t) is not int or not 0<=t<=4095 for t in operator_start_ticks)):
                raise RuntimeError('Invalid operator-confirmed return starting positions')
        self.saved_target_ticks = None if target_ticks is None else list(target_ticks)
        if startup_policy != 'calibrated_return' and self.saved_target_ticks is None:
            raise RuntimeError('Local head hold requires an operator reference')
        if self.saved_target_ticks is not None:
            if len(self.saved_target_ticks)!=2 or any(type(t) is not int for t in self.saved_target_ticks):
                raise RuntimeError('Invalid saved startup encoder targets')
            for i,tick in enumerate(self.saved_target_ticks):
                lo,hi = cfg['joint%d'%i]
                if startup_policy != 'calibrated_return':
                    if not 0 <= tick <= 4095:
                        raise RuntimeError('Local reference must be within one encoder revolution')
                    if startup_policy == 'operator_endpoint_return':
                        lower,upper = self.command_bounds(i)
                        if upper-lower > 2048:
                            raise RuntimeError('Operator endpoint would require an unverified half-revolution span')
                elif not lo-1 <= self.angle(i,tick) <= hi+1:
                    raise RuntimeError('Operator reference outside the recorded safety range')
        self.touched = False
        self.cold_capture_attempts = []

    def angle(self, servo, ticks):
        return (ticks-self.cfg['servo%d_calibration'%servo])*360/4096*self.cfg['direction'][servo]+self.cfg['joint%d'%servo][0]

    def encoder(self, servo, angle):
        return round(self.cfg['servo%d_calibration'%servo]+self.cfg['direction'][servo]*(angle-self.cfg['joint%d'%servo][0])*4096/360)

    def command_bounds(self, servo):
        if self.startup_policy == 'hold_near_reference':
            target = self.saved_target_ticks[servo]
            return [max(0,target-5),min(4095,target+5)]
        lo,hi = self.cfg['joint%d'%servo]
        bounds = sorted([self.encoder(servo,lo),self.encoder(servo,hi)])
        if self.saved_target_ticks is not None:
            # Only the explicitly captured operator endpoint may extend a bound;
            # calibration and the existing one-degree feedback safety guard stay intact.
            target = self.saved_target_ticks[servo]
            bounds = [min(bounds[0],target),max(bounds[1],target)]
        if self.operator_start_ticks is not None:
            start=self.operator_start_ticks[servo]
            bounds=[min(bounds[0],start),max(bounds[1],start)]
        if self.startup_policy == 'operator_endpoint_return':
            margin = self.operator_position_margin_ticks
            bounds = [max(0,bounds[0]-margin),min(4095,bounds[1]+margin)]
        return bounds

    def feedback_accepted(self, servo, tick):
        if self.startup_policy == 'hold_near_reference':
            lower,upper = self.command_bounds(servo)
            return lower <= tick <= upper
        if self.startup_policy == 'operator_endpoint_return':
            # The only added endpoint is the operator-confirmed target; no
            # unrelated encoder region or full-turn wrap becomes valid.
            lower,upper = self.command_bounds(servo)
            return lower-4096/360 <= tick <= upper+4096/360
        lo,hi = self.cfg['joint%d'%servo]
        return lo-1 <= self.angle(servo,tick) <= hi+1

    def write(self, servo, address, size, value):
        # All motor writes are RAM. EEPROM, mode and calibration are never changed.
        if (address,size) not in {(64,1),(80,2),(82,2),(84,2),(108,4),(112,4),(116,4)}:
            raise RuntimeError('Motor write is outside the allowed RAM registers')
        if address == 116 and self.startup_policy != 'calibrated_return':
            lower,upper = self.command_bounds(servo)
            if not lower <= value <= upper:
                raise RuntimeError('Head goal outside the local operator reference')
        self.touched = True
        method = {1:self.packet.write1ByteTxRx,2:self.packet.write2ByteTxRx,
                  4:self.packet.write4ByteTxRx}[size]
        raw_target = value & ((1 << (size*8))-1)
        comm,error = method(self.port,servo,address,raw_target)
        read_method = {1:self.packet.read1ByteTxRx,2:self.packet.read2ByteTxRx,
                       4:self.packet.read4ByteTxRx}[size]
        # READ is diagnostic only: even a missing/error ACK must not hide whether
        # the register changed. This method never retries WRITE or accepts a
        # non-matching register; cold-position recapture is handled separately.
        raw_readback,read_comm,read_error = None,None,None
        read_exception = None
        verification_started = self.clock.monotonic()
        verification_deadline = verification_started + .2
        read_attempts = []
        while True:
            raw_readback,read_comm,read_error = None,None,None
            read_exception = None
            try:
                raw_readback,read_comm,read_error = read_method(self.port,servo,address)
            except Exception as failure:
                read_exception = str(failure)
            read_attempts.append(dict(raw_readback=raw_readback,read_comm=read_comm,
                                      read_error=read_error,read_exception=read_exception,
                                      elapsed=self.clock.monotonic()-verification_started))
            # Only a clean but non-matching READ after a clean ACK may be checked
            # again. Exact matching and all ACK/READ errors end verification now.
            if (comm or error or read_exception is not None or read_comm or read_error
                    or raw_readback == raw_target):
                break
            remaining = verification_deadline-self.clock.monotonic()
            if remaining <= 0:
                break
            self.clock.sleep(min(.01,remaining))
            if self.clock.monotonic() >= verification_deadline:
                break

        def decode(name, code):
            if code is None:
                return None
            decoder = getattr(self.packet,name,None)
            return decoder(code) if decoder is not None else str(code)

        details = dict(servo=servo,address=address,size=size,
                       requested_value=value,raw_target=raw_target,
                       write_comm=comm,write_error=error,
                       write_comm_text=decode('getTxRxResult',comm),
                       write_error_text=decode('getRxPacketError',error),
                       raw_readback=raw_readback,read_comm=read_comm,read_error=read_error,
                       read_comm_text=decode('getTxRxResult',read_comm),
                       read_error_text=decode('getRxPacketError',read_error),
                       read_exception=read_exception,read_attempts=read_attempts,
                       verification_elapsed=self.clock.monotonic()-verification_started)
        self.last_write_result = details
        if (comm or error or read_exception is not None or read_comm or read_error
                or raw_readback != raw_target):
            raise RuntimeError('Motor write/readback failed: '+json.dumps(details,sort_keys=True))

    def capture_idle_goal(self, servo, initial_position):
        """Capture fresh feedback while torque is off, retaining exact checks.

        On this bus an idle goal follows encoder feedback. A one-tick change
        between capture and readback may therefore replace the written value.
        Recapture at most twice, only after a clean ACK/READ, when the new goal
        equals the measured idle position. Never retry a command to move.
        """
        lower,upper = self.command_bounds(servo)
        for attempt in range(3):
            if any(self.read(i,64,1)!=0 or self.read(i,10,1)&12 or self.read(i,98,1)!=0
                   for i in (0,1)):
                raise RuntimeError('Head idle state changed before position capture')
            actual = signed(self.read(servo,132,4),32)
            if abs(actual-initial_position)>12 or not lower-1<=actual<=upper+1:
                raise RuntimeError('Head moved before enabling hold')
            captured = max(lower,min(upper,actual))
            try:
                self.write(servo,116,4,captured)
            except RuntimeError as failure:
                details = getattr(self,'last_write_result',{})
                self.cold_capture_attempts.append(dict(servo=servo,attempt=attempt+1,
                                                      captured=captured,verified=False,
                                                      write_result=details))
                if (not str(failure).startswith('Motor write/readback failed: ')
                        or attempt==2 or details.get('write_comm')!=0
                        or details.get('write_error')!=0 or details.get('read_comm')!=0
                        or details.get('read_error')!=0 or details.get('read_exception') is not None
                        or abs(signed(details['raw_readback'],32)-captured)!=1):
                    raise
                # Distinguish actual idle feedback drift from an ignored write,
                # clipping, or a different writer. The next write uses fresh q.
                now = signed(self.read(servo,132,4),32)
                if (now != signed(details['raw_readback'],32)
                        or abs(now-initial_position)>12 or not lower<=now<=upper):
                    raise
            else:
                self.cold_capture_attempts.append(dict(servo=servo,attempt=attempt+1,
                                                      captured=captured,verified=True,
                                                      write_result=self.last_write_result))
                return

    def states(self):
        states = [self.inspect(i) for i in [0,1]]
        for i,s in enumerate(states):
            if s['model_number']!=1240 or s['operating_mode']!=4 or s['hardware_error']:
                raise RuntimeError('Unexpected head model/mode or hardware fault')
            if s['temperature_c'] >= min(60,s['temperature_limit_c']-5):
                raise RuntimeError('Head temperature warning')
            if abs(s['present_current_ma']) > .9*s['current_limit_ma']:
                raise RuntimeError('Head current warning')
            s['angle_deg'] = self.angle(i,s['present_position'])
            if not self.feedback_accepted(i,s['present_position']):
                if self.startup_policy == 'hold_near_reference':
                    raise RuntimeError('Head moved away from the operator reference; manual positioning required (no automatic large return)')
                raise RuntimeError('Head position outside the recorded limits')
            if not 0 < s['position_p_gain'] <= 1100:
                raise RuntimeError('Unexpected head position gain')
        return states

    def stop_at_actual(self):
        # Keep each axis holding its measured bounded pose; never drop the camera.
        if not self.touched:
            return
        self.failure_hold_report = []
        for i in [0,1]:
            report = dict(servo=i,held_at_measured_pose=False)
            try:
                actual = signed(self.read(i,132,4),32)
                lo,hi = self.cfg['joint%d'%i]
                actual_angle = self.angle(i,actual)
                report.update(present_position=actual,angle_deg=actual_angle)
                if self.feedback_accepted(i,actual) and self.read(i,64,1)==1:
                    # Feedback tolerance is not permission to command beyond limits.
                    lower,upper = self.command_bounds(i)
                    goal = max(lower,min(upper,actual))
                    self.write(i,116,4,goal)
                    report.update(held_at_measured_pose=True,goal_position=goal)
                else:
                    report['reason'] = 'Axis disabled or feedback outside the accepted range; retaining last goal'
            except Exception as failure:
                # No trusted feedback: retain the last bounded goal and issue no blind write.
                report['reason'] = str(failure)
            self.failure_hold_report.append(report)

    def run(self, speed=2, dry_run=False):
        if not 0 < speed <= 2:
            raise ValueError('Boot motion must be at most 2 degrees/second')
        initial = self.states()
        for i in [0,1]:
            if self.read(i,10,1) & 12 or self.read(i,98,1)!=0:
                raise RuntimeError('Unexpected automatic-torque/profile mode or watchdog')
        torque = [s['torque_enable'] for s in initial]
        if torque not in ([0,0],[1,1]):
            raise RuntimeError('Only one head axis is enabled; manual inspection required')
        if torque == [1,1] and any(abs(self.angle(i,initial[i]['goal_position'])-initial[i]['angle_deg'])>3 for i in [0,1]):
            raise RuntimeError('Active head targets do not match its measured pose')
        samples=[]
        for _ in range(12):
            sampled=self.states()
            if [s['torque_enable'] for s in sampled] != torque:
                raise RuntimeError('Head enable state changed during startup inspection')
            samples.append(sampled)
            self.clock.sleep(.05)
        if any(max(s[i]['present_position'] for s in samples)-min(s[i]['present_position'] for s in samples)>23 for i in [0,1]):
            raise RuntimeError('Head is moving; automatic startup refused')
        initial = samples[-1]
        if torque == [1,1] and any(abs(self.angle(i,initial[i]['goal_position'])-initial[i]['angle_deg'])>3 for i in [0,1]):
            raise RuntimeError('Active head targets changed during startup inspection')
        if self.saved_target_ticks is None:
            targets = [0.0,float(self.cfg['joint1'][0])]
            target_ticks = [self.encoder(i,targets[i]) for i in [0,1]]
        else:
            target_ticks = self.saved_target_ticks[:]
            targets = [self.angle(i,target_ticks[i]) for i in [0,1]]
        bounds_by_axis = []
        for i in [0,1]:
            lo,hi = self.cfg['joint%d'%i]
            if self.saved_target_ticks is None and not lo <= targets[i] <= hi:
                raise RuntimeError('Startup target outside the saved limits')
            # The recorded pitch endpoint fluctuates by one tick. Accept only that
            # measurement margin; every issued goal remains within encoder bounds.
            bounds = self.command_bounds(i)
            bounds_by_axis.append(bounds)
            if not bounds[0]-1 <= initial[i]['present_position'] <= bounds[1]+1:
                raise RuntimeError('Head position outside the strict startup command range')
        result={'initial':initial,'requested_yaw_deg':targets[0],
                'requested_pitch_deg':targets[1],'target_ticks':target_ticks,
                'yaw_motion_requested':initial[0]['present_position']!=target_ticks[0],
                'motion_order':['yaw','pitch'],'speed_deg_per_sec':None if self.direct else speed,
                'direct_target':self.direct,'dry_run':dry_run,
                'startup_policy':self.startup_policy}
        if dry_run:
            return result
        profile = [(i,address,self.read(i,address,4)) for i in [0,1] for address in [108,112]]
        motion_observed = [initial]

        def observe(axis,commanded,stationary_angle):
            actual = self.states()
            if any(s['torque_enable']!=1 for s in actual):
                raise RuntimeError('Head hold was lost during startup motion')
            if not self.direct and abs(actual[axis]['angle_deg']-commanded)>8:
                raise RuntimeError('Head could not follow the slow %s motion'%('yaw' if axis==0 else 'pitch'))
            other = 1-axis
            if abs(actual[other]['angle_deg']-stationary_angle)>3:
                raise RuntimeError('Uncommanded %s axis moved unexpectedly'%('yaw' if other==0 else 'pitch'))
            motion_observed.append(actual)
            return actual

        try:
            # Bound the initial hold as well as later trajectories; never drop
            # an already enabled axis while applying the slow RAM profiles.
            for i in [0,1]:
                if not self.direct:
                    self.write(i,108,4,1)
                    self.write(i,112,4,2)
            if torque == [0,0]:
                # Capture actual positions BEFORE enabling torque; no stale startup goal.
                for i in [0,1]:
                    self.write(i,80,2,300)
                    self.write(i,82,2,0)
                    self.write(i,84,2,initial[i]['position_p_gain'])
                    self.capture_idle_goal(i,initial[i]['present_position'])
                current = self.states()
                if any(abs(current[i]['present_position']-initial[i]['present_position'])>12 for i in [0,1]):
                    raise RuntimeError('Head moved before enabling hold')
                if any(s['torque_enable']!=0 or abs(s['goal_position']-s['present_position'])>1
                       for s in current):
                    raise RuntimeError('Head idle goal does not match its measured pose')
                for i in [0,1]:
                    self.write(i,64,1,1)
            # Hardware caps each axis; software stays at <=2 deg/s and <=2 ticks
            # (0.176 deg) per 100ms iteration, including after a scheduler stall.
            for axis in [0,1]:
                current = self.states()
                if any(s['torque_enable']!=1 for s in current):
                    raise RuntimeError('Head hold was lost before startup motion')
                stationary_angle = current[1-axis]['angle_deg']
                lower,upper = bounds_by_axis[axis]
                measured_tick = current[axis]['present_position']
                tick = max(lower,min(upper,measured_tick))
                if measured_tick!=tick:
                    # At most one endpoint-noise tick was accepted at inspection.
                    self.write(axis,116,4,tick)
                    observe(axis,self.angle(axis,tick),stationary_angle)
                if tick==target_ticks[axis] and current[axis]['goal_position']==tick:
                    continue
                if tick==target_ticks[axis] and measured_tick==tick:
                    # Replace a stale active goal without requesting physical motion.
                    self.write(axis,116,4,tick)
                last = self.clock.monotonic()
                credit = 0.0
                if self.direct and tick!=target_ticks[axis]:
                    tick = target_ticks[axis]
                    self.write(axis,116,4,tick)
                while tick!=target_ticks[axis]:
                    now = self.clock.monotonic()
                    credit = min(.2*4096/360,credit+speed*max(0,now-last)*4096/360)
                    last = now
                    step = min(2,int(credit),abs(target_ticks[axis]-tick))
                    if step:
                        tick += step if target_ticks[axis]>tick else -step
                        credit -= step
                        self.write(axis,116,4,tick)
                    observe(axis,self.angle(axis,tick),stationary_angle)
                    self.clock.sleep(.1)
                # Allow time for each axis to approach its goal. Arrival error is
                # diagnostic only; the operator disabled the precise-pose gate.
                # Bus, torque, current, temperature and position bounds still apply.
                settle_deadline = self.clock.monotonic()+5
                while True:
                    actual = observe(axis,targets[axis],stationary_angle)
                    if abs(actual[axis]['present_position']-target_ticks[axis])<=TARGET_TOLERANCE_TICKS:
                        break
                    if self.clock.monotonic()>=settle_deadline:
                        break
                    self.clock.sleep(.1)
            observed=[]
            until=self.clock.monotonic()+5
            while self.clock.monotonic()<until:
                current=self.states()
                if any(s['torque_enable']!=1 for s in current):
                    raise RuntimeError('Head hold was lost')
                observed.append(current)
                motion_observed.append(current)
                self.clock.sleep(.1)
            for i,address,value in profile:
                if not self.direct:
                    self.write(i,address,4,value)
            result.update(final=observed[-1],holding=True,
                          cold_capture_attempts=self.cold_capture_attempts,
                          target_tolerance_ticks=TARGET_TOLERANCE_TICKS,
                          target_tolerance_deg=TARGET_TOLERANCE_DEG,
                          position_error_check_enforced=False,
                          position_error_ticks=[observed[-1][i]['present_position']-target_ticks[i] for i in [0,1]],
                          position_span_ticks=[max(s[i]['present_position'] for s in observed)-min(s[i]['present_position'] for s in observed) for i in [0,1]],
                          motion_position_span_ticks=[max(s[i]['present_position'] for s in motion_observed)-min(s[i]['present_position'] for s in motion_observed) for i in [0,1]])
            return result
        except BaseException:
            self.stop_at_actual()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--startup-delay',type=float,default=10)
    parser.add_argument('--wait-seconds',type=float,default=180)
    parser.add_argument('--state-path',type=Path,default=Path('/run/sonic-g1-head/status.json'))
    parser.add_argument('--target-path',type=Path,default=ROOT/'startup-pose.json')
    args=parser.parse_args()
    if not 0 <= args.startup_delay <= 30 or not 0 <= args.wait_seconds <= 180:
        raise ValueError('Invalid boot delay or connection wait')
    def stop(*unused):
        raise RuntimeError('Head startup was interrupted')
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    def record(data):
        data['timestamp']=time.time()
        args.state_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2))
        temporary.replace(args.state_path)
        print(json.dumps(data,ensure_ascii=False),flush=True)
    try:
        cfg=verified_config(CONFIG,ROOT/'autocheck-reference.json')
        record({'phase':'waiting','dry_run':args.dry_run,'automatic_recalibration':False})
        time.sleep(args.startup_delay)
        deadline=time.monotonic()+args.wait_seconds
        while not DEVICE.exists():
            if time.monotonic()>=deadline:
                raise RuntimeError('Dedicated head adapter not found within the startup wait')
            time.sleep(1)
        with exclusive_head_bus(DEVICE,ROOT):
            # Recheck the saved configuration immediately before opening the motor bus.
            cfg=verified_config(CONFIG,ROOT/'autocheck-reference.json')
            target_ticks,policy,starts=verified_startup_target(args.target_path,cfg,with_policy=True)
            margin=json.loads(args.target_path.read_text()).get('operator_position_margin_ticks',0)
            with StartupHead(DEVICE,cfg,target_ticks=target_ticks,direct=True,
                             startup_policy=policy,operator_start_ticks=starts,
                             operator_position_margin_ticks=margin) as head:
                result=head.run(dry_run=args.dry_run)
        result['phase']='dry_run_complete' if args.dry_run else 'holding_reference'
        result['target_path']=str(args.target_path)
        record(result)
    except BaseException as error:
        failure={'phase':'failed','error':str(error),'automatic_retry':False}
        if 'head' in locals() and hasattr(head,'failure_hold_report'):
            failure['bounded_hold_on_failure']=head.failure_hold_report
        record(failure)
        raise


if __name__=='__main__':
    main()
