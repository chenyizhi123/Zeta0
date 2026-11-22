#!/usr/bin/python3
"""Continuous head telemetry. Only READ packets; never correct targets or torque."""
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from head_check import DEVICE, ROOT, ReadOnlyHeadBus, signed
from head_serial_lock import exclusive_head_bus


class MonitorBus(ReadOnlyHeadBus):
    def checked(self, reply):
        value, comm, error=reply
        # Hardware Alert must not prevent reading the actual fault register.
        if comm or error & 0x7f:
            raise RuntimeError(f'Head READ failed: communication={comm}, packet={error}')
        return value

    def sample(self, servo):
        registers={'operating_mode':(11,1),'torque_enable':(64,1),'hardware_error':(70,1),
                   'goal_position':(116,4),'present_pwm':(124,2),'present_current_ma':(126,2),
                   'present_position':(132,4),'voltage_decivolt':(144,2),'temperature_c':(146,1),
                   'd_gain':(80,2),'p_gain':(84,2),'i_gain':(82,2),'goal_pwm':(100,2)}
        state={'id':servo}
        for name,(address,size) in registers.items():
            value=self.read(servo,address,size)
            state[name]=signed(value,size*8) if name in (
                'goal_position','present_position','present_pwm','present_current_ma','goal_pwm') else value
        state['position_error_deg']=(state['present_position']-state['goal_position'])*360/4096
        return state


def classify(states, expected):
    events=[]
    for s in states:
        axis=s['id']
        def flag(kind):events.append(dict(axis=axis,kind=kind))
        if s['torque_enable']!=1:flag('torque_lost')
        if s['hardware_error']:flag('hardware_fault')
        if s['operating_mode']!=4:flag('mode_changed')
        if expected is not None and s['goal_position']!=expected[axis]:flag('goal_changed')
        if abs(s['position_error_deg'])>3:flag('position_drift_over_3deg')
        if abs(s['present_pwm'])>=.9*max(1,abs(s['goal_pwm'])):flag('output_near_limit')
        if s['temperature_c']>=60:flag('temperature_warning')
        if not 35<=s['voltage_decivolt']<=70:flag('voltage_warning')
    return events


def latest_reference():
    candidates=[]
    for name in ('sonic-g1-head-current-hold','sonic-g1-head-reference','sonic-g1-head-manual-adjust'):
        try:
            d=json.loads((Path('/run')/name/'status.json').read_text())
            targets=d.get('target_ticks')
            if d.get('holding') and isinstance(targets,list) and len(targets)==2:
                if all(type(v) is int and 0<=v<=4095 for v in targets):
                    candidates.append((d.get('timestamp',0),targets))
        except (OSError,ValueError,TypeError):pass
    return max(candidates,key=lambda d:d[0]) if candidates else None


def sample_head():
    """Keep the bus lock inside a disposable, read-only sampling process."""
    with exclusive_head_bus(DEVICE,ROOT,timeout=0):
        with MonitorBus(DEVICE) as bus:
            return [bus.sample(i) for i in (0,1)]


def bounded_sample(timeout=1.5):
    try:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--sample'],
                              capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('Head monitor read timed out; sampling process stopped') from error
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Head monitor sampling failed')
    return json.loads(result.stdout)


def main():
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    directory=Path('/var/log/sonic-g1-head-hold');directory.mkdir(exist_ok=True)
    logger=logging.getLogger('head-hold');logger.setLevel(logging.INFO)
    handler=RotatingFileHandler(directory/'samples.jsonl',maxBytes=20*1024*1024,backupCount=5)
    handler.setFormatter(logging.Formatter('%(message)s'));logger.addHandler(handler)
    status=Path('/run/sonic-g1-head-hold-monitor/status.json');status.parent.mkdir(exist_ok=True)
    baseline=None;expected=None;previous=None
    while not stop.is_set():
        started=time.monotonic()
        report=dict(timestamp=time.time(),read_only=True,events=[],servos=[])
        try:
            reference=latest_reference()
            if reference and (baseline is None or reference[0]>baseline):
                baseline,expected=reference
                report['reference_updated']=True
            report['servos']=bounded_sample()
            if expected is None:
                expected=[s['goal_position'] for s in report['servos']]
                report['reference_source']='initial_readback'
            report['events']=classify(report['servos'],expected)
            report['expected_targets']=expected
            report['status']='attention' if report['events'] else 'holding'
        except Exception as error:
            report.update(status='unavailable',error=str(error))
        signature=(report['status'],json.dumps(report['events']),report.get('error'))
        if signature!=previous:
            print(json.dumps(report,ensure_ascii=False),flush=True);previous=signature
        logger.info(json.dumps(report,ensure_ascii=False))
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(status)
        # Leave the lock available between failed reads for explicit head requests.
        stop.wait(max(.2 if report['status']=='unavailable' else 0,
                      .2-(time.monotonic()-started)))


if __name__=='__main__':
    if sys.argv[1:]==['--sample']:
        try:
            print(json.dumps(sample_head()))
        except Exception as error:
            print(str(error),file=sys.stderr)
            sys.exit(1)
    else:
        main()
