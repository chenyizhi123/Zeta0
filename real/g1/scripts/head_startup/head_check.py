#!/usr/bin/python3
"""Boot-time head inspection. Sends only Dynamixel PING and READ packets."""
import argparse
import fcntl
import json
import os
import signal
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get('ZETA0_HEAD_SETUP_ROOT', Path.home()/'g1_head_setup')).expanduser()
CONFIG = Path(os.environ.get('ZETA0_HEAD_SERVO_CONFIG',
                             ROOT.parent/'g1_comp_servo_service/config/config.yaml')).expanduser()
EXPECTED = ROOT/'autocheck-reference.json'
# A physical actuator must be selected explicitly by its verified persistent path.
DEVICE = Path(os.environ.get('ZETA0_HEAD_SERVO_DEVICE',
                             '/dev/serial/by-id/REPLACE_HEAD_SERVO_DEVICE')).expanduser()


def signed(value, bits):
    return value-(1 << bits) if value >= (1 << (bits-1)) else value


class ReadOnlyHeadBus:
    def __init__(self, device, sdk=None):
        if sdk is None:
            from dynamixel_sdk import PortHandler, PacketHandler
        else:
            PortHandler, PacketHandler = sdk
        self.port = PortHandler(str(device))
        self.packet = PacketHandler(2.0)

    def __enter__(self):
        try:
            if not self.port.openPort() or not self.port.setBaudRate(1000000):
                raise RuntimeError('Cannot open dedicated head adapter')
            return self
        except Exception:
            # A failed open can leave the SDK's serial object unset. Do not
            # replace the original USB/open error with a cleanup exception.
            try:
                self.port.closePort()
            except Exception:
                pass
            raise

    def __exit__(self, *unused):
        self.port.closePort()

    def checked(self, reply):
        value, comm, error = reply
        if comm != 0 or error:
            raise RuntimeError('Communication error %s, packet error %s' % (comm,error))
        return value

    def read(self, servo, address, size):
        method = {1:self.packet.read1ByteTxRx, 2:self.packet.read2ByteTxRx,
                  4:self.packet.read4ByteTxRx}[size]
        return self.checked(method(self.port,servo,address))

    def inspect(self, servo):
        state = {'id':servo, 'model_number':self.checked(self.packet.ping(self.port,servo))}
        registers = {'operating_mode':(11,1), 'torque_enable':(64,1),
                     'hardware_error':(70,1), 'position_d_gain':(80,2),
                     'position_i_gain':(82,2), 'position_p_gain':(84,2),
                     'goal_position':(116,4), 'present_current_ma':(126,2),
                     'present_position':(132,4), 'temperature_c':(146,1)}
        for name,reg in registers.items():
            state[name] = self.read(servo,*reg)
        for name in ['goal_position','present_position']:
            state[name] = signed(state[name],32)
        state['present_current_ma'] = signed(state['present_current_ma'],16)
        state['temperature_limit_c'] = self.read(servo,31,1)
        state['current_limit_ma'] = self.read(servo,38,2)
        return state


def check_once(config=CONFIG, expected=EXPECTED, device=DEVICE,
               lock_path=ROOT/'serial.lock', sdk=None):
    report = {'checked_at':time.time(), 'automatic_actuation':False,
              'mechanical_zero_verified_at_boot':False, 'status':'waiting_for_device',
              'calibration_record_match':False, 'issues':[], 'servos':[]}
    cfg = None
    try:
        cfg = yaml.safe_load(Path(config).read_text())
        reference = json.loads(Path(expected).read_text())['config']
        if not isinstance(cfg,dict):
            raise ValueError('Calibration config is not a mapping')
        mismatch = [key for key,value in reference.items()
                    if type(cfg.get(key)) is not type(value) or cfg.get(key) != value]
        report['calibration_record_match'] = not mismatch and cfg.get('has_calibrate') == 1
        if not report['calibration_record_match']:
            report['issues'].append('Calibration differs from the operator record: '+', '.join(mismatch))
    except Exception as error:
        report['issues'].append('Cannot validate calibration record: '+str(error))
    if not Path(device).exists():
        report['issues'].append('Dedicated head USB adapter is not connected')
        return report
    try:
        # Read-open still supports exclusive flock on Linux; never overwrite this lock file.
        with Path(lock_path).open('r') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report['status'] = 'serial_in_use'
                return report
            with ReadOnlyHeadBus(device,sdk=sdk) as bus:
                for servo in [0,1]:
                    try:
                        state = bus.inspect(servo)
                        report['servos'].append(state)
                        if state['model_number'] != 1240 or state['operating_mode'] != 4:
                            report['issues'].append('Unexpected model/mode on servo %d' % servo)
                        if state['hardware_error']:
                            report['issues'].append('Hardware fault on servo %d' % servo)
                        if state['temperature_c'] >= min(60,state['temperature_limit_c']-5):
                            report['issues'].append('Temperature warning on servo %d' % servo)
                        if abs(state['present_current_ma']) > .9*state['current_limit_ma']:
                            report['issues'].append('Current warning on servo %d' % servo)
                        if report['calibration_record_match']:
                            q = (state['present_position']-cfg['servo%d_calibration' % servo])*360/4096*cfg['direction'][servo]+cfg['joint%d' % servo][0]
                            state['angle_from_record_deg'] = q
                            if not cfg['joint%d' % servo][0]-1 <= q <= cfg['joint%d' % servo][1]+1:
                                report['issues'].append('Encoder outside recorded range on servo %d' % servo)
                    except Exception as error:
                        report['issues'].append('Cannot inspect servo %d: %s' % (servo,error))
    except Exception as error:
        report['issues'].append('Head inspection unavailable: '+str(error))
    report['status'] = 'attention_required' if report['issues'] else 'connected_read_only'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=10)
    parser.add_argument('--state-path',type=Path,default=Path('/run/sonic-g1-head-check/status.json'))
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError('Check interval must be 1 to 60 seconds')
    stopping = threading.Event()
    for sig in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(sig,lambda *_:stopping.set())
    previous = None
    while not stopping.is_set():
        report = check_once()
        args.state_path.parent.mkdir(parents=True,exist_ok=True)
        temporary = args.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(args.state_path)
        key = (report['status'],tuple(report['issues']),report['calibration_record_match'])
        if args.once or key != previous:
            print(json.dumps(report,ensure_ascii=False),flush=True)
            previous = key
        if args.once:
            return
        stopping.wait(args.interval)


if __name__ == '__main__':
    main()
