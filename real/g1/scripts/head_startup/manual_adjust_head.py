#!/usr/bin/python3
"""Explicit supported manual adjustment, followed by latching the measured pose."""
import fcntl
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from boot_head import verified_config
from head_check import CONFIG, DEVICE, ROOT, ReadOnlyHeadBus
from hold_current_head import CurrentPoseHold, hold_current
from head_serial_lock import exclusive_head_bus


def adjust(device, cfg, exchange, *, sdk=None, clock=None):
    # The caller owns the serial lock for the entire operator interaction.
    with ReadOnlyHeadBus(device, sdk=sdk) as bus:
        initial = [bus.inspect(i) for i in (0, 1)]
    ticks = [s['present_position'] for s in initial]
    options = dict(target_ticks=ticks, direct=True, startup_policy='hold_near_reference', sdk=sdk)
    if clock is not None:
        options['clock'] = clock
    result = None
    unlock_attempted = False
    try:
        with CurrentPoseHold(device, cfg, **options) as head:
            state = head.states()
            if [s['torque_enable'] for s in state] not in ([0, 0], [1, 1]):
                raise RuntimeError('Only one head axis is enabled; support the head and inspect manually')
            if any(head.read(i, 10, 1) & 12 or head.read(i, 98, 1) != 0 for i in (0, 1)):
                raise RuntimeError('Unexpected automatic-torque mode or watchdog')
            if exchange('support_head') != 'UNLOCK':
                return dict(phase='cancelled', motor_writes=False)
            unlock_attempted = True
            for i in (0, 1):
                if state[i]['torque_enable']:
                    head.write(i, 64, 1, 0)
            if any(head.read(i, 64, 1) != 0 for i in (0, 1)):
                raise RuntimeError('Head did not unlock completely; continue supporting it')
        # No angle-based inference of operator intent: only this explicit phase
        # permits teaching a new pose. EOF/cancellation also attempts to relock.
        exchange('adjusting')
    except (KeyboardInterrupt, EOFError):
        if not unlock_attempted:
            return dict(phase='cancelled', motor_writes=False)
    finally:
        if unlock_attempted:
            result = hold_current(device, cfg, sdk=sdk, clock=clock)
    return dict(result, manual_adjustment=True)


def main():
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def exchange(phase):
        emit(dict(phase=phase))
        return sys.stdin.readline().strip()

    try:
        if not DEVICE.exists():
            raise RuntimeError('Head adapter missing')
        with exclusive_head_bus(DEVICE, ROOT):
            cfg = verified_config(CONFIG,
                                  ROOT/'autocheck-reference.json')
            result=adjust(DEVICE, cfg, exchange)
            if result.get('holding'):
                result['timestamp']=time.time()
                status=Path('/run/sonic-g1-head-manual-adjust/status.json')
                status.parent.mkdir(exist_ok=True)
                temporary=status.with_suffix('.tmp')
                temporary.write_text(json.dumps(result));temporary.replace(status)
            emit(result)
    except BaseException as error:
        emit(dict(phase='failed', error=str(error) or type(error).__name__,
                  holding_not_confirmed=True, instruction='Continue supporting the head'))
        raise


if __name__ == '__main__':
    main()
