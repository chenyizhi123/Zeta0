"""Record CRC-verified temperatures for one SONIC PID, with no control output.

Temperature, missing telemetry and monitor errors never request robot shutdown.
This process creates DDS readers only and does not send keys to the deployment.
"""
import argparse
import json
import math
from pathlib import Path
import time


def deployment_running(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[0]
        return Path(command.decode(errors='replace')).name == 'g1_deploy_onnx_ref'
    except FileNotFoundError:
        return False


def valid_temperatures(temperatures):
    return (len(temperatures) == 29
            and all(len(pair) == 2 for pair in temperatures)
            and all(math.isfinite(float(t)) for pair in temperatures for t in pair))


def write_status(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def monitor(reader, crc, pid, status_path, log_path, *,
            is_running=deployment_running, clock=time.monotonic,
            timestamp=time.time, sleep=time.sleep):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_valid = clock()
    last_log = float('-inf')
    samples = 0
    state = {'deployment_pid': pid, 'automatic_stop': False,
             'phase': 'waiting_for_telemetry', 'checked_at': timestamp()}
    write_status(status_path, state)
    with log_path.open('a', buffering=1) as log:
        try:
            while is_running(pid):
                for message in reader.take(100):
                    if not hasattr(message, 'motor_state') or crc.Crc(message) != message.crc:
                        continue
                    motors = message.motor_state[:29]
                    temperatures = [list(m.temperature) for m in motors]
                    if not valid_temperatures(temperatures):
                        raise ValueError('Invalid motor temperatures')
                    now = clock()
                    last_valid = now
                    samples += 1
                    state = {'checked_at': timestamp(), 'deployment_pid': pid,
                             'automatic_stop': False, 'samples': samples,
                             'crc_verified': True, 'phase': 'monitoring',
                             'temperatures': temperatures,
                             'max_temperature': max(map(max, temperatures)),
                             'waist_pitch_q': motors[14].q,
                             'waist_pitch_dq': motors[14].dq,
                             'waist_pitch_torque_est': motors[14].tau_est}
                    if now - last_log >= .5:
                        last_log = now
                        log.write(json.dumps(state) + '\n')
                        write_status(status_path, state)
                if clock() - last_valid > (5 if samples == 0 else 1):
                    raise RuntimeError('No fresh CRC-verified robot state')
                sleep(.01)
        except KeyboardInterrupt:
            state.update(phase='monitor_stopped', checked_at=timestamp())
        except Exception as error:
            state.update(phase='monitor_error', reason=str(error), checked_at=timestamp())
            write_status(status_path, state)
            log.write(json.dumps(state) + '\n')
            print('Temperature monitor error (no stop requested): ' + str(error), flush=True)
            return 1
        else:
            state.update(phase='deployment_exited', checked_at=timestamp())
        write_status(status_path, state)
        log.write(json.dumps(state) + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    from robot_network import LOCAL_CONFIG, local_dds_config
    parser.add_argument('--config', type=Path, default=LOCAL_CONFIG)
    args = parser.parse_args()
    if not deployment_running(args.pid):
        raise RuntimeError('Expected SONIC deployment PID not found')

    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.core import Qos, Policy
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC
    cfg = local_dds_config(args.config, peer_key='body_dds_peer')
    domain = Domain(0, cfg)
    participant = DomainParticipant(0)
    reader = DataReader(participant, Topic(participant, 'rt/lowstate', LowState_),
                        qos=Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1)))
    return monitor(reader, CRC(), args.pid, args.status, args.log)


if __name__ == '__main__':
    raise SystemExit(main())
