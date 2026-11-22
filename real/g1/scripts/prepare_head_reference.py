"""Hold the measured pose or apply the saved head reference after SONIC init."""
import argparse
import getpass
import json
import math
import os
from pathlib import Path
import shlex
import time

SERVICE = 'sonic-g1-head-reference.service'


def validate_result(state, expected, *, hold_current=False):
    phase='holding_current' if hold_current else 'holding_reference'
    if state.get('phase') != phase or state.get('holding') is not True:
        raise RuntimeError('头部未回位并保持: ' + str(state.get('error', state.get('phase'))))
    targets=state.get('hold_capture_ticks') if hold_current else expected['target_ticks']
    if (not isinstance(targets,list) or len(targets)!=2
            or any(type(t) is not int or not 0<=t<=4095 for t in targets)):
        raise RuntimeError('头部保持目标无效')
    if hold_current and (state.get('hold_current') is not True or state.get('saved_reference_unchanged') is not True):
        raise RuntimeError('头部未确认采用当前位置保持')
    if state.get('target_ticks') != targets:
        raise RuntimeError('头部目标与保存的约定姿态不一致')
    final = state.get('final', [])
    if len(final) != 2:
        raise RuntimeError('头部两轴反馈不完整')
    for axis, motor in enumerate(final):
        actual = motor.get('present_position')
        if (motor.get('torque_enable') != 1 or motor.get('hardware_error') != 0
                or motor.get('goal_position') != targets[axis]
                or not isinstance(actual, (int, float)) or not math.isfinite(actual)):
            raise RuntimeError('头部未保持在约定姿态')
        if hold_current and not 0<=actual<=4095:
            raise RuntimeError('头部反馈编码超出有效范围')


def remote_program(*, hold_current=False):
    program='''import json,pathlib,subprocess,sys,time
expected=json.loads(sys.argv[1])
root=pathlib.Path(sys.argv[2]).expanduser()
device=pathlib.Path(sys.argv[3])
saved=json.loads((root/'startup-pose.json').read_text())
for key in ('source','calibration_config','target_ticks','operator_confirmed_module_pose','automatic_recalibration','startup_policy','operator_verified_start_ticks','operator_verified_start_source','operator_position_margin_ticks'):
    if saved.get(key)!=expected.get(key): raise RuntimeError('机器人头部保存姿态与本机记录不同: '+key)
if saved.get('operator_confirmed_module_pose') is not True or saved.get('automatic_recalibration') is not False:
    raise RuntimeError('头部保存姿态尚未确认')
if not device.is_absolute() or not device.exists():
    raise RuntimeError('头部独立 USB 转接器未识别')
started=time.time()
password=sys.stdin.readline()
result=subprocess.run(['sudo','-S','-p','','systemctl','restart','sonic-g1-head-reference.service'],
    input=password,text=True,capture_output=True,timeout=50)
path=pathlib.Path('/run/sonic-g1-head-reference/status.json')
state=json.loads(path.read_text()) if path.exists() else {}
if result.returncode:
    detail=json.dumps(state,ensure_ascii=False) if state.get('timestamp',0)>=started else None
    if not detail:
        journal=subprocess.run(['journalctl','-u','sonic-g1-head-reference.service','--since','@'+str(int(started)),'-n','25','--no-pager'],text=True,capture_output=True,timeout=5)
        detail=result.stderr.strip()+chr(10)+getattr(journal,'stdout','').strip()
    raise RuntimeError('头部回位失败: '+str(detail))
if state.get('timestamp',0)<started:
    raise RuntimeError('头部返回了过期状态')
print(json.dumps(state))
'''
    if hold_current:
        program=program.replace('sonic-g1-head-reference','sonic-g1-head-current-hold')
        program=program.replace('头部回位失败','头部原位保持失败')
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--body-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hold-current',action='store_true',help='读取当前位置并上力保持，不执行保存姿态回位')
    args = parser.parse_args()
    from watch_robot_temperature import deployment_running
    if not deployment_running(args.body_pid):
        raise RuntimeError('指定 SONIC 身体进程已退出，未执行头部回位')
    config = json.loads(args.config.read_text())
    device = config.get('head_servo_device')
    if not isinstance(device, str) or not device.startswith('/dev/') or 'REPLACE_' in device:
        parser.error('Set head_servo_device to the verified device path in the local config')
    head_root = config.get('robot_head_root', '~/g1_head_setup')
    expected = json.loads((args.config.parent/'head-startup-pose.json').read_text())
    password = os.environ.get('G1_SSH_PASSWORD')
    if password is None:
        password = getpass.getpass('G1 内置电脑 unitree 的 SSH 密码（不保存）: ')
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    for known in (Path('/tmp/sonic-g1-known-hosts'), args.config.parent/'robot-known-hosts'):
        if known.exists():
            client.load_host_keys(str(known))
    report = {'body_pid': args.body_pid, 'target_ticks': expected['target_ticks'],
              'mode':'hold_current' if args.hold_current else 'reference_return',
              'body_reinitialized': False, 'automatic_recalibration': False,
              'checked_at': time.time()}
    try:
        client.connect(config['robot_host'], username='unitree', password=password,
                       timeout=8, look_for_keys=True, allow_agent=True)
        command = shlex.join(['python3', '-c', remote_program(hold_current=args.hold_current),
                              json.dumps(expected), head_root, device])
        stdin, stdout, stderr = client.exec_command(command, timeout=60)
        stdin.write(password + '\n')
        stdin.flush()
        stdin.channel.shutdown_write()
        output, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error.strip() or '头部服务没有成功退出')
        state = json.loads(output)
        validate_result(state, expected,hold_current=args.hold_current)
        if args.hold_current:
            report['saved_reference_ticks']=expected['target_ticks']
            report['target_ticks']=state['target_ticks']
        report.update(ready=True, result=state, checked_at=time.time())
        label='头部已在当前位置上力保持，目标编码:' if args.hold_current else '头部相机保持已启用，保存姿态目标编码:'
        print(label,state['target_ticks'],flush=True)
        if args.hold_current and state.get('position_warning'):
            print('头部力矩及目标确认正常；位置偏差仅记录，不阻止启动。偏差刻度:',
                  state.get('position_error_ticks'),flush=True)
    except Exception as error:
        report.update(ready=False, error=str(error), checked_at=time.time())
        raise
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)


if __name__ == '__main__':
    main()
