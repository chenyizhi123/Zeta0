#!/usr/bin/env python3
"""Interactive head-only manual adjustment; does not restart SONIC or cameras."""
import getpass
import json
import os
from pathlib import Path
import shlex
import time

ROOT = Path(__file__).resolve().parents[1]


def operator_reply(phase, prompt=input):
    if phase == 'support_head':
        try:
            prompt('请先用手托住相机/头部，再按 Enter 解锁调整（Ctrl+C 取消）：')
        except (KeyboardInterrupt, EOFError):
            return 'CANCEL'
        return 'UNLOCK'
    if phase == 'adjusting':
        try:
            prompt('已解锁，请始终托住。调好并保持静止后按 Enter 上力；确认锁定成功后再松手：')
        except (KeyboardInterrupt, EOFError):
            print('\n正在尝试在当前位置重新上力，请继续托住。', flush=True)
        return 'LOCK'
    raise ValueError('Unexpected adjustment phase: '+phase)


def main():
    import paramiko
    config = json.loads((ROOT/'config/local.json').read_text())
    device = config.get('head_servo_device')
    if not isinstance(device, str) or not device.startswith('/dev/') or 'REPLACE_' in device:
        raise ValueError('Set head_servo_device to the verified device path in the local config')
    head_root = config.get('robot_head_root', '~/g1_head_setup')
    password = os.environ.get('G1_SSH_PASSWORD')
    if password is None:
        password = getpass.getpass('G1 内置电脑 unitree 的 SSH 密码（不保存）: ')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    for path in (Path('/tmp/sonic-g1-known-hosts'), ROOT/'config/robot-known-hosts'):
        if path.exists():
            client.load_host_keys(str(path))
    terminal = None
    report = dict(started_at=time.time(), kind='manual_head_adjustment', body_commands_sent=False)
    try:
        client.connect(config['robot_host'], username='unitree', password=password, timeout=8)
        program = '''import os, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser()
command = ['sudo', '-S', '-p', '', 'env', 'PYTHONDONTWRITEBYTECODE=1',
           'PYTHONPATH='+str(root/'deps'), 'ZETA0_HEAD_SETUP_ROOT='+str(root),
           'ZETA0_HEAD_SERVO_DEVICE='+sys.argv[2],
           'python3', '-u', str(root/'manual_adjust_head.py')]
os.execvp(command[0], command)
'''
        stdin, stdout, stderr = client.exec_command(
            shlex.join(['python3', '-c', program, head_root, device]))
        stdin.write(password+'\n'); stdin.flush()
        for line in stdout:
            state = json.loads(line)
            phase = state.get('phase')
            if phase in ('support_head', 'adjusting'):
                stdin.write(operator_reply(phase)+'\n'); stdin.flush()
            else:
                terminal = state
        code = stdout.channel.recv_exit_status()
        error = stderr.read().decode()
        report.update(result=terminal, returncode=code)
        if code or terminal is None or terminal.get('phase') == 'failed':
            raise RuntimeError((terminal or {}).get('error') or error or '没有收到锁定回执')
        if terminal.get('phase') == 'cancelled':
            print('已取消，未解锁头部。')
            return
        if terminal.get('holding') is not True or terminal.get('torque_verified') is not True:
            raise RuntimeError('没有确认两轴上力保持')
        print('头部已锁定在你调整后的角度，可以松手。目标编码:',terminal['target_ticks'])
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        print('头部调整未确认完成，请继续托住相机并查看错误：'+report['error'], flush=True)
        raise
    finally:
        client.close()
        path = ROOT/'logs/head-manual-adjustment.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
