#!/usr/bin/env python3
"""Start SONIC once, independently from per-task collection."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
SESSION='chenyz_sonic'


def processes(filename):
    result=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                argv=(p/'cmdline').read_bytes().split(b'\0')
                if any(Path(a.decode(errors='replace')).name==filename for a in argv if a):
                    result.append(int(p.name))
            except OSError:pass
    return result


def component(window,command,log):
    windows=subprocess.check_output(['tmux','list-windows','-t',SESSION,'-F','#{window_name}'],text=True).splitlines()
    full='cd '+shlex.quote(str(ROOT))+' && '+command+' 2>&1 | tee '+shlex.quote(str(ROOT/'logs'/log))
    if window in windows:
        subprocess.run(['tmux','respawn-pane','-k','-t',SESSION+':'+window,full],check=True)
    else:
        subprocess.run(['tmux','new-window','-d','-t',SESSION,'-n',window,full],check=True)


def retire_legacy_exporter():
    for pid in processes('export_dataset.py'):
        print('关闭旧时间戳采集进程，改用按任务入口，PID:',pid)
        try:os.kill(pid,2)
        except ProcessLookupError:pass


def stop_receivers(filename):
    # A new deployment must not inherit an old hand hold target or active
    # manager mode. These processes are stopped only while body control is off.
    for pid in processes(filename):
        try:os.kill(pid,2)
        except ProcessLookupError:pass
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        if not processes(filename):return
        time.sleep(.05)
    raise RuntimeError('旧接收进程未退出: '+filename)


def restore_head_reference(data, config, pid, remote_env, *, hold_current=False):
    message='SONIC 已初始化，正在让头部在当前位置上力保持。' if hold_current else 'SONIC 已初始化，正在让头部相机回到保存姿态并保持。'
    print(message,flush=True)
    command=[str(data),str(ROOT/'scripts/prepare_head_reference.py'),
             '--config',str(config),'--body-pid',str(pid),
             '--output',str(ROOT/'logs/head-startup-result.json')]
    if hold_current:command.append('--hold-current')
    try:
        result=subprocess.run(command,env=remote_env,check=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as error:
        action='头部上力保持失败' if hold_current else '头部回位失败'
        raise RuntimeError(action+'；SONIC 身体进程没有因此被停止。请检查 logs/head-startup-result.json。') from error
    if getattr(result,'stdout',None):print(result.stdout,end='',flush=True)
    return True


def robot_credentials():
    remote_env = dict(os.environ)
    if 'G1_SSH_PASSWORD' not in remote_env:
        remote_env['G1_SSH_PASSWORD'] = getpass.getpass('G1 内置电脑 unitree 的 SSH 密码（不保存）: ')
    return remote_env


def ensure_hand_feedback(sim, data, config, remote_env, allow_recovery=True):
    """One recovery for a measured missing-hand failure, before body startup."""
    output = ROOT/'logs/hand-start-preflight.json'
    command = [str(sim),str(ROOT/'scripts/hand_preflight.py'),'--output',str(output)]
    started = time.time()
    checked = subprocess.run(command,env=remote_env,check=False)
    if checked.returncode == 0:
        return
    try:
        result = json.loads(output.read_text())
    except (OSError,ValueError):
        result = {}
    # Import failures, stale reports and other runtime exceptions are not
    # evidence of a partial hand service and must not trigger a restart.
    missing = result.get('missing',[])
    measured_missing = (result.get('ready') is False and bool(missing) and
        set(missing) <= {'left','right'} and result.get('checked_at',0) >= started)
    if not allow_recovery or not measured_missing:
        raise subprocess.CalledProcessError(checked.returncode,command)
    if processes('g1_deploy_onnx_ref'):
        raise RuntimeError('身体控制已在运行，不重启手部服务')
    print('双手反馈检查未通过；自动恢复手部服务一次，再检查真实反馈。身体尚未启动。',flush=True)
    subprocess.run([str(data),str(ROOT/'scripts/prepare_robot_services.py'),
                    '--config',str(config),'--restart-hands'],env=remote_env,check=True)
    subprocess.run(command,env=remote_env,check=True)


def main():
    parser=argparse.ArgumentParser(description='启动 SONIC；任务录制请另运行 record_episode.sh')
    parser.add_argument('--robot-ready',action='store_true',help='兼容旧命令；启动已无需输入 START 确认')
    parser.add_argument('--skip-remote',action='store_true',help='内置电脑的相机和手部服务已手动恢复')
    parser.add_argument('--require-cameras',action='store_true',help='可选：同时要求相机设备检查通过；默认允许无相机遥操作')
    parser.add_argument('--dry-run',action='store_true',help='仅显示流程，不启动服务或发送机器人指令')
    args=parser.parse_args()
    steps=['检查身体 DDS 温度与故障','恢复内置电脑双手服务；相机缺失默认提示而不阻止遥操作','核验双手真实 DDS 反馈',
        '启动相机组合、双手桥接和记录源','初始化 SONIC（身体与双臂进入默认姿态，手指无开合指令）',
        '绑定本次部署 PID 的只读温度记录（不自动停控）','等待 Init Done 后在头部当前位置上力保持；手动调角使用 adjust_head.sh',
        '头部原位保持成功后启动 PICO OFF，等待人工 ABXY 启控',
        '任务采集另用 record_episode.sh']
    if args.dry_run:
        print(json.dumps(dict(steps=steps,root=str(ROOT),robot_commands_sent=False),ensure_ascii=False,indent=2));return
    local=json.loads((ROOT/'config/local.json').read_text())
    sonic=Path(local['sonic_root']);sim=sonic/'.venv_sim/bin/python';data=sonic/'.venv_data_collection/bin/python'
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/online-model-control.lock').open('a+') as control_lock:
        try:
            fcntl.flock(control_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('模型正在在线控制，请先正常退出模型客户端，再恢复 SONIC/PICO 会话。')
    live=processes('g1_deploy_onnx_ref')
    if live:
        retire_legacy_exporter()
        if len(live) != 1: raise RuntimeError('发现多个 SONIC 身体进程，无法绑定头部保持')
        print('SONIC 已在运行，PID:',live,'；在头部当前位置保持，检查 PICO 控制进程。')
        restore_head_reference(data, ROOT/'config/local.json', live[0], robot_credentials(),hold_current=True)
        if live[0] not in processes('g1_deploy_onnx_ref'):
            raise RuntimeError('设置头部保持期间 SONIC 已退出，未恢复 PICO 控制')
        if not processes('pico_manager.py'):
            component('teleop','python3 run.py teleop','teleop.log')
            print('PICO 控制进程缺失，已重新启动为 OFF；等待 Manager controls，松开全部按键后按 ABXY 启控。',flush=True)
        else:
            print('PICO 控制进程已存在，保留当前会话。',flush=True)
        return
    print('启动会退出官方运动模式，身体与双臂进入 SONIC 默认姿态；ABXY 不再额外固定双臂，手指不自动开合。')
    env=dict(os.environ,PYTHONPATH=str(sonic/'external_dependencies/unitree_sdk2_python'))
    subprocess.run([str(sim),str(ROOT/'scripts/body_preflight.py'),'--output',str(ROOT/'logs/body-start-preflight.json')],env=env,check=True)
    stop_receivers('brainco_bridge.py')
    stop_receivers('pico_manager.py')
    # Reload read-only telemetry on a clean body start as well, so raw hand
    # channels used by grasp recording cannot remain on an old source version.
    stop_receivers('brainco_record_source.py')
    remote_env = robot_credentials()
    if not args.skip_remote:
        preparation=[str(data),str(ROOT/'scripts/prepare_robot_services.py'),'--config',str(ROOT/'config/local.json')]
        if args.require_cameras:preparation.append('--require-cameras')
        subprocess.run(preparation,env=remote_env,check=True)
    # Pass the DDS SDK path explicitly rather than relying on the caller's shell.
    feedback_env = dict(remote_env,PYTHONPATH=env['PYTHONPATH'])
    ensure_hand_feedback(sim,data,ROOT/'config/local.json',feedback_env,
                         allow_recovery=not args.skip_remote)
    if subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','console','-c',str(ROOT)],check=True)
    subprocess.run(['tmux','set-option','-t',SESSION,'remain-on-exit','on'],check=True)
    # Retire the old always-running exporter so PICO save buttons cannot write
    # an unrelated timestamp dataset in parallel with the new task entry point.
    retire_legacy_exporter()
    # Clean body starts also load the current shared observation contract.
    stop_receivers('compose_camera_streams.py')
    try:
        with socket.create_connection(('127.0.0.1',local['camera_output_port']),timeout=.3):pass
    except OSError:
        component('cameras','python3 run.py compose-cameras','cameras.log')
    # The old manager was stopped above. Launch the new OFF manager only after
    # body initialization and head return, so early ABXY cannot be lost or stop
    # a deployment that is still initializing.
    if not processes('brainco_record_source.py'):
        component('hand-data','python3 run.py brainco-record','brainco-record.log')
    component('hand-bridge','python3 run.py brainco-hands --publish --duration 0','brainco-bridge.log')
    component('robot-control','python3 run.py deploy-real','deploy-current.log')
    deadline=time.monotonic()+10;live=[]
    while time.monotonic()<deadline:
        live=processes('g1_deploy_onnx_ref')
        if live:break
        time.sleep(.1)
    if len(live)!=1:raise RuntimeError('SONIC 进程未正常启动，请检查 logs/deploy-current.log')
    pid=live[0]
    command='PYTHONPATH='+shlex.quote(env['PYTHONPATH'])+' '+shlex.quote(str(sim))+' scripts/watch_robot_temperature.py --pid '+str(pid)+' --status logs/temperature-status.json --log logs/temperature-history.jsonl'
    component('thermal-monitor',command,'temperature-monitor.log')
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        if pid not in processes('g1_deploy_onnx_ref'):raise RuntimeError('SONIC 启动后退出，请检查部署和温度日志')
        if 'Init Done' in (ROOT/'logs/deploy-current.log').read_text(errors='replace'):break
        time.sleep(.2)
    else:raise RuntimeError('等待初始化超时，请检查部署终端；程序可能仍运行')
    print('SONIC Init Done，PID:',pid)
    restore_head_reference(data, ROOT/'config/local.json', pid, remote_env,hold_current=True)
    if pid not in processes('g1_deploy_onnx_ref'):
        raise RuntimeError('设置头部保持期间 SONIC 已退出，未启动 PICO 控制；请检查 logs/deploy-current.log')
    component('teleop','python3 run.py teleop','teleop.log')
    print('PICO 控制已启动为 OFF；等待 teleop 日志出现 Manager controls，松开全部按键后再启控。',flush=True)
    print('PICO: ABXY 启动 SONIC，松开后 A+X 切 POSE 全身跟随；POSE 中 B+Y 切摇杆走动+头部/双手跟随。扳机直接控制手指，无需摇杆解锁。')
    print('采集: bash record_episode.sh 任务名 [episode编号]；不会因采集退出而停止 SONIC。')
    print('部署终端: tmux attach -t '+SESSION+'；运行中 ABXY 或部署终端 O 退出并进入阻尼。')


if __name__=='__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/sonic-start.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
        except Exception as error:
            print('启动失败:',error,flush=True)
            raise SystemExit(1)
