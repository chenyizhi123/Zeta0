"""Finish collection first, then stop the previous project's control processes."""
import os
from pathlib import Path
import signal
import subprocess
import time

from recorder_handoff import save_and_exit,start_ticks


def wait_exit(pid,identity,timeout):
    deadline=time.monotonic()+timeout
    while start_ticks(pid)==identity:
        if time.monotonic()>deadline:
            raise RuntimeError(f'等待旧进程 PID {pid} 正常退出超时；未强制终止，本次部署停止。')
        time.sleep(.05)


def stop_body(pid,local):
    identity=start_ticks(pid)
    if identity is None:return
    expected=Path(local['sonic_root'])/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
    process=Path('/proc')/str(pid)
    executable=os.readlink(process/'exe').removesuffix(' (deleted)')
    if Path(executable).resolve()!=expected.resolve():
        raise RuntimeError(f'身体进程 PID {pid} 不属于本项目，保留进程。')
    print(f'正在请求旧SONIC PID {pid} 正常停控。',flush=True)
    if b'vmf_autostart_version' in (process/'exe').read_bytes():
        os.kill(pid,signal.SIGINT)  # This version handles SIGINT through Stop()/damping.
    else:
        terminal=os.readlink(process/'fd/0')
        try:
            rows=subprocess.check_output(['tmux','list-panes','-a','-F','#{pane_id}\t#{pane_tty}'],text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('旧身体程序没有自动停控接口，请在原部署终端按 O 正常退出一次。') from error
        panes=[line.split('\t')[0] for line in rows.splitlines() if line.endswith('\t'+terminal)]
        if len(panes)!=1:
            raise RuntimeError('无法定位旧身体控制终端，请在原部署终端按 O 正常退出一次。')
        subprocess.run(['tmux','send-keys','-l','-t',panes[0],'o'],check=True)
    wait_exit(pid,identity,15.)


def prepare_handoff(root,local,process_ids,timeout=120.):
    if process_ids('g1_deploy_onnx_ref'):
        raise RuntimeError('SONIC 仍在运行，保留原控制；请在原控制终端主动退出后再启动新部署。')
    recorders=sorted(set(process_ids('record_lerobot.py')+process_ids('record_episode.py')))
    for pid in recorders:
        save_and_exit(root,pid,timeout=timeout)
    if process_ids('record_lerobot.py') or process_ids('record_episode.py'):
        raise RuntimeError('仍有采集进程，停止切换，不启动机器人。')
    if process_ids('g1_deploy_onnx_ref'):
        raise RuntimeError('SONIC 已在运行，保留原控制，不自动停控。')
    for filename in ('pico_manager.py','brainco_bridge.py'):
        if process_ids('g1_deploy_onnx_ref'):
            raise RuntimeError('身体程序仍在运行，保留原输入发布器。')
        for pid in process_ids(filename):
            identity=start_ticks(pid)
            if identity is None:continue
            print(f'清理旧控制发布器 {filename}，PID {pid}。',flush=True)
            try:os.kill(pid,signal.SIGINT)
            except ProcessLookupError:continue
            wait_exit(pid,identity,5.)
