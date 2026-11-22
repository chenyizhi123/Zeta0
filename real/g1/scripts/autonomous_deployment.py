"""Own a model deployment's services and SONIC body; never start VR software."""
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import build_opener, ProxyHandler

from start_sonic import ensure_hand_feedback, restore_head_reference, robot_credentials
from persistent_sonic_body import launch_body


STEPS = ['核对模型、prompt和示范首帧','检查旧会话；保留已有SONIC，不自动停控', '自动启动或复用匹配的模型服务',
         '检查身体DDS故障，恢复相机和双手服务并核验反馈', '启动本次模型DDS手桥',
         '自动启动SONIC，等待Init Done', '读取头部当前位置并上力保持，视角由操作者调整',
         '自动启控进入零速度站立并核对身体、双手和三路图像', '显示f / v / m菜单；异常保留SONIC，仅主动退出才停控']


def health_at(endpoint):
    with build_opener(ProxyHandler({})).open(endpoint.rstrip('/')+'/health', timeout=2) as response:
        return json.load(response)


def validate_model(health, manifest):
    if health.get('weights_sha256') != manifest['weights_sha256'] or health.get('task') != manifest['task']:
        raise RuntimeError('在线服务的任务或权重与所选模型不一致')
    from psi.deploy.real_sonic_client import check_health
    grid = manifest['temporal_abi']
    if (health.get('action_hz'), health.get('action_horizon')) != (grid['action_hz'], grid['action_horizon']):
        raise RuntimeError('模型服务动作频率或长度与权重不一致')
    kwargs = {} if grid['action_hz'] == 50 else dict(action_hz=grid['action_hz'], action_horizon=grid['action_horizon'])
    check_health(health, state_count=1, rtc_prefix_steps=15, **kwargs)
    if health.get('observation_contract') != manifest['observation_contract']:
        raise RuntimeError('在线服务的观测/动作契约与模型不一致')


def require_free_robot(process_ids):
    current = process_ids('record_lerobot.py')
    legacy = process_ids('record_episode.py')
    if current or legacy:
        locations = []
        for pid in sorted(set(current+legacy)):
            try:
                terminal = os.readlink(f'/proc/{pid}/fd/0')
            except OSError:
                terminal = '原采集终端'
            locations.append(f'PID {pid}，{terminal}')
        action = ('请在该采集终端按 Esc 保存并正常退出，再运行本脚本；Ctrl+C 会丢弃当前未保存片段。'
                  if current else '请在原采集终端保存并正常退出，再运行本脚本。')
        raise RuntimeError('采集仍在运行（'+'；'.join(locations)+'）。'+action)
    if process_ids('g1_deploy_onnx_ref'):
        raise RuntimeError('已有身体部署占用机器人；请正常退出旧部署，再运行本命令。新部署将自行启动SONIC。')
    for filename in ('pico_manager.py', 'brainco_bridge.py'):
        if process_ids(filename):
            raise RuntimeError('其他控制发布器仍在运行：'+filename+'；请正常退出旧部署后再启动独立模型部署。')


class AutonomousDeployment(AbstractContextManager):
    def __init__(self, root, local):
        self.root, self.local = Path(root), local
        self.sonic = Path(local['sonic_root'])
        self.sim = self.sonic/'.venv_sim/bin/python'
        self.data = self.sonic/'.venv_data_collection/bin/python'
        self.sdk_env = dict(os.environ, PYTHONPATH=str(self.sonic/'external_dependencies/unitree_sdk2_python'))
        self.children, self.files = [], []
        self.body = None
        self.body_monitor = None
        self.remote_env = None

    def __enter__(self):
        return self

    def launch(self, command, log_name, *, body=False, env=None):
        if body:
            self.body = launch_body(self.root, command, self.root/'logs'/log_name, env or self.sdk_env)
            return self.body
        log = (self.root/'logs'/log_name).open('w')
        self.files.append(log)
        child = subprocess.Popen([str(value) for value in command], cwd=self.root,
            env=env or self.sdk_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        self.children.append(child)
        return child

    def model(self, task, endpoint, manifest, model_paths_override=None):
        try:
            health = health_at(endpoint)
        except HTTPError as exc:
            raise RuntimeError('已有模型服务的健康检查失败，不启动重复服务。') from exc
        except (URLError, TimeoutError):
            parsed = urlparse(endpoint)
            if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.path not in ('', '/'):
                raise RuntimeError('指定模型服务不可用；自动启动仅支持本机HTTP地址。')
            command = [self.root/'.venv_policy/bin/python', self.root/'scripts/serve_local_policy.py',
                '--task', task, '--host', parsed.hostname, '--port', str(parsed.port or 80)]
            if model_paths_override is not None:
                command.extend(['--model-paths', model_paths_override])
            child = self.launch(command, 'online-model-server.log')
            deadline, update = time.monotonic()+180., time.monotonic()
            while True:
                if child.poll() is not None:
                    raise RuntimeError('模型服务启动失败，查看 logs/online-model-server.log。')
                try:
                    health = health_at(endpoint)
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError('等待模型服务超时，查看 logs/online-model-server.log。')
                    if time.monotonic()-update > 15.:
                        print('正在加载模型服务，日志：logs/online-model-server.log', flush=True)
                        update = time.monotonic()
                    time.sleep(.1)
        validate_model(health, manifest)
        return health

    def prepare_robot(self, skip_remote=False):
        subprocess.run([str(self.sim), str(self.root/'scripts/body_preflight.py'),
                        '--output', str(self.root/'logs/online-body-preflight.json')], env=self.sdk_env, check=True)
        self.remote_env = robot_credentials()
        if not skip_remote:
            subprocess.run([str(self.data), str(self.root/'scripts/prepare_robot_services.py'),
                '--config', str(self.root/'config/local.json'), '--require-cameras'], env=self.remote_env, check=True)
        feedback_env = dict(self.remote_env, PYTHONPATH=self.sdk_env['PYTHONPATH'])
        ensure_hand_feedback(self.sim, self.data, self.root/'config/local.json', feedback_env,
                             allow_recovery=not skip_remote)

    def start_body(self):
        print('自动启动SONIC身体程序，等待初始化。', flush=True)
        child = self.launch(['bash', self.root/'scripts/deploy_g1.sh'], 'online-sonic-body.log', body=True)
        deadline, update = time.monotonic()+90., time.monotonic()
        while True:
            if child.poll() is not None:
                raise RuntimeError('SONIC初始化时退出，查看 logs/online-sonic-body.log。')
            if 'Init Done' in (self.root/'logs/online-sonic-body.log').read_text(errors='replace'):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('SONIC初始化超时，查看 logs/online-sonic-body.log。')
            if time.monotonic()-update > 15.:
                print('等待SONIC Init Done，日志：logs/online-sonic-body.log', flush=True)
                update = time.monotonic()
            time.sleep(.1)
        self.body_monitor = self.launch([self.sim, self.root/'scripts/watch_robot_temperature.py', '--pid', str(child.pid),
                     '--status', 'logs/online-temperature-status.json', '--log', 'logs/online-temperature-history.jsonl'],
                    'online-temperature-monitor.log')
        try:
            restore_head_reference(self.data, self.root/'config/local.json', child.pid, self.remote_env,
                                   hold_current=True)
        except RuntimeError as exc:
            raise RuntimeError('头部原位上力保持失败；保留SONIC，查看 logs/head-startup-result.json。') from exc
        if child.poll() is not None:
            raise RuntimeError('头部原位上力保持期间身体程序已退出。')
        return child.pid

    def close(self, *, stop_body=False):
        error = None
        body_alive = self.body is not None and self.body.poll() is None
        if body_alive and stop_body:
            # This is only reached for an explicit operator exit, never errors.
            self.body.send_signal(signal.SIGINT)
            try:
                self.body.wait(timeout=10)
                body_alive = False
            except subprocess.TimeoutExpired:
                error = RuntimeError('身体尚未确认正常退出；保留进程，检查 logs/online-sonic-body.log。')
        elif body_alive:
            print(f'SONIC PID {self.body.pid} 已保留；客户端退出不会停控。'
                  '原生超时处理回到零速规划；主动退出请在 tmux attach -t chenyz_model_sonic 中按 O。', flush=True)
        for child in reversed(self.children):
            if body_alive and child is self.body_monitor:
                continue
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=5)
        if self.body is not None and not body_alive:
            self.body.remove_exited_terminal()
        for log in self.files:
            log.close()
        if error:
            raise error

    def __exit__(self, *exc):
        self.close(stop_body=exc[0] is KeyboardInterrupt)


def bootstrap_control(io, child, timeout=15.):
    """Start only our initialized native child, after its startup capability ACK."""
    import numpy as np
    from online_controls import require_controller, scalar
    from psi.deploy.sonic_wire import build_planner_message, pack_pose_message

    socket = io.subscribe(io.config.body_endpoint, b'robot_config')
    deadline = time.monotonic()+timeout
    epoch = time.time_ns()//1_000_000
    capable = False
    next_send = 0.
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('SONIC在自动启控阶段退出。')
            now = time.monotonic()
            io.poll(now)
            config = io.receive(socket, b'robot_config')
            if config is not None:
                if config.get('controller_pid') != child.pid or config.get('vmf_autostart_version') != 1:
                    raise RuntimeError('身体版本不支持独立自动启控；请使用本次编译的身体程序。')
                capable = True
            if capable and now >= next_send:
                io.pose.send(build_planner_message(0, (0., 0., 0.), (1., 0., 0.), speed=0.))
                io.pose.send(pack_pose_message(dict(start=np.array([True]), stop=np.array([False]),
                    planner=np.array([True]), vmf_epoch=np.array([epoch], dtype=np.int64)), topic='command', version=1))
                io.active = True
                next_send = now+.02
            status = io.controller_status
            if status is not None:
                require_controller(status, time.monotonic(), child.pid)
                if (scalar(status, 'vmf_epoch') == epoch and scalar(status, 'input_control_mode') == 0
                        and scalar(status, 'planner_ready') == 1 and scalar(status, 'vmf_watchdog_tripped') == 0):
                    try:
                        io.check_feedback(time.monotonic())
                        io.observation(time.monotonic())
                    except ValueError:
                        pass
                    else:
                        print('SONIC自动启控完成，零速度站立；请选择 f / v / m。', flush=True)
                        return
            time.sleep(.002)
        raise RuntimeError('自动启控或传感器就绪等待超时，查看身体、相机和手桥日志。')
    finally:
        socket.close(linger=0)
