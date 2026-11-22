"""Keep the native SONIC terminal alive independently of a policy client."""
import os
from pathlib import Path
import shlex
import subprocess
import time

from recorder_handoff import start_ticks


SESSION = 'chenyz_model_sonic'


class PersistentBody:
    def __init__(self, pid, session=SESSION):
        self.pid = int(pid)
        self.session = session
        self.identity = start_ticks(self.pid)
        if self.identity is None:
            raise RuntimeError('SONIC 启动后已退出，请检查身体日志。')

    def poll(self):
        return None if start_ticks(self.pid) == self.identity else 0

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('SONIC', timeout)
            time.sleep(.02)
        return 0

    def send_signal(self, value):
        if self.poll() is None:
            os.kill(self.pid, value)

    def remove_exited_terminal(self):
        if self.poll() is not None:
            subprocess.run(['tmux', 'kill-session', '-t', self.session],
                           capture_output=True, check=False)


def launch_body(root, command, log_path, env, *, session=SESSION):
    """tmux owns stdin; closing/crashing the Python client cannot hang it up."""
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise RuntimeError('SONIC 控制终端仍存在，保留原会话：tmux attach -t ' + session)
    Path(log_path).write_text('')
    shell_command = 'exec ' + shlex.join(map(str, command)) + ' > ' + shlex.quote(str(log_path)) + ' 2>&1'
    arguments = ['tmux', 'new-session', '-d', '-s', session, '-n', 'body', '-c', str(root),
                 '-P', '-F', '#{pane_pid}']
    if env.get('PYTHONPATH'):
        arguments.extend(['-e', 'PYTHONPATH=' + env['PYTHONPATH']])
    result = subprocess.run([*arguments, shell_command], capture_output=True, text=True, check=True)
    return PersistentBody(int(result.stdout.strip()), session)
