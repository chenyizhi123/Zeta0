"""Cooperative, acknowledged recorder save-and-exit over a private local socket."""
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import tempfile
import threading
import time
import uuid


def start_ticks(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except (OSError,IndexError):
        return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def receive_json(connection):
    data = b''
    while b'\n' not in data:
        part = connection.recv(4096)
        if not part:
            raise ValueError('Incomplete recorder-control message')
        data += part
        if len(data)>8192:
            raise ValueError('Recorder-control message is too large')
    value = json.loads(data.split(b'\n',1)[0])
    if not isinstance(value,dict):
        raise ValueError('Recorder-control message must be a map')
    return value


class RecorderControl:
    def __init__(self,root,dataset_root):
        self.root,self.dataset_root = Path(root),Path(dataset_root)
        self.pid = os.getpid()
        self.identity = start_ticks(self.pid)
        self.token = secrets.token_hex(24)
        self.directory = self.root/'logs/recorder-control'
        self.capability_path = self.directory/f'{self.pid}.json'
        self.result_path = self.directory/f'{self.pid}.result.json'
        self.operator = None
        self.request_id = None
        self.accepted_state = None
        self.error = None
        self.stopped = threading.Event()
        self.thread = None
        self.socket = None
        self.temporary = None

    def metadata(self):
        return dict(format='chenyz.recorder.save_exit.v1',pid=self.pid,start_ticks=self.identity,
                    token=self.token,socket=str(Path(self.temporary.name)/'control.sock'),
                    dataset_root=str(self.dataset_root),request_id=self.request_id)

    def attach(self,operator):
        self.operator = operator
        operator.control = self

    def start(self):
        if self.thread is not None:
            return
        self.temporary = tempfile.TemporaryDirectory(prefix=f'chenyz-recorder-{self.pid}-')
        self.socket = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        path = str(Path(self.temporary.name)/'control.sock')
        self.socket.bind(path)
        os.chmod(path,0o600)
        self.socket.listen(2)
        self.socket.settimeout(.1)
        atomic_json(self.capability_path,self.metadata())
        self.thread = threading.Thread(target=self._serve,daemon=True,name='recorder-save-exit')
        self.thread.start()

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection,_ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(1.)
                try:
                    _,uid,_ = struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    request = receive_json(connection)
                    if (uid != os.getuid() or not isinstance(request.get('token'),str)
                            or not hmac.compare_digest(request['token'],self.token)
                            or request.get('command') != 'save_exit'
                            or not isinstance(request.get('request_id'),str)):
                        raise ValueError('Invalid recorder save-exit request')
                    if self.request_id is None:
                        state = self.operator.request_save_exit()
                        if state['accepted']:
                            self.request_id = request['request_id']
                            self.accepted_state = state
                            atomic_json(self.capability_path,self.metadata())
                    else:
                        state = self.accepted_state
                    reply = dict(state,request_id=self.request_id or request['request_id'])
                    connection.sendall(json.dumps(reply,ensure_ascii=False).encode()+b'\n')
                except Exception as error:
                    try:connection.sendall(json.dumps(dict(accepted=False,reason=str(error))).encode()+b'\n')
                    except OSError:pass

    def stop(self):
        self.stopped.set()
        if self.socket is not None:
            self.socket.close()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def note_error(self,error):
        self.error = str(error) or type(error).__name__

    def finish(self):
        """Called only after LeRobot.finalize and session-metadata cleanup."""
        self.stop()
        if self.request_id is not None:
            operator = self.operator
            reason = self.error or operator.failure_kind
            if operator.save_uncertain or operator.segment_open:
                reason = reason or '当前片段未确认保存'
            atomic_json(self.result_path,dict(format='chenyz.recorder.save_exit.result.v1',
                pid=self.pid,start_ticks=self.identity,token=self.token,request_id=self.request_id,
                status='failed' if reason else 'completed',reason=reason,
                session_saved=operator.session_saved,dataset_root=str(self.dataset_root),finished_at=time.time()))
        self.capability_path.unlink(missing_ok=True)
        if self.temporary is not None:
            self.temporary.cleanup()


def save_and_exit(root,pid,timeout=120.):
    root = Path(root)
    identity = start_ticks(pid)
    if identity is None:
        return
    path = root/'logs/recorder-control'/f'{pid}.json'
    try:
        capability = json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        raise RuntimeError(f'采集器 PID {pid} 是未提供保存接口的旧会话；请在原采集终端按 Esc 保存退出一次。后续新会话将自动处理。') from exc
    if (capability.get('format') != 'chenyz.recorder.save_exit.v1' or capability.get('pid') != pid
            or capability.get('start_ticks') != identity or path.is_symlink() or path.stat().st_uid != os.getuid()):
        raise RuntimeError(f'采集器 PID {pid} 的保存接口身份不匹配，保留进程。')
    request_id = capability.get('request_id')
    if request_id is None:
        request_id = uuid.uuid4().hex
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
                connection.settimeout(2.)
                connection.connect(capability['socket'])
                connection.sendall(json.dumps(dict(command='save_exit',request_id=request_id,
                    token=capability['token'])).encode()+b'\n')
                reply = receive_json(connection)
        except (OSError,ValueError) as exc:
            raise RuntimeError(f'无法请求采集器 PID {pid} 保存退出；保留进程，请查看原采集终端。') from exc
        if reply.get('accepted') is not True:
            raise RuntimeError(f'采集器 PID {pid} 未接受保存退出：'+str(reply.get('reason','状态不允许')))
        request_id = reply['request_id']
    print(f'已请求采集器 PID {pid} 保存并退出，等待文件写入和收尾完成。',flush=True)
    deadline,update = time.monotonic()+timeout,time.monotonic()
    result_path = path.with_name(f'{pid}.result.json')
    while True:
        if start_ticks(pid) != identity:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'等待采集器 PID {pid} 保存退出超时；保留进程，不启动机器人。')
        if time.monotonic()-update > 15:
            print(f'仍在等待 PID {pid} 完成数据保存和退出。',flush=True)
            update = time.monotonic()
        time.sleep(.05)
    try:
        result = json.loads(result_path.read_text())
    except (OSError,ValueError) as exc:
        raise RuntimeError(f'采集器 PID {pid} 已退出，但没有保存完成回执；本次部署停止。') from exc
    if (result.get('pid') != pid or result.get('start_ticks') != identity or
            result.get('request_id') != request_id or result.get('token') != capability['token']):
        raise RuntimeError('采集保存回执身份不匹配；本次部署停止。')
    if result.get('status') != 'completed':
        raise RuntimeError('采集保存或退出失败：'+str(result.get('reason')))
    print(f'采集器 PID {pid} 已确认保存并退出。',flush=True)
