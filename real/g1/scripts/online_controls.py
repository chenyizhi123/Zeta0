"""Local V/M/F controls. No sockets or actuator commands are created on import."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
import hashlib
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
import uuid

import msgpack
import numpy as np

from online_runtime import KIT, import_runtime

import_runtime()
from psi.deploy.real_sonic_client import check_health
from psi.deploy.real_sonic_protocol import FORMAT, HORIZON, PERIOD, PROTOCOL, STATE_SCHEMA
from psi.deploy.sonic_wire import build_planner_message, pack_pose_message
from psi.sonic_actions import controller_command_values
from online_rtc import AdaptiveRTCTiming, RTCActionQueue, RTCBudgetExceeded, RTC_SCHEDULER


def scalar(packet, name):
    value = np.asarray(packet[name], dtype=float)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError('Invalid native control feedback: ' + name)
    return float(value.reshape(-1)[0])


def require_controller(packet, now=None, pid=None):
    if 'vmf_control_version' not in packet:
        raise RuntimeError('身体进程缺少 V/M/F 接口；请正常退出旧部署后重新运行独立模型部署命令。')
    if scalar(packet, 'vmf_control_version') != 1:
        raise RuntimeError('身体进程缺少 V/M/F 接口；请正常退出旧部署后重新运行独立模型部署命令。')
    if pid is not None and scalar(packet, 'controller_pid') != pid:
        raise RuntimeError('SONIC 身体进程已更换，停止本次模型会话。')
    if now is not None:
        for name in ('controller_time_monotonic', 'vmf_state_time'):
            if not 0 <= now - scalar(packet, name) <= .2:
                raise RuntimeError('身体控制模式反馈过期。')
    if scalar(packet, 'control_active') != 1:
        raise RuntimeError('SONIC 未处于活动控制。')


def initial_frame(task, manifest, trajectory_dir=None):
    root = Path(trajectory_dir or KIT/'trajectories'/task).expanduser().resolve()
    spec_path = root/'trajectory.json'
    spec = json.loads(spec_path.read_text())
    expected = {'format': 'zeta0.offline_trajectory.v1', 'task_key': task,
                'instruction': manifest['task'], 'model_weights_sha256': manifest['weights_sha256'],
                'storage_fps': manifest['temporal_abi']['action_hz'], 'observation_contract': manifest['observation_contract']}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError('首帧轨迹与任务、权重或动作接口不匹配。')
    relative = spec['demonstration_actions_file']
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Invalid first-frame trajectory path')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['files_sha256'][relative]:
        raise ValueError('首帧动作文件校验失败。')
    with np.load(path, allow_pickle=False) as data:
        actions = data['actions']
        if actions.shape != (spec['frames'], 76) or not len(actions):
            raise ValueError('Invalid first-frame action array')
        action = np.asarray(actions[0], dtype=np.float32).copy()
    projection = (manifest['observation_contract'] or {}).get('action_projection')
    canonical = controller_command_values(action, projection=projection)
    if not np.isfinite(action).all() or not np.allclose(action, canonical, rtol=0, atol=1e-7):
        raise ValueError('示范首帧不符合模型 token/手指范围。')
    return action, dict(directory=str(root), frame_index=0, actions_sha256=digest,
                        trajectory_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest())


class TerminalKeys(AbstractContextManager):
    def __init__(self):
        self.escape_state = 'normal'

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError('V/M/F 控制需要交互终端。')
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # Keep Ctrl+C/SIGINT; restore settings on every exit.
        return self

    def read(self):
        if select.select([self.fd], [], [], 0)[0]:
            value = os.read(self.fd, 64)
            if not value:
                raise EOFError('Control terminal closed')
            return self.decode(value.decode(errors='ignore'))
        return ''

    def decode(self, text):
        keys = []
        for char in text:
            if self.escape_state == 'normal':
                if char == '\x1b':
                    self.escape_state = 'escape'
                else:
                    keys.append(char.lower())
            elif self.escape_state == 'escape':
                self.escape_state = 'sequence' if char in '[O' else 'normal'
            elif '@' <= char <= '~':
                self.escape_state = 'normal'
        return ''.join(keys)

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class InteractiveSession:
    """One control-thread owner; HTTP work cannot send actions or change modes."""
    def __init__(self, io, transport, health, *, task, first_action, blend_seconds=2.,
                 speed=.4, worker=None, emit=None, now=None, expected_pid=None, key_hold_seconds=.8,audit=None):
        if not math.isfinite(blend_seconds) or blend_seconds < 1.:
            raise ValueError('首帧过渡时间至少 1 秒。')
        if not math.isfinite(speed) or not .1 <= speed <= .4:
            raise ValueError('手动速度需要在 0.1–0.4 m/s。')
        if not math.isfinite(key_hold_seconds) or not .1 <= key_hold_seconds <= 2.:
            raise ValueError('移动按键有效期需要在0.1–2秒。')
        self.action_hz, self.horizon = health['action_hz'], health['action_horizon']
        self.period = 1. / self.action_hz
        grid = {} if self.action_hz == 50 else dict(action_hz=self.action_hz, action_horizon=self.horizon)
        check_health(health, state_count=1, rtc_prefix_steps=15, **grid)
        io.configure_observation_contract(health['observation_contract'])
        self.io, self.transport, self.task = io, transport, task
        self.audit = audit
        self.first = np.asarray(first_action, dtype=np.float32).copy()
        if self.first.shape != (76,) or not np.isfinite(self.first).all():
            raise ValueError('Invalid first-frame action')
        self.blend_seconds, self.speed = blend_seconds, speed
        self.key_hold_seconds = key_hold_seconds
        self.last_velocity_request = None
        self.last_velocity_feedback = None
        self.worker = worker or ThreadPoolExecutor(max_workers=1, thread_name_prefix='vmf-http')
        self.emit = emit or (lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
        self.rtc_capacity = health['rtc_max_prefix_steps']
        self.timing = AdaptiveRTCTiming(self.rtc_capacity, action_hz=self.action_hz)
        self.queue = RTCActionQueue(rtc_max_prefix_steps=self.rtc_capacity, action_projection=io.action_projection, **grid)
        self.future = None
        self.epoch = time.time_ns()//1_000_000
        self.generation = 0
        self.pid = expected_pid
        self.ready = False
        self.feedback_seen = False
        self.feedback_error = None
        self.feedback_wait_logged = -float('inf')
        self.last_action = None
        self.mode = 'v'
        self.selected = time.monotonic() if now is None else now
        self.movement = (0., 0., 0.)
        self.movement_until = 0.
        self.facing_yaw = 0.
        self.facing_initialized = False
        self.hand_hold = None
        self._new_queue()
        self.emit(dict(event='mode', mode='v', message='零速度站立；f 首帧，m 模型，长按W/S前后、A/D左右，QE转向，空格立即回中，O/Ctrl+C停控退出。',
                       key_hold_seconds=self.key_hold_seconds))

    def _new_queue(self):
        self.generation += 1
        self.session_id = uuid.uuid4().hex
        self.queue.reset(self.session_id)
        self.reset_done = False
        self.request_seq = 0
        self.model_active = False
        self.timing.reset()
        self.pending_rtc = None
        self.last_rtc_prefix_steps = None
        self.observation_wait_since = None
        self.observation_wait_logged = -float('inf')

    def select(self, key, now):
        if key == 'm' and not (self.mode == 'first_hold' and self.ready):
            self.emit(dict(event='key_rejected', key='m', message='先按 f，等待首帧参考过渡完成。'))
            return False
        if key not in ('v', 'f', 'm'):
            return False
        entering_planner=key=='v' and self.mode!='v'
        if key == 'f':
            if not self.feedback_seen:
                self.emit(dict(event='key_rejected', key='f', message='等待身体和双手实际反馈。'))
                return False
            try:
                self.io.check_feedback(now)
            except ValueError as exc:
                self.pause_feedback(now, exc)
                return False
            token = np.asarray(self.io.controller_status['token_state'], dtype=np.float32)
            hands = np.concatenate([sample[1] for sample in self.io.hand_samples])
            if token.shape != (64,) or not np.isfinite(token).all():
                raise RuntimeError('缺少当前身体参考 token，不能直接跳到示范首帧。')
            self.blend_start = np.concatenate((token, hands)).astype(np.float32)
            self.blend_started = None
        self._new_queue()  # Revokes every old reply, including HTTP still in flight.
        self.epoch += 1
        self.selected = now
        self.ready = False
        self.mode = {'v': 'v', 'f': 'first_entry', 'm': 'model_priming'}[key]
        self.movement, self.movement_until = (0., 0., 0.), 0.
        self.hand_hold = None
        if key == 'v':
            if entering_planner:
                # Native reinitializes the planner context and captures the
                # current IMU heading when leaving streamed f/m control. Its
                # new local yaw zero is the current body heading.
                self.facing_yaw=0.
                self.facing_initialized=True
            self.io.stream_selected_at = None
            self.io.stream_mode_acknowledged = False
        self.emit(dict(event='mode', key=key, mode=self.mode, generation=self.generation,session_id=self.session_id))
        return True

    def keys(self, keys, now):
        for key in keys:
            if key in ('o', '\x03'):
                raise KeyboardInterrupt
            if self.feedback_error is not None and key not in ('v', ' '):
                continue
            if key in ('v', 'f', 'm'):
                self.select(key, now)
            elif self.mode == 'v':
                if key in 'wasd':
                    movement = {'w': (1., 0., 0.), 's': (-1., 0., 0.),
                                'a': (0., 1., 0.), 'd': (0., -1., 0.)}[key]
                    if movement != self.movement or now >= self.movement_until:
                        self.emit(dict(event='velocity_key',key=key,local_direction=movement,
                                       speed=self.speed,key_hold_seconds=self.key_hold_seconds))
                    self.movement = movement
                    self.movement_until = now+self.key_hold_seconds
                elif key in 'qe':
                    self.facing_yaw += .08 if key == 'q' else -.08
                elif key == ' ':
                    self.movement_until = 0.

    def command(self, planner):
        self.io.pose.send(pack_pose_message({
            'start': np.array([True]), 'stop': np.array([False]),
            'planner': np.array([planner]), 'vmf_epoch': np.array([self.epoch], dtype=np.int64)
        }, topic='command', version=1))
        self.io.active = True

    def send_pose(self, action, *, source='reference'):
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (76,) or not np.isfinite(value).all() or np.any((value[64:] < 0) | (value[64:] > 1)):
            raise ValueError('Invalid interpolated first-frame reference')
        # Initial-pose interpolation is continuous in the SONIC latent space.
        # Model predictions still pass through their trained FSQ projection.
        self.io.pose.send(pack_pose_message(dict(token_state=value[:64].reshape(1, 64),
            frame_index=np.array([self.io.frame_index], dtype=np.int64),
            vmf_epoch=np.array([self.epoch], dtype=np.int64)), topic='pose', version=4))
        self.io.hand_commands.send(msgpack.packb(dict(format='oraclevla.brainco12.command.v1',
            positions=value[64:].tolist(), sequence=self.io.frame_index), use_bin_type=True))
        if self.audit:
            self.audit.record('published_action',session_id=self.session_id,generation=self.generation,
                mode=self.mode,epoch=self.epoch,frame_index=self.io.frame_index,action=value.copy(),
                source=source,stage='sonic_token64_and_hand_bridge_target12')
        self.io.frame_index += 1
        self.io.active = True
        self.last_action = value.copy()

    def native_confirmed(self, planner):
        status = self.io.controller_status
        return (scalar(status, 'vmf_epoch') == self.epoch and
                scalar(status, 'input_control_mode') == (0 if planner else 1) and
                scalar(status, 'vmf_watchdog_enabled') == 1 and
                scalar(status, 'vmf_watchdog_tripped') == 0 and
                (scalar(status, 'planner_ready') == 1 if planner else scalar(status, 'encoder_active') == 0))

    def idle(self, now):
        movement = self.movement if now < self.movement_until else (0., 0., 0.)
        moving = any(movement)
        # Rotate local forward/left into the planner's global reference frame,
        # using the same facing convention as the existing planner streamer.
        facing = (math.cos(self.facing_yaw), math.sin(self.facing_yaw), 0.)
        x,y,_ = movement
        global_movement = (facing[0]*x-facing[1]*y,facing[1]*x+facing[0]*y,0.)
        request = (bool(moving),movement,self.speed)
        if request != self.last_velocity_request:
            self.emit(dict(event='velocity_request',locomotion_mode=1 if moving else 0,
                           movement=global_movement,facing=facing,speed=self.speed if moving else 0.))
            self.last_velocity_request = request
        self.io.pose.send(build_planner_message(1 if moving else 0, global_movement, facing,
                                              speed=self.speed if moving else 0., height=-1.))
        if self.audit:
            self.audit.record('planner_command',mode=self.mode,epoch=self.epoch,
                locomotion_mode=1 if moving else 0,movement=global_movement,facing=facing,
                speed=self.speed if moving else 0.)
        self.command(True)
        if (self.feedback_error is None and self.hand_hold is None
                and all(sample is not None for sample in self.io.hand_samples)):
            self.hand_hold = np.concatenate([sample[1] for sample in self.io.hand_samples])
        packet = dict(format='oraclevla.brainco12.command.v1', stop=True)
        if self.feedback_error is None and self.hand_hold is not None:
            packet = dict(format='oraclevla.brainco12.command.v1', positions=self.hand_hold.tolist(),
                          sequence=self.io.frame_index)
        self.io.hand_commands.send(msgpack.packb(packet, use_bin_type=True))

    def fallback(self, now, reason):
        self.select('v', now)
        self.emit(dict(event='fallback', mode='v', reason=str(reason), automatic_model_resume=False))
        self.idle(now)

    def pause_feedback(self, now, reason):
        first_failure = self.feedback_error is None
        self.feedback_error = str(reason)
        self.feedback_seen = False
        self.movement, self.movement_until = (0., 0., 0.), 0.
        self.hand_hold = None
        if first_failure:
            self.fallback(now, '反馈不可用，保留SONIC零速控制：' + str(reason))
        else:
            self.idle(now)
        if now - self.feedback_wait_logged >= 1.:
            self.emit(dict(event='feedback_wait', mode='v', reason=str(reason),
                           automatic_model_resume=False, sonic_kept_running=True))
            self.feedback_wait_logged = now

    def step(self, tick, now):
        status = self.io.controller_status
        if status is None:
            if now-self.selected > 3.:
                raise RuntimeError('接管后未收到身体状态。')
            return
        require_controller(status, now, self.pid)
        if self.audit:
            self.audit.record('body_feedback',mode=self.mode,session_id=self.session_id,
                generation=self.generation,epoch=self.epoch,
                body_q=self.io.body_sample[1].copy() if getattr(self.io,'body_sample',None) else status.get('body_q'),
                body_received=self.io.body_sample[0] if getattr(self.io,'body_sample',None) else None,
                body_dq=status.get('body_dq'),token_state=status.get('token_state'),
                motor_target=status.get('last_action'),
                base_quat=status.get('base_quat'),base_ang_vel=status.get('base_ang_vel'),
                native_time=status.get('controller_time_monotonic'),
                native_input_frame=status.get('vmf_token_input_frame'),
                native_input_sequence=status.get('vmf_token_input_sequence'),
                input_control_mode=status.get('input_control_mode'),encoder_active=status.get('encoder_active'),
                locomotion_mode=status.get('locomotion_mode'),
                hand_received=[sample[0] if sample is not None else None for sample in self.io.hand_samples],
                hand_age_ms=[(now-sample[0])*1000 if sample is not None else None for sample in self.io.hand_samples],
                hand_positions=[sample[1].copy() if sample is not None else None for sample in self.io.hand_samples])
        if self.pid is None:
            self.pid = scalar(status, 'controller_pid')
        if not self.facing_initialized:
            facing = np.asarray(status['planner_facing'], dtype=float)
            if facing.shape != (3,) or not np.isfinite(facing).all() or not np.linalg.norm(facing[:2]):
                raise RuntimeError('身体朝向反馈无效。')
            self.facing_yaw = math.atan2(facing[1], facing[0])
            self.facing_initialized = True
        try:
            self.io.check_feedback(now)
        except ValueError as exc:
            self.pause_feedback(now, exc)
            return
        if self.feedback_error is not None:
            self.feedback_error = None
            self.emit(dict(event='feedback_ready', mode=self.mode, automatic_model_resume=False,
                           message='反馈已恢复，SONIC保持零速；由操作者重新选择模式。'))
        self.feedback_seen = True
        if self.mode == 'v':
            if 'locomotion_mode' in status:
                feedback = scalar(status,'locomotion_mode')
                if feedback != self.last_velocity_feedback:
                    self.emit(dict(event='velocity_feedback',locomotion_mode=feedback,
                                   planner_ready=scalar(status,'planner_ready')))
                    self.last_velocity_feedback = feedback
            self.idle(now)
        if not self.native_confirmed(self.mode == 'v'):
            if scalar(status, 'vmf_watchdog_tripped') == 1 and self.mode != 'v':
                self.fallback(now, '身体端检测到控制或 token 超时；需重新按 f / m。')
                return
            if now-self.selected > 5.:
                raise RuntimeError('身体未确认所请求的控制模式。')
        if self.mode == 'v':
            self._network(now, tick, enabled=False)
            return
        self.command(False)
        if self.mode == 'first_entry':
            self.send_pose(self.blend_start)
            if self.native_confirmed(False):
                self.blend_started = now
                self.mode = 'first_blend'
            return
        if self.mode == 'first_blend':
            alpha = min(1., (now-self.blend_started)/self.blend_seconds)
            action = ((1.-alpha)*self.blend_start+alpha*self.first).astype(np.float32)
            self.send_pose(action)
            if alpha >= 1. and self.native_confirmed(False) and np.allclose(
                    status['token_state'], self.first[:64], rtol=0, atol=1e-6):
                self.mode, self.ready = 'first_hold', True
                self.emit(dict(event='first_reference_ready', message='示范首帧参考已反馈，保持中；按 m 开始模型。'))
            return
        if self.mode == 'first_hold':
            self.send_pose(self.first)
            self._network(now, tick, enabled=False)
            return
        if self.mode in ('model_priming', 'model'):
            if not self.native_confirmed(False):
                if not self.model_active and now-self.selected <= .5:
                    # Mode feedback is asynchronous. Seed the new epoch while
                    # awaiting its routing ACK; do not mistake it for a failure.
                    self.send_pose(self.first)
                    return
                self.fallback(now, '模型运行期间身体未确认 token 模式。')
                return
            action = self.queue.pop(tick)
            if action is not None:
                action = controller_command_values(action, projection=self.io.action_projection)
                if not self.model_active:
                    self.emit(dict(event='model_running',rtc_capacity_steps=self.rtc_capacity,
                        initial_rtc_prefix_steps=0,queue_depth=self.queue.depth(now),scheduler=RTC_SCHEDULER))
                self.send_pose(action,source='policy')
                self.mode, self.model_active = 'model', True
            elif self.model_active:
                self.fallback(now, '模型动作队列耗尽，已清队列并请求零速度站立。')
                return
            else:
                self.send_pose(self.first)
            self._network(now, tick, enabled=True)

    def _network(self, now, tick, *, enabled):
        if self.future is not None and self.future[3].done():
            kind, generation, request, future = self.future
            context = self.pending_rtc
            self.future = None
            self.pending_rtc = None
            try:
                reply = future.result()
                if not enabled or generation != self.generation:
                    return
                if kind == 'reset':
                    if reply.get('session_id') != self.session_id:
                        raise ValueError('Server reset session mismatch')
                    self.reset_done = True
                else:
                    received=time.monotonic()
                    elapsed=received-request['anchor_time']
                    prefix_count=len(request['committed_prefix'])
                    self.timing.observe(elapsed,bootstrap=prefix_count==0)
                    if prefix_count and elapsed > prefix_count*self.period:
                        raise RTCBudgetExceeded(
                            f'本次推理用时 {elapsed:.3f}s，超过已承诺的 {prefix_count} 帧前缀保护时间；需重新按 f / m。')
                    installed=self.queue.install(request, reply, received)
                    consumed=self.queue.consumed_steps-context['consumed_before']
                    delay_steps=math.ceil(elapsed/self.period-1e-9)
                    if self.audit:
                        self.audit.record('inference_accepted',session_id=request['session_id'],
                            request_seq=request['request_seq'],rtc_prefix_steps=prefix_count,
                            installed_steps=installed,queue_depth=self.queue.depth(received),
                            anchor_to_accept_s=elapsed,inference_seconds=reply.get('inference_seconds'),
                            actual_delay_steps=delay_steps,consumed_during_inference=consumed,
                            skipped_during_inference=self.queue.skipped_steps-context['skipped_before'],
                            scheduling=context['plan'],merge=self.queue.last_merge)
                    if prefix_count and abs(consumed-delay_steps)>1:
                        self.emit(dict(event='rtc_consumption_mismatch',request_seq=request['request_seq'],
                            consumed_steps=consumed,elapsed_delay_steps=delay_steps))
                    if not self.model_active and self.queue.depth(received)<=self.rtc_capacity:
                        # A cold first inference may leave too little horizon to
                        # protect the next one. Hold f and retry a fresh chunk.
                        self.emit(dict(event='model_priming_retry',reason='首个动作块剩余长度不足以保护下一次推理',
                                       remaining_steps=self.queue.depth(received)))
                        self.queue.reset(self.session_id)
            except RTCBudgetExceeded as exc:
                self.fallback(now,str(exc))
                return
            except Exception as exc:
                self.emit(dict(event='reply_rejected', reason=str(exc),kind=kind,
                    request_seq=request.get('request_seq') if request else None,
                    rtc_prefix_steps=len(request.get('committed_prefix',[])) if request else None))
                return
        if not enabled or self.future is not None:
            return
        if not self.reset_done:
            future = self.worker.submit(self.transport.call, '/reset', {'session_id': self.session_id})
            self.future = ('reset', self.generation, None, future)
            return
        # Like LeRobot's horizon-sized queue threshold, refill promptly after a
        # reply. Do not issue a second empty-prefix request before first playback.
        if not self.model_active and self.queue.depth(now):
            return
        if self.queue.depth(now) > self.horizon:
            return
        try:plan=self.timing.plan()
        except RTCBudgetExceeded as exc:
            self.fallback(now,str(exc))
            return
        prefix_steps = plan['rtc_prefix_steps'] if self.model_active else 0
        revision, prefix = self.queue.snapshot(tick, prefix_steps)
        if len(prefix) != prefix_steps:
            if self.model_active:
                self.fallback(now,'未来动作队列不足以覆盖 RTC 推理延迟；需重新按 f / m。')
            return
        try:
            observed = self.io.observation(tick)
        except ValueError as exc:
            if self.observation_wait_since is None:
                self.observation_wait_since = now
            if now-self.observation_wait_logged >= 1.:
                self.emit(dict(event='observation_wait', mode=self.mode, reason=str(exc),
                    waiting_seconds=now-self.observation_wait_since))
                self.observation_wait_logged = now
            if not self.model_active and now-self.observation_wait_since >= 2.:
                # Cancel this start attempt. Restored cameras must not trigger
                # an old, indefinitely armed m command without another keypress.
                self._new_queue()
                self.mode, self.ready = 'first_hold', True
                self.emit(dict(event='model_start_blocked', mode=self.mode, reason=str(exc),
                    message='模型输入连续 2 秒不可用，已取消启动并保持首帧；恢复后重新按 m。'))
            return  # Current future actions remain valid until their own deadlines.
        self.observation_wait_since = None
        self.observation_wait_logged = -float('inf')
        request = dict(observed, format=FORMAT, session_id=self.session_id, request_seq=self.request_seq,
            queue_revision=revision, anchor_time=tick, task=self.task, action_protocol_id=PROTOCOL,
            state_schema_id=STATE_SCHEMA, committed_prefix=prefix)
        self.request_seq += 1
        self.pending_rtc=dict(consumed_before=self.queue.consumed_steps,
                              skipped_before=self.queue.skipped_steps,plan=plan)
        if prefix_steps!=self.last_rtc_prefix_steps:
            self.emit(dict(plan,event='rtc_schedule',request_seq=request['request_seq'],
                           rtc_prefix_steps=prefix_steps,rtc_capacity_steps=self.rtc_capacity))
            self.last_rtc_prefix_steps=prefix_steps
        self.future = ('infer', self.generation, request, self.worker.submit(self.transport.infer, request))

    def close(self):
        self._new_queue()
        self.worker.shutdown(wait=False, cancel_futures=True)


def run_interactive(io, transport, health, *, task, first_action, blend_seconds=2., speed=.4, duration=0., emit=None,
                    expected_pid=None,key_hold_seconds=.8,audit=None):
    session = None
    try:
        session = InteractiveSession(io, transport, health, task=task, first_action=first_action,
                                     blend_seconds=blend_seconds, speed=speed, emit=emit, expected_pid=expected_pid,
                                     key_hold_seconds=key_hold_seconds,audit=audit)
        with TerminalKeys() as keys:
            started = time.monotonic()
            tick = (math.floor(started/session.period)+1)*session.period
            while not duration or time.monotonic()-started < duration:
                now = time.monotonic()
                io.poll(now)
                session.keys(keys.read(), now)
                if now < tick:
                    time.sleep(min(.001, tick-now))
                    continue
                now = time.monotonic()
                if now-tick >= session.period:
                    lag=now-tick
                    tick = math.floor(now/session.period)*session.period
                    skipped=session.queue.skip_to(tick)
                    if audit:
                        audit.record('control_loop_skip',mode=session.mode,lag_s=lag,
                                     missed_periods=int(lag/session.period),discarded_slots=skipped)
                session.step(tick, now)
                tick += session.period
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            io.close()
