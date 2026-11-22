"""Manual terminal/PICO episode controls around the official LeRobot loops.

LeRobot still owns frame sampling, images, dataset buffers, discard and save.
This adapter only gates starts and maps operator requests to its event flags.
"""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time
import tty

import zmq

from brainco_bridge import decode_manager
from episode_operator_controls import PicoEpisodeGesture


class OperatorExit(Exception):
    """An explicit exit before a new episode, or without saving a fragment."""


RESET_PROMPT='请人工复位，按 S、Enter、空格或左 Grip+A 确认；确认后还需再按一次开始。'


class ConsoleKeys:
    def __enter__(self):
        self.original=termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
        if self.original:tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if not select.select([sys.stdin],[],[],0)[0]:return None
        value=os.read(sys.stdin.fileno(),1)
        if not value:return 'eof'
        if value==b'\x1b':
            if select.select([sys.stdin],[],[],.03)[0]:
                suffix=os.read(sys.stdin.fileno(),2)
                return {b'[C':'right',b'OC':'right',b'[D':'left',b'OD':'left'}.get(suffix)
            return 'esc'
        return value.decode('ascii',errors='ignore').lower()

    def __exit__(self,*args):
        if self.original:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,self.original)


class Operator:
    def __init__(self,local,events):
        self.endpoint=f"tcp://127.0.0.1:{local.get('pico_port',5556)}"
        self.events=events
        self.lock=threading.RLock()
        self.commands=queue.Queue()
        self.gesture=PicoEpisodeGesture()
        self.stopped=threading.Event()
        self.thread=None
        self.console=None
        self.phase='BUSY'
        self.dataset=None
        self.finish=None
        self.next_phase='READY'
        self.pending_save=None
        self.pending_discard_notice=False
        self.session_saved=0
        self.save_confirmed=False
        self.save_uncertain=False
        self.segment_open=False
        self.failure_kind=None
        self.outcome=None
        self.history=[]
        self.control=None
        self.last_busy_notice=None

    def start(self):
        self.console=ConsoleKeys().__enter__()
        self.thread=threading.Thread(target=self._receive,daemon=True,name='lerobot-operator')
        self.thread.start()
        if self.control:self.control.start()

    def stop(self):
        self.stopped.set()
        if self.thread:self.thread.join(timeout=2)
        if self.console:self.console.__exit__()
        if self.control:self.control.stop()

    def request_save_exit(self):
        with self.lock:
            phase,frames=self.phase,self.frames()
            if self.failure_kind or self.save_uncertain:
                return dict(accepted=False,reason='采集器已有保存或录制错误，请查看原终端。')
            if phase in ('RECORDING','FROZEN'):
                if frames==1:
                    return dict(accepted=False,reason='当前只有1帧，不满足最少2帧保存要求；片段保留。')
                if frames==0:
                    self.dispatch('exit','deployment')
                    self.segment_open=False
                else:
                    self.dispatch('save_exit','deployment')
            elif phase in ('READY','WAIT','RESET'):
                if frames or self.segment_open:
                    return dict(accepted=False,reason='等候状态存在未确认片段，请在原终端处理。')
                self.dispatch('save_exit','deployment')
            elif phase=='BUSY':
                if self.segment_open and self.finish not in ('save','save_exit'):
                    return dict(accepted=False,reason='当前片段正在切换状态，请稍后重试。')
            else:
                return dict(accepted=False,reason='采集状态无法确认，保留进程。')
            self.events['stop_recording']=True
            return dict(accepted=True,phase=phase,frames=frames)

    def _receive(self):
        context=zmq.Context();sock=context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE,b'manager_state');sock.setsockopt(zmq.CONFLATE,1)
        sock.setsockopt(zmq.LINGER,0);sock.connect(self.endpoint)
        try:
            while not self.stopped.is_set():
                if sock.poll(10):
                    try:self.feed_manager(decode_manager(sock.recv()),time.monotonic())
                    except (ValueError,KeyError,TypeError):pass
                key=self.console.read()
                if key is not None:self.on_key(key)
                # Conflate bursty manager publications and keep this UI reader
                # from consuming a core or starving the official frame loop.
                self.stopped.wait(.01)
        finally:
            sock.close();context.term()

    def feed_manager(self,manager,now):
        with self.lock:
            event=self.gesture.feed(manager,now)
        if event=='toggle_save':self.dispatch('toggle','pico')
        elif event=='discard':self.dispatch('discard_continue','pico')

    def on_key(self,key):
        command={'s':'start',' ':'toggle','\n':'toggle','\r':'toggle',
            'q':'discard_ready','x':'discard_continue','r':'discard_reset',
            'y':'success','f':'failure','eof':'exit','\x03':'exit',
            'esc':'save_exit','right':'toggle','left':'discard_ready'}.get(key)
        if command:self.dispatch(command,'terminal')

    def stage(self,phase,dataset=None):
        with self.lock:
            self.phase=phase
            if dataset is not None:self.dataset=dataset
            while not self.commands.empty():
                try:self.commands.get_nowait()
                except queue.Empty:break
            self.gesture.reset(time.monotonic())

    def frames(self):
        return self.dataset.writer.episode_buffer['size'] if self.dataset is not None else 0

    def dispatch(self,command,source):
        with self.lock:
            phase=self.phase
            self.history.append(dict(command=command,source=source,phase=phase,at=time.time()))
            if phase=='BUSY':
                if command in ('exit','save_exit'):self.events['stop_recording']=True
                else:
                    now=time.monotonic()
                    if self.last_busy_notice is None or now-self.last_busy_notice>=2:
                        self.last_busy_notice=now
                        work='保存处理中' if self.finish in ('save','save_exit') else '采集状态切换中'
                        print(f'{work}，本次按键未执行；请等就绪提示后再按。',flush=True)
                return
            if phase in ('READY','WAIT','RESET'):
                if command in ('toggle','start'):
                    self.commands.put('start')
                    print('已收到复位确认；回到 READY 后再按一次开始。' if phase=='RESET' else
                          '已收到开始请求，正在检查输入；以“开始录制”提示为准。',flush=True)
                elif command in ('exit','save_exit'):self.commands.put('exit')
                elif command=='discard_ready':self.commands.put('cancel_continue')
                elif command=='discard_reset':self.commands.put('reset')
                return
            if phase not in ('RECORDING','FROZEN') or self.finish is not None:return
            if command in ('success','failure'):
                self.outcome=command=='success'
                print('任务结果已标记：'+('成功' if self.outcome else '失败')+'；仍需保存。',flush=True)
                return
            if command in ('toggle','save_exit'):
                if self.frames()<2:
                    print('不足2帧，暂不能保存；请继续录制或丢弃。',flush=True);return
                self.finish='save_exit' if command=='save_exit' else 'save'
                if command=='save_exit':self.events['stop_recording']=True
            elif command in ('discard_ready','discard_continue','discard_reset'):
                self.finish=command;self.events['rerecord_episode']=True
            elif command=='exit':self.finish='exit'
            else:return
            self.phase='BUSY'
            self.events['exit_early']=True
            self.last_busy_notice=None
            # Terminal acknowledgement is immediate; completion speech remains
            # in the actual save/clear path. Never run speech RPCs here.
            if self.finish in ('save','save_exit'):
                print('已收到停止并保存请求，正在结束录制；保存完成会另行提示。',flush=True)
            elif self.finish.startswith('discard_'):
                print('已收到丢弃请求，正在结束录制；清除完成会另行提示。',flush=True)

    def take(self,timeout=.03):
        try:return self.commands.get(timeout=timeout)
        except queue.Empty:return None

    def confirm_saved(self,announce,followup=''):
        pending=self.pending_save
        if pending is None or not self.save_confirmed:return False
        path=Path(self.dataset.root)/'meta/sonic_outcomes'
        path.mkdir(exist_ok=True)
        (path/f"episode_{pending['episode']:06d}.json").write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n')
        announce(f"第{pending['episode']}条已保存，共{pending['frames']}帧。本次累计保存{self.session_saved}条。"+followup)
        self.pending_save=None
        self.save_confirmed=False
        return True

    def confirm_discarded(self,announce):
        self.segment_open=False
        self.pending_discard_notice=True
        announce('本段已丢弃。'+(RESET_PROMPT if self.next_phase=='RESET' else '等待你重新开始。'))

    def announce_summary(self,announce,*,failed=False,cleanup_failed=False):
        if cleanup_failed:
            announce(f'退出收尾失败，本次有{self.session_saved}条完成保存，请检查数据完整性。')
            return
        text=('采集异常结束。' if failed else '本次采集结束。')+f'本次共保留{self.session_saved}条。'
        if self.save_uncertain:text+='当前段未确认保存，请检查数据。'
        elif self.segment_open:text+='当前未保存片段不保留。'
        announce(text)

    def wait_for_start(self,robot,teleop,dataset,announce):
        mode=self.next_phase;auto=mode=='CONTINUE'
        self.stage('RESET' if mode=='RESET' else 'READY',dataset)
        reset_prompt=RESET_PROMPT
        save_announced=self.confirm_saved(announce,reset_prompt if mode=='RESET' else '')
        discard_announced=self.pending_discard_notice
        if discard_announced:
            self.pending_discard_notice=False
        if mode=='RESET':
            if not save_announced and not discard_announced:announce(reset_prompt)
            while True:
                command=self.take()
                if command=='exit' or self.events['stop_recording']:raise OperatorExit()
                if command=='start':break
            self.stage('READY',dataset)
        announced=False;spoken=bool(discard_announced);waiting_spoken=False;last_notice=0
        while True:
            if self.events['stop_recording']:raise OperatorExit()
            ready=True;reason=''
            try:
                robot.get_observation();teleop.get_action()
            except (TimeoutError,ValueError,RuntimeError) as error:
                ready=False;reason=str(error)
            command=self.take()
            if command=='exit':raise OperatorExit()
            if command=='reset':
                self.next_phase='RESET'
                return self.wait_for_start(robot,teleop,dataset,announce)
            if command=='cancel_continue':auto=False
            if not ready:
                self.phase='WAIT';announced=False
                if not waiting_spoken:
                    announce('输入尚未就绪，等待相机和机器人数据，尚未开始录制。')
                    waiting_spoken=True
                if time.monotonic()-last_notice>2 or command=='start':
                    print(f'WAIT episode {dataset.num_episodes}：尚未录制；{reason}',flush=True)
                    last_notice=time.monotonic()
                continue
            self.phase='READY'
            if not announced and not auto:
                print(f'READY episode {dataset.num_episodes}：尚未录制；按 S / Enter / 空格或左 Grip+A 开始。',flush=True)
                if not spoken:announce(f'第{dataset.num_episodes}条已就绪，等待你按开始。');spoken=True
                announced=True
            if command=='start' or auto:
                # Check again at the actual start, after processing the request.
                try:
                    robot.get_observation();teleop.get_action()
                except (TimeoutError,ValueError,RuntimeError):
                    announced=False
                    continue
                with self.lock:
                    if self.events['stop_recording']:raise OperatorExit()
                    self.finish=None;self.outcome=None
                    self.events['exit_early']=False
                    self.events['rerecord_episode']=False
                    self.phase='RECORDING'
                    self.gesture.reset(time.monotonic())
                return


@contextmanager
def operator_recording(local,*,voice_enabled=True,control=None):
    """Keep the official recorder/save lifecycle; add explicit human start gates."""
    from lerobot.scripts import lerobot_record
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils import utils
    from lerobot_robot_sonic import CameraStreamUnavailable
    original_loop=lerobot_record.record_loop
    original_listener=lerobot_record.init_keyboard_listener
    original_log=lerobot_record.log_say
    original_save=LeRobotDataset.save_episode
    original_clear=LeRobotDataset.clear_episode_buffer
    events=dict(exit_early=False,stop_recording=False,rerecord_episode=False)
    operator=Operator(local,events)
    if control:control.attach(operator)
    cleanup_started=False
    cleanup_cause=None

    def announce(text):utils.log_say(text,voice_enabled)

    def listener():
        operator.start()
        return operator,events

    def log(text,*args,**kwargs):
        nonlocal cleanup_started,cleanup_cause
        if text.startswith('Recording episode ') or text in ('Reset the environment','Re-record episode'):
            return  # These are announced only after the operator's actual choice.
        if text=='Stop recording':
            cleanup_started=True
            cleanup_cause=sys.exc_info()[1]
            return original_log(text,False)
        if text=='Exiting':return original_log(text,False)
        return original_log(text,*args,**kwargs)

    @wraps(original_save)
    def save_episode(dataset,*args,**kwargs):
        if dataset is not operator.dataset or operator.pending_save is None:
            return original_save(dataset,*args,**kwargs)
        announce('正在保存，请稍候。')
        try:
            result=original_save(dataset,*args,**kwargs)
        except BaseException:
            operator.failure_kind='save'
            operator.save_uncertain=True
            announce('保存未完成，当前段未确认保存，请查看采集终端。')
            raise
        with operator.lock:
            operator.session_saved+=1
            operator.save_confirmed=True
            operator.segment_open=False
            operator.pending_save['save_completed_at']=time.time()
        operator.confirm_saved(announce)
        return result

    @wraps(original_clear)
    def clear_episode(dataset,*args,**kwargs):
        tracked=dataset is operator.dataset and operator.finish in ('discard_ready','discard_continue','discard_reset')
        try:
            result=original_clear(dataset,*args,**kwargs)
        except BaseException:
            if tracked:
                operator.failure_kind='discard'
                announce('丢弃失败，当前段未确认清除，请查看采集终端。')
            raise
        if tracked:operator.confirm_discarded(announce)
        return result

    @wraps(original_loop)
    def manual_loop(*args,**kwargs):
        dataset=kwargs.get('dataset')
        if dataset is None:return  # Human reset happens after the official save.
        operator.wait_for_start(kwargs['robot'],kwargs['teleop'],dataset,announce)
        with operator.lock:
            if events['stop_recording']:raise OperatorExit()
            operator.segment_open=True
        original_log(f'Recording episode {dataset.num_episodes}',False)
        announce(f'第{dataset.num_episodes}条开始录制。')
        print(f'RECORDING episode {dataset.num_episodes}：左 Grip+A / Enter 保存；左 Grip+B / X / Q 丢弃后等待重新开始。',flush=True)
        writer=getattr(dataset,'writer',None)
        image_writer=getattr(writer,'image_writer',None)
        writer_settings=(image_writer.num_processes,image_writer.num_threads) if image_writer is not None else None
        input_interruption=None
        try:
            result=original_loop(*args,**kwargs)
        except (OperatorExit,KeyboardInterrupt):
            raise
        except CameraStreamUnavailable as error:
            # The official loop added no frame for this failed observation.
            # Its safe-stop wrapper drains/stops the image writer on exceptions;
            # restore the writer so the valid prefix can be saved and later
            # episodes can record normally. Never continue across the gap.
            operator.failure_kind='record'
            if writer_settings is not None:
                writer.stop_image_writer()
                writer.start_image_writer(num_processes=writer_settings[0],num_threads=writer_settings[1])
            operator.failure_kind=None
            input_interruption=str(error)
            result=None
        except Exception:
            operator.failure_kind='record'
            announce('录制中断，当前段未保存，请查看采集终端。')
            raise
        if operator.finish is None:
            operator.stage('FROZEN')
            if input_interruption is not None:
                print('相机输入中断，停止追加：'+input_interruption,flush=True)
                print(f'已保留本段前 {operator.frames()} 帧有效数据，尚未保存；Enter / 左 Grip+A 保存前段，X / 左 Grip+B 丢弃。恢复相机、复位后重新开始。',flush=True)
                announce('相机输入中断，本段已暂停，已有数据尚未保存。请选择保存前段或丢弃重录。')
            else:
                print('已到设定时长，停止追加；请保存或丢弃当前段。',flush=True)
                announce('已到设定时长，录制暂停，等待保存或丢弃。')
            while operator.finish is None:
                if operator.stopped.wait(.03):raise OperatorExit()
        finish=operator.finish
        if finish=='exit':raise OperatorExit()
        if finish in ('save','save_exit'):
            operator.pending_save=dict(episode=dataset.num_episodes,frames=operator.frames(),
                task_success=operator.outcome,saved_at=time.time(),operator_events=operator.history[:])
            operator.next_phase='RESET'
        else:
            operator.next_phase={'discard_ready':'READY','discard_continue':'READY',
                                 'discard_reset':'RESET'}[finish]
        operator.phase='BUSY'
        return result

    try:
        lerobot_record.init_keyboard_listener=listener
        lerobot_record.log_say=log
        lerobot_record.record_loop=manual_loop
        LeRobotDataset.save_episode=save_episode
        LeRobotDataset.clear_episode_buffer=clear_episode
        try:
            yield operator
        except BaseException as error:
            cleanup_failed=cleanup_started and error is not cleanup_cause
            cancelled=isinstance(error,(OperatorExit,KeyboardInterrupt)) and not cleanup_failed
            if not cancelled and not operator.failure_kind and not cleanup_failed:
                announce('采集异常，请查看终端错误信息。')
            operator.announce_summary(announce,failed=not cancelled,cleanup_failed=cleanup_failed)
            raise
        else:
            operator.announce_summary(announce)
    finally:
        operator.stop()
        lerobot_record.record_loop=original_loop
        lerobot_record.init_keyboard_listener=original_listener
        lerobot_record.log_say=original_log
        LeRobotDataset.save_episode=original_save
        LeRobotDataset.clear_episode_buffer=original_clear
