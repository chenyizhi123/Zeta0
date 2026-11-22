"""Task-name convenience entry for the unmodified official LeRobot recorder."""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np

from lerobot_robot_sonic import (SonicRobot, SonicRobotConfig, SonicActions,
                                 SonicActionsConfig, STATE_NAMES, ACTION_NAMES)


ROOT = Path(__file__).resolve().parents[1]


def launch_camera_preview(config_path, cameras):
    """Run the plain camera window separately; display cannot stall recording."""
    status_path=ROOT/'logs/three-camera-live-window.json'
    try:
        status=json.loads(status_path.read_text())
        process=Path('/proc',str(status['pid']),'cmdline')
        if (status.get('live') and time.time()-status['updated_at']<3
                and status.get('cameras','all')==cameras and process.exists()
                and b'preview_live_cameras.py' in process.read_bytes()):
            return  # Reuse the existing camera window.
    except (OSError,ValueError,KeyError):
        pass
    local=json.loads(Path(config_path).read_text())
    python=Path(local['sonic_root'])/'.venv_sim/bin/python'
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/three-camera-live-window.log').open('a') as output:
        subprocess.Popen([str(python),str(ROOT/'scripts/preview_live_cameras.py'),
            '--config',str(config_path),'--cameras',cameras],stdin=subprocess.DEVNULL,
            stdout=output,stderr=output,start_new_session=True)


def resolve_task(args):
    if not args.task or not re.fullmatch(r'[\w.-]+', args.task) or args.task in ('.', '..'):
        raise ValueError('请提供只含文字、数字、下划线、点和横线的任务名')
    destination = args.root.resolve() / args.task
    resume = (destination / 'meta/info.json').exists()
    prompt = args.prompt
    if destination.exists():
        if not resume:
            raise ValueError('已有目录不是完整数据集，请换新任务名：' + str(destination))
        info = json.loads((destination / 'meta/info.json').read_text())
        if (info.get('robot_type') != 'sonic_g1' or info.get('codebase_version') != 'v3.0'
                or info.get('features', {}).get('observation.state', {}).get('names') != list(STATE_NAMES)
                or info.get('features', {}).get('action', {}).get('names') != list(ACTION_NAMES)):
            raise ValueError('已有任务属于旧采集结构，请换新任务名；旧数据保留：' + str(destination))
        from lerobot.datasets.io_utils import load_tasks
        # An interrupted first episode has info.json but no tasks.parquet yet.
        saved = load_tasks(destination).index.tolist() if info.get('total_tasks',0) else []
        if len(saved) == 1:
            if prompt is not None and prompt != saved[0]:
                raise ValueError('任务描述与已有数据不同，请换任务名')
            prompt = saved[0]
    if not prompt:
        raise ValueError('新任务需要 --prompt "英文任务描述"')
    return destination, resume, prompt


def check_inputs(robot_config, teleop_config, seconds, *, display=False):
    robot, teleop = SonicRobot(robot_config), SonicActions(teleop_config)
    count = 0
    ages = {name: [] for name in robot.cameras}
    identities = {name: set() for name in robot.cameras}
    try:
        if display:
            launch_camera_preview(robot_config.local_config,robot_config.recording_cameras)
        teleop.connect()
        robot.connect()
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            loop = time.perf_counter()
            robot.get_observation()
            teleop.get_action()
            count += 1
            for name, camera in robot.cameras.items():
                with camera.frame_lock:
                    timestamp = camera.latest_timestamp
                ages[name].append((time.perf_counter() - timestamp) * 1000)
                identities[name].add(timestamp)
            time.sleep(max(0, 1/30 - (time.perf_counter() - loop)))
        duration = time.perf_counter() - started
        result = dict(recorder='lerobot.scripts.lerobot_record', robot_commands_sent=False,
            dataset_created=False, duration_s=duration, successful_samples=count,
            sample_hz=count/duration, state_dimensions=41, action_dimensions=76,
            camera_pairing='independent official latest-frame reads',
            cameras={name:dict(observed_unique_hz=len(identities[name])/duration,
                              max_age_ms=max(values),p95_age_ms=float(np.percentile(values,95)))
                     for name,values in ages.items()})
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return result
    finally:
        robot.disconnect()
        teleop.disconnect()


def main(argv=None):
    # The official PNG encoder creates a process pool while camera workers are
    # alive. Spawn clean workers instead of inheriting those threads via fork.
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='官方 LeRobot 采集；SONIC/PICO 继续独立控制机器人')
    parser.add_argument('task', nargs='?')
    parser.add_argument('--prompt')
    parser.add_argument('--root', type=Path, default=ROOT/'data/tasks')
    parser.add_argument('--config', type=Path, default=ROOT/'config/local.json')
    parser.add_argument('--cameras', choices=('all','head'), default='all')
    parser.add_argument('--num-episodes', type=int, default=0,help='本次保存条数；0为不限')
    parser.add_argument('--episode-time-s', type=float, default=0,
                        help='可选的单段时长上限；0为手动结束，到上限后等待保存/丢弃')
    parser.add_argument('--reset-time-s', type=float, default=0,
                        help='兼容旧命令；当前复位由人工确认，不按时间自动开始')
    parser.add_argument('--display-data', action=argparse.BooleanOptionalAction, default=True,
                        help='使用独立普通窗口显示相机画面，默认开启')
    parser.add_argument('--voice', action=argparse.BooleanOptionalAction, default=True,
                        help='通过机器人扬声器中文播报录制状态，默认开启')
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True,
                        help='每秒显示官方实际录制帧数、帧率与时长，默认开启')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true', help='只读验证实际输入，不创建数据集')
    parser.add_argument('--seconds', type=float, default=15)
    args = parser.parse_args(argv)
    if (args.num_episodes < 0 or not np.isfinite([args.episode_time_s,args.reset_time_s,args.seconds]).all()
            or min(args.episode_time_s,args.reset_time_s) < 0 or args.seconds <= 0):
        parser.error('条数与录制时长不可为负，检查时长必须为正')
    robot_cfg = SonicRobotConfig(id='g1_sonic', local_config=str(args.config.resolve()),
                                recording_cameras=args.cameras)
    teleop_cfg = SonicActionsConfig(id='pico_sonic_actions', local_config=robot_cfg.local_config)
    if args.check:
        return check_inputs(robot_cfg,teleop_cfg,args.seconds,display=args.display_data)
    destination,resume,prompt = resolve_task(args)
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts.lerobot_record import RecordConfig, record
    config = RecordConfig(robot=robot_cfg,teleop=teleop_cfg,play_sounds=args.voice,
        display_data=False,resume=resume,
        dataset=DatasetRecordConfig(repo_id='local/'+args.task,root=destination,single_task=prompt,
            fps=30,num_episodes=args.num_episodes or 2**31-1,
            episode_time_s=args.episode_time_s or math.inf,
            reset_time_s=0,push_to_hub=False,no_stamp=True,encoder_threads=4))
    if args.dry_run:
        plan=asdict(config)
        plan['dataset']['episode_time_s']=args.episode_time_s or None
        plan['dataset']['num_episodes']=args.num_episodes or None
        print(json.dumps(dict(entrypoint='lerobot.scripts.lerobot_record.record',
            robot_commands_sent=False,automatic_start=False,
            recording_controls='terminal and PICO left Grip+A/B',config=plan),ensure_ascii=False,indent=2,default=str))
        return
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/online-model-control.lock').open('a+') as model_lock:
        try:
            fcntl.flock(model_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('模型正在部署或控制，不能同时启动另一任务的采集。') from error
    with (ROOT/'logs/task-collection.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('已有采集进程运行，请先保存并退出旧采集终端，再启动官方采集') from error
        # Recheck under the same exclusive lock used by the retired collector.
        resolve_task(args)
        metadata = dict(recorder='lerobot.scripts.lerobot_record.record',
            schema='sonic_g1_lerobot_v1',state_names=STATE_NAMES,action_names=ACTION_NAMES,
            state_layout='body29,left_hand6,right_hand6',
            action_layout='actual_sonic_token64,actual_left_hand_command6,actual_right_hand_command6',
            camera_reading='official ZMQ worker/read_latest with MessagePack transport adapter',
            camera_hardware_synchronized=False,robot_commands_sent_by_recorder=False,
            recording_control='manual_terminal_and_pico',automatic_start=False,
            started_at=time.time(),resumed=resume,
            local_config=json.loads(args.config.read_text()),runtime={})
        for name in ('sonic-body-runtime','pico-hand-runtime','brainco-bridge-runtime'):
            path=ROOT/'logs'/(name+'.json')
            if path.exists():metadata['runtime'][name]=json.loads(path.read_text())
        print('LeRobot采集：启动后等待开始。左 Grip+A / S / Enter / 空格开始；左 Grip+A / Enter / 空格保存。',flush=True)
        print('左 Grip+B / X / Q：丢弃后等待；R：人工复位重录；Ctrl+C退出。手柄组合键松开ABXY后生效。',flush=True)
        print('普通相机窗口：'+('开启' if args.display_data else '关闭')+
              '；机器人语音：'+('开启' if args.voice else '关闭')+
              '；帧数进度：'+('开启' if args.progress else '关闭'),flush=True)
        from recorder_handoff import RecorderControl
        control=RecorderControl(ROOT,destination)
        try:
            if args.display_data:
                launch_camera_preview(args.config,args.cameras)
            from lerobot_recording_feedback import recording_feedback,task_speech_label
            from lerobot_operator_controls import operator_recording,OperatorExit
            with recording_feedback(metadata['local_config'],voice_enabled=args.voice,
                                    progress_enabled=args.progress,task_label=task_speech_label(args.task,prompt)):
                try:
                    with operator_recording(metadata['local_config'],voice_enabled=args.voice,control=control):
                        return record(config)
                except (OperatorExit,KeyboardInterrupt):
                    if control.request_id and control.operator.segment_open:
                        control.note_error('保存退出被中断，当前片段未确认保存。')
                    print('已退出采集；本次保留条数见上方结算，已保存数据保留。',flush=True)
        except BaseException as error:
            control.note_error(error)
            raise
        finally:
            try:
                if (destination/'meta/info.json').exists():
                    metadata['finished_at']=time.time()
                    sessions=destination/'meta/sonic_sessions'
                    sessions.mkdir(exist_ok=True)
                    (sessions/(uuid.uuid4().hex+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
            except BaseException as error:
                control.note_error(error)
                raise
            finally:
                control.finish()


if __name__ == '__main__':
    main()
