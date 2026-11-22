"""V/M/F model deployment: velocity planner, demonstration first frame, model."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from urllib.error import URLError

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def process_ids(filename):
    result = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p/'cmdline').read_bytes().split(b'\0')
            if any(Path(a.decode(errors='replace')).name == filename for a in args if a):
                result.append(int(p.name))
        except OSError:
            pass
    return result


def mask_disabled_fingers(positions, hardware):
    value = np.asarray(positions, dtype=np.float64).copy()
    if value.shape != (6,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Expected six finite normalized hand targets')
    # The current hardware mapping explicitly keeps both pinkies open. Do not
    # re-enable them by replacing the PICO bridge with the model DDS bridge.
    profiles = hardware.get('grasp_input', {}).get('profiles', [])
    if not profiles or any(len(p.get('closed_pose', [])) != 6 for p in profiles):
        raise ValueError('Hardware finger endpoint policy is missing or invalid')
    if all(p['closed_pose'][5] == 0 for p in profiles):
        value[5] = 0.
    return value


def robot_config(task, model_paths_override=None):
    local = json.loads((ROOT/'config/local.json').read_text())
    from policy_profile import model_paths
    paths = model_paths(task, model_paths_override)
    manifest = json.loads((Path(paths['tasks'][task]['weights_dir'])/'inference_manifest.json').read_text())
    bound = manifest['observation_contract'] or {}
    endpoints = {name:f"tcp://{source['host']}:{source['port']}" for name,source in local['camera_sources'].items()}
    return dict(body_endpoint='tcp://127.0.0.1:5557',camera_sources=endpoints,
        hand_state_endpoint='tcp://127.0.0.1:5561',hand_command_endpoint='tcp://127.0.0.1:5562',
        action_bind='tcp://127.0.0.1:5556',state_history_count=1,
        real_observation_mode=bound.get('mode'),real_observation_contract=bound.get('collector'),
        real_action_projection=bound.get('action_projection')),manifest,local


def serve_hands(local):
    from online_runtime import import_runtime
    import_runtime()
    from psi.deploy.brainco_bridge import _NativeDdsTransport,serve_existing_dds
    hardware=json.loads((ROOT/'config/hand-hardware.json').read_text())
    if hardware.get('service_verified') is not True:
        raise RuntimeError('Hand service is not verified')
    class MaskedTransport(_NativeDdsTransport):
        def write(self,side,positions,speeds):
            return super().write(side,mask_disabled_fingers(positions,hardware),speeds)
    from robot_network import dds_config
    config=dict(state_bind='tcp://127.0.0.1:5561',command_bind='tcp://127.0.0.1:5562',
        dds_config=dds_config(local))
    transport=MaskedTransport(config,True)
    asyncio.run(serve_existing_dds(config,execute=True,transport=transport))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,help='Historical task or custom task declared by --model-paths')
    parser.add_argument('--model-paths',type=Path,default=None,help='覆盖所选任务的模型路径配置')
    parser.add_argument('--server',default='http://127.0.0.1:8014')
    parser.add_argument('--execute',action='store_true',help='自动启动模型服务和SONIC，进入f/v/m交互控制')
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--dry-run',action='store_true',help='仅显示独立部署流程，不连接机器人或启动服务')
    parser.add_argument('--skip-remote',action='store_true',help='相机和双手服务已就绪，跳过远端服务恢复')
    parser.add_argument('--no-handoff',action='store_true',help='保留旧采集和控制进程，仅在机器人空闲时启动')
    parser.add_argument('--handoff-timeout',type=float,default=120.,help='等待旧采集器保存退出的秒数，默认120')
    parser.add_argument('--trajectory-dir',help='配套轨迹目录；f 使用其中第 0 帧，默认所选任务的离线轨迹')
    parser.add_argument('--first-frame-seconds',type=float,default=2.,help='首帧参考过渡时间，至少 1 秒（默认 2）')
    parser.add_argument('--manual-speed',type=float,default=.4,help='v 模式 WASD 速度，0.1–0.4 m/s（默认 0.4）')
    parser.add_argument('--key-hold-seconds',type=float,default=.8,help='WASD按键有效期，0.1–2秒；默认0.8秒以覆盖键盘首次重复延迟')
    parser.add_argument('--diagnostic-images',action='store_true',help='后台保存每次实际模型输入图像，用于排查场景与抓取误差')
    parser.add_argument('--internal-hands',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.model_paths is not None:
        args.model_paths = args.model_paths.expanduser().resolve()
    if not np.isfinite(args.duration) or args.duration<0:
        parser.error('--duration must be finite and nonnegative')
    if not np.isfinite(args.first_frame_seconds) or args.first_frame_seconds<1:
        parser.error('--first-frame-seconds must be finite and at least 1')
    if not np.isfinite(args.manual_speed) or not .1<=args.manual_speed<=.4:
        parser.error('--manual-speed must be 0.1–0.4')
    if not np.isfinite(args.key_hold_seconds) or not .1<=args.key_hold_seconds<=2.:
        parser.error('--key-hold-seconds must be 0.1–2')
    if not np.isfinite(args.handoff_timeout) or args.handoff_timeout<=0:
        parser.error('--handoff-timeout must be positive and finite')
    config,manifest,local=robot_config(args.task, args.model_paths)
    if args.internal_hands:
        serve_hands(local);return
    from online_controls import initial_frame,run_interactive
    from online_rtc import RTC_SCHEDULER
    from autonomous_deployment import (AutonomousDeployment,STEPS,bootstrap_control,
                                       health_at,require_free_robot,validate_model)
    first_action,first_info=initial_frame(args.task,manifest,args.trajectory_dir)
    if not args.execute or args.dry_run:
        health=None
        if not args.dry_run:
            try:health=health_at(args.server)
            except (URLError,TimeoutError):pass
            if health is not None:validate_model(health,manifest)
        print(json.dumps(dict(task=args.task,server=health,robot_io=config,
            prompt=manifest['task'],initial_frame=first_info,startup_steps=STEPS,
            controls={'v':'velocity planner (starts at zero)','f':'blend to demonstration frame 0','m':'model after f'},
            body_process_ids=process_ids('g1_deploy_onnx_ref'),execute=False,
            pico_required=False,automatic_sonic_start=True,manual_speed=args.manual_speed,
            key_hold_seconds=args.key_hold_seconds,
            diagnostic_images=args.diagnostic_images,
            rtc=dict(scheduler=RTC_SCHEDULER,prefix_capacity_steps=manifest['rtc_max_prefix_steps'],
                     margin_steps=2)),ensure_ascii=False,indent=2));return
    if args.no_handoff:require_free_robot(process_ids)
    if not sys.stdin.isatty():
        raise RuntimeError('f/v/m控制需要交互终端；请在终端直接运行部署命令。')
    from psi.deploy.real_sonic_io import SonicIOConfig
    from persistent_sonic_io import PersistentSonicIO
    from runtime_audit import RuntimeAudit,audit_path,audited_http_client
    (ROOT/'logs').mkdir(exist_ok=True)
    os.environ['NO_PROXY']='127.0.0.1,localhost'
    with ExitStack() as stack:
        for filename in ('online-model-control.lock','sonic-start.lock'):
            lock=stack.enter_context((ROOT/'logs'/filename).open('a+'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Request a cooperative save before touching any old actuator writers.
        if not args.no_handoff:
            from urllib.error import HTTPError
            try:existing=health_at(args.server)
            except HTTPError as error:
                raise RuntimeError('已有模型服务健康检查失败；未切换旧采集和控制。') from error
            except (URLError,TimeoutError):existing=None
            if existing is not None:validate_model(existing,manifest)
            from deployment_handoff import prepare_handoff
            prepare_handoff(ROOT,local,process_ids,timeout=args.handoff_timeout)
        require_free_robot(process_ids)
        def interrupt(signum,frame):
            raise RuntimeError('模型客户端收到终止信号；保留SONIC，由操作者主动退出。')
        previous=signal.signal(signal.SIGTERM,interrupt)
        stack.callback(signal.signal,signal.SIGTERM,previous)
        deployment=stack.enter_context(AutonomousDeployment(ROOT,local))
        health=deployment.model(args.task,args.server,manifest,args.model_paths)
        audit=stack.enter_context(RuntimeAudit(audit_path(ROOT,'client'),dict(task=args.task,
            prompt=manifest['task'],weights_sha256=manifest['weights_sha256'],
            rtc_capacity_steps=health['rtc_max_prefix_steps'],action_hz=health['action_hz'],
            action_horizon=health['action_horizon'],first_frame=first_info,
            server_audit_log=health.get('rtc_audit_log'),rtc_scheduler=RTC_SCHEDULER,rtc_margin_steps=2,
            diagnostic_images=args.diagnostic_images),image_interval_s=.001 if args.diagnostic_images else None))
        print('RTC与动作诊断日志：'+str(audit.path),flush=True)
        if audit.images:print('实际模型输入图像（每次推理请求）：'+str(audit.images.path),flush=True)
        deployment.prepare_robot(skip_remote=args.skip_remote)
        env=dict(deployment.sdk_env)
        for key in list(env):
            if key.lower() in ('http_proxy','https_proxy','all_proxy','grpc_proxy'):
                env.pop(key)
        hand_command=[sys.executable,Path(__file__).resolve(),'--task',args.task,'--internal-hands']
        if args.model_paths is not None:
            hand_command.extend(['--model-paths', args.model_paths])
        child=deployment.launch(hand_command,
                                'online-model-hands.log',env=env)
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError('模型DDS手桥启动失败，查看 logs/online-model-hands.log。')
        io=PersistentSonicIO(SonicIOConfig(**config,execute=True))
        try:
            io.configure_observation_contract(health['observation_contract'])
            pid=deployment.start_body()
            if child.poll() is not None:
                raise RuntimeError('模型DDS手桥已退出，查看 logs/online-model-hands.log。')
            bootstrap_control(io,deployment.body)
            from model_video_recorder import ModelVideoRecorder
            videos=stack.enter_context(ModelVideoRecorder(local,
                audit.path.parent/(audit.path.stem+'-videos'),audit=audit))
            with (ROOT/'logs/online-model-events.jsonl').open('a',buffering=1) as events:
                def emit(event):
                    videos.on_event(event)
                    value=dict(event,time_monotonic=time.monotonic(),task=args.task)
                    line=json.dumps(value,ensure_ascii=False,allow_nan=False)
                    print(line,flush=True);events.write(line+'\n')
                    audit.record('client_event',data=value)
                emit(dict(event='session_start',prompt=manifest['task'],weights_sha256=manifest['weights_sha256'],
                          first_frame=first_info,blend_seconds=args.first_frame_seconds,
                          body_pid=pid,autonomous_sonic=True,key_hold_seconds=args.key_hold_seconds,
                          manual_speed=args.manual_speed,rtc_scheduler=RTC_SCHEDULER))
                run_interactive(io,audited_http_client(args.server,audit),health,task=manifest['task'],first_action=first_action,
                    blend_seconds=args.first_frame_seconds,speed=args.manual_speed,duration=args.duration,emit=emit,
                    expected_pid=pid,key_hold_seconds=args.key_hold_seconds,audit=audit)
        finally:
            if not io.context.closed:
                io.close()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print('部署已中止：'+str(error),file=sys.stderr,flush=True)
        raise SystemExit(1)
