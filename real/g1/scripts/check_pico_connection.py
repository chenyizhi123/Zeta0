#!/usr/bin/env python3
"""Read PICO tracking through the SDK without publishing robot commands."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def safe_pose(value):
    array = np.asarray(value, dtype=float)
    return array.tolist() if np.isfinite(array).all() else None


def summarize_frames(frames):
    """Keep transport and body timestamps separate; stationary poses are valid."""
    transport = set(); body_stamps = set(); body_frames = 0; invalid = 0
    last_stamp = None; last_body_received = None; gaps = []; resets = 0
    previous_positions = None; max_jump = 0.; norms = []; last_body = None
    inconsistent = 0
    for frame in frames:
        if frame['timestamp_ns'] > 0:
            transport.add(frame['timestamp_ns'])
        pose = frame.get('body_poses')
        if not frame.get('body_read_consistent',True):
            inconsistent += 1
            continue
        if pose is None:
            if frame.get('body_available'):invalid += 1
            continue
        body = np.asarray(pose, dtype=float)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            invalid += 1; continue
        qnorm = np.linalg.norm(body[:, 3:], axis=1)
        norms.extend(qnorm.tolist())
        if np.any((qnorm < .5) | (qnorm > 1.5)):
            invalid += 1; continue
        body_frames += 1; last_body = body
        stamp = frame.get('body_timestamp_ns')
        if stamp is None or stamp <= 0 or stamp == last_stamp:continue
        body_stamps.add(stamp)
        if last_stamp is not None:
            if stamp < last_stamp:resets += 1
            else:gaps.append((stamp-last_stamp)/1e6)
        last_stamp = stamp; last_body_received = frame['received_monotonic']
        positions = body[:, :3]-body[0, :3]
        if previous_positions is not None:
            max_jump = max(max_jump, float(np.linalg.norm(positions-previous_positions, axis=1).max()))
        previous_positions = positions
    age = ((frames[-1]['received_monotonic']-last_body_received)*1000
           if frames and last_body_received is not None else None)
    result = dict(distinct_timestamps=len(transport), tracking_received=len(transport)>=3,
        full_body_frames=body_frames, distinct_body_timestamps=len(body_stamps),
        full_body_tracking_received=len(body_stamps)>=3 and age is not None and age<=500,
        last_body_update_age_ms=age, invalid_body_frames=invalid,
        inconsistent_body_reads=inconsistent,
        body_timestamp_resets=resets, max_body_timestamp_gap_ms=max(gaps,default=0.),
        max_root_relative_joint_step_m=max_jump,
        quaternion_norm_range=[min(norms),max(norms)] if norms else None,
        body_timestamp_api_available=any(f.get('body_timestamp_ns') is not None for f in frames),
        robot_commands_sent=False)
    if last_body is not None:
        result.update(body_shape=[24,7], last_body_poses=last_body.tolist())
    return result


def active_control_processes():
    active=[]
    for command_file in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=command_file.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        names={Path(arg.decode(errors='replace')).name for arg in command if arg}
        if names & {'pico_manager.py','g1_deploy_onnx_ref'}:
            active.append(int(command_file.parent.name))
    return active


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=10)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'logs/pico-status.json')
    parser.add_argument('--raw-output',type=Path,help='Optional raw SDK snapshots (JSONL); no robot commands')
    args=parser.parse_args()
    if args.duration<=0:parser.error('--duration must be positive')
    # A second SDK client's close() can cancel the existing manager stream.
    active=active_control_processes()
    if active:
        parser.error(f'身体部署或 PICO manager 仍在运行（PID: {active}）；先正常退出控制，再运行独立 SDK 只读检查。')
    import xrobotoolkit_sdk as xrt
    body_timestamp=getattr(xrt,'get_body_timestamp_ns',None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=None
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True,exist_ok=True)
        raw=args.raw_output.open('w')
    frames=[];started=time.monotonic();last_saved=started
    print('只读 PICO 检查：不启动 SONIC，不发布机器人指令。请保持站直，依次缓慢抬左手、右手和左右脚。',flush=True)
    initialized=False
    try:
        xrt.init();initialized=True
        started=time.monotonic();last_saved=started
        while time.monotonic()-started<args.duration:
            stamp=int(body_timestamp()) if body_timestamp else None
            available=bool(xrt.is_body_data_available())
            pose=safe_pose(xrt.get_body_joints_pose()) if available else None
            after=int(body_timestamp()) if body_timestamp else None
            frame=dict(received_monotonic=time.monotonic(),timestamp_ns=int(xrt.get_time_stamp_ns()),
                body_timestamp_ns=stamp,body_available=available,
                body_poses=pose if stamp==after else None,
                body_read_consistent=stamp==after,headset_pose=safe_pose(xrt.get_headset_pose()),
                left_controller_pose=safe_pose(xrt.get_left_controller_pose()),
                right_controller_pose=safe_pose(xrt.get_right_controller_pose()),
                buttons_abxy=[bool(getattr(xrt,f'get_{key}_button')()) for key in 'ABXY'])
            frames.append(frame)
            if raw:raw.write(json.dumps(frame,allow_nan=False)+'\n')
            now=time.monotonic()
            if now-last_saved>=1:
                report=summarize_frames(frames)
                report.update(updated_at=time.time(),duration_seconds=now-started)
                temporary=args.output.with_suffix('.tmp')
                temporary.write_text(json.dumps(report,indent=2));temporary.replace(args.output)
                last_saved=now
            time.sleep(.02)
    finally:
        if initialized:xrt.close()
        if raw:raw.close()
    result=summarize_frames(frames)
    result.update(updated_at=time.time(),duration_seconds=time.monotonic()-started,
        raw_output=str(args.raw_output) if args.raw_output else None)
    if frames:
        for key in ('timestamp_ns','body_timestamp_ns','headset_pose','left_controller_pose','right_controller_pose'):
            result[key]=frames[-1][key]
        result['full_body_available']=frames[-1]['body_available']
    args.output.write_text(json.dumps(result,indent=2))
    summary={key:value for key,value in result.items() if key!='last_body_poses'}
    print(json.dumps(summary,indent=2))
    print('报告：'+str(args.output),flush=True)


if __name__=='__main__':main()
