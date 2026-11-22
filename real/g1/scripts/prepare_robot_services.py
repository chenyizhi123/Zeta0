"""Restore camera and read-only-at-start hand services on the known G1 PC."""
import argparse
import inspect
import getpass
import json
import os
from pathlib import Path
import shlex
import sys
import time
from wrist_camera import (wrist_camera_settings, wrist_identity_matches, wrist_status_matches,
                          capture_capabilities, wrist_capture_matches,
                          select_wrist_capture_device, wrist_publisher_args,
                          realsense_sdk_devices, resolve_wrist_usb_identity)


def report_preparation_result(path, exit_code, output, error):
    """Keep the remote cause available after the launcher prints its own error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(checked_at=time.time(),exit_code=exit_code,
        stdout=output,stderr=error),ensure_ascii=False,indent=2)+'\n')
    if exit_code:
        lines=[line.strip() for line in (error or output).splitlines() if line.strip()]
        cause=lines[-1] if lines else '远端命令未返回错误详情'
        raise RuntimeError(f'机器人相机/手部服务恢复失败（退出码 {exit_code}）：{cause}；完整输出：{path}')


def hand_binary_needs_reload(expected, deployed, running_digests, body_active):
    """Validate disk deployment; never replace a live body session's hand service."""
    if not expected:
        return False
    if deployed != expected:
        raise RuntimeError('手部服务文件版本校验失败，不启动未知版本')
    outdated = any(value != expected for value in running_digests)
    return outdated and not body_active


def parse_ros_device_info(output):
    """Read identity fields from the existing ROS service's YAML response."""
    info = {}
    for line in output.splitlines():
        key, separator, value = line.strip().partition(':')
        if separator and key in ('serial_number', 'firmware_update_id'):
            info[key] = value.strip().strip('\"\'')
    return info


def release_factory_wrist_capture(device, password):
    """Stop only the factory chest video service holding the selected wrist RGB."""
    import pathlib
    import subprocess
    import time
    binary = b'/unitree/module/video_hub_pc4/videohub_pc4_chest'

    def owns_selected_camera():
        for process in pathlib.Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                args = (process / 'cmdline').read_bytes().split(b'\0')
                if args[:2] == [binary, device.encode()]:
                    return True
            except OSError:
                pass
        return False

    if not owns_selected_camera():
        return
    result = subprocess.run(['sudo', '-S', '-p', '', '/unitree/sbin/mscli',
                             'stopservice', 'video_hub_pc4_chest'],
                            input=password, capture_output=True, text=True, timeout=8)
    deadline = time.monotonic() + 3
    while owns_selected_camera():
        if time.monotonic() >= deadline:
            raise RuntimeError('内置胸部视频服务仍占用腕部相机 ' + device + ': ' + result.stderr.strip())
        time.sleep(.1)
    print('已释放内置胸部视频服务占用的腕部 RGB: ' + device, flush=True)


def stop_stale_wrist_publisher(name, publisher_pids):
    """Stop only the selected wrist publisher, including a blocked capture read."""
    import os
    import signal
    import time
    for sig, timeout in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        for pid in publisher_pids(name):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while publisher_pids(name):
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        if not publisher_pids(name):
            return
    raise RuntimeError(name + ' 旧相机进程无法退出；可能阻塞在 USB 驱动，检查连接；未启动重复实例')


def remote_wrist_program(config):
    """Recover only selected wrist publishers and require fresh frame evidence."""
    return inspect.getsource(release_factory_wrist_capture) + inspect.getsource(stop_stale_wrist_publisher) + ''.join(inspect.getsource(function) for function in (
        realsense_sdk_devices, resolve_wrist_usb_identity,
        wrist_camera_settings, wrist_identity_matches, wrist_status_matches,
        capture_capabilities, wrist_capture_matches,
        select_wrist_capture_device, wrist_publisher_args)) + '''import json, os, pathlib, subprocess, sys, time
config=json.loads(sys.argv[1])
password=sys.stdin.readline()
camera_root=pathlib.Path(config.get('robot_camera_root', '~/sonic_camera')).expanduser()
script=str(camera_root/'scripts/wrist_publish.py')
names=('video6','video8') if config['wrist_only']=='all' else (config['wrist_only'],)
def publisher_pids(name):
    result=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            parts=(p/'cmdline').read_bytes().split(b'\\0')
            if script.encode() in parts and name.encode() in parts: result.append(int(p.name))
        except OSError: pass
    return result
for name in names:
    source=config['camera_sources'][name]
    pids=publisher_pids(name)
    if pids:
        try:
            status=json.loads(pathlib.Path('/tmp/sonic-wrist-'+name+'-status.json').read_text())
        except (OSError,ValueError):
            status={}
        if not wrist_status_matches(status,name,source,time.time(),pids):
            print(name+' 进程仍在但反馈无效或已过期；重启该腕部相机发布进程',flush=True)
            stop_stale_wrist_publisher(name,publisher_pids)
    if not publisher_pids(name):
        device=select_wrist_capture_device(source)
        if device is None:
            raise RuntimeError(name+' RGB USB 不可用: '+source.get('usb_serial',source['usb_path'])+'；检查相机连接')
        release_factory_wrist_capture(device,password)
        env=dict(os.environ,PYTHONPATH=str(camera_root/'deps'))
        with (camera_root/('wrist-'+name+'.log')).open('a') as log:
            subprocess.Popen(wrist_publisher_args(script,name,source,config['robot_host'],device),
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print('已启动 '+name+'，RGB 节点 '+device+'；等待新帧',flush=True)
    deadline=time.monotonic()+8
    while True:
        try:
            status=json.loads(pathlib.Path('/tmp/sonic-wrist-'+name+'-status.json').read_text())
            if wrist_status_matches(status,name,source,time.time(),publisher_pids(name)):
                print(json.dumps(status,ensure_ascii=False),flush=True)
                break
        except (OSError,ValueError): pass
        if time.monotonic()>=deadline:
            raise RuntimeError(name+' 未恢复新鲜帧；检查 '+str(camera_root/('wrist-'+name+'.log')))
        time.sleep(.2)
'''


def remote_program(config):
    # This program starts no head servo or body motion service.
    return (inspect.getsource(hand_binary_needs_reload) +
            inspect.getsource(parse_ros_device_info) +
            inspect.getsource(release_factory_wrist_capture) +
            ''.join(inspect.getsource(function) for function in (realsense_sdk_devices,
                resolve_wrist_usb_identity, wrist_camera_settings,
                wrist_identity_matches,
                capture_capabilities, wrist_capture_matches, select_wrist_capture_device,
                wrist_publisher_args))) + '''import json, os, pathlib, signal, subprocess, sys, time
config=json.loads(sys.argv[1])
password=sys.stdin.readline()
camera_root=pathlib.Path(config.get('robot_camera_root', '~/sonic_camera')).expanduser()
camera_errors={}
def camera_problem(name,reason):
    camera_errors[name]=reason
    print('相机提示 '+name+': '+reason+'；可启动 SONIC，完整采集须恢复三路画面。',flush=True)
    if config.get('require_cameras',False): raise RuntimeError(reason)
def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout
def running(fragment):
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            parts=(p/'cmdline').read_bytes().split(b'\\0')
            if any(part.decode(errors='replace')==fragment for part in parts): return True
        except OSError: pass
    return False
def launch(args,log,cwd=None,env=None):
    with open(log,'a') as f:
        subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
interfaces=json.loads(run(['ip','-j','address']))
interface=next(row['ifname'] for row in interfaces
    if any(a.get('local')==config['robot_host'] for a in row.get('addr_info',[])))
hand=str(pathlib.Path(config['hand_server_path']).expanduser())
# The current body session keeps its already-running hand binary. On a clean
# start compare /proc/PID/exe, so replacing the disk file cannot look loaded.
expected=config.get('hand_binary_sha256')
version_outdated=False
if expected:
    import hashlib
    digest=lambda path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    deployed=digest(hand)
    active=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            parts=(p/'cmdline').read_bytes().split(b'\\0')
            if hand.encode() in parts: active.append(digest(p/'exe'))
        except FileNotFoundError: pass
    version_outdated=any(value!=expected for value in active)
    if hand_binary_needs_reload(expected,deployed,active,config.get('body_control_active',False)):
        config['restart_hands']=True
        print('检测到旧版手部进程；身体未启控，加载新版独立重连服务。',flush=True)
if not pathlib.Path(hand).exists(): raise RuntimeError('安全手部服务未安装')
for binding in config['hand_bindings'].values():
    if not pathlib.Path(binding['port']).exists(): raise RuntimeError('手部 USB 缺失: '+binding['port'])
if config.get('restart_hands',False):
    # Only the exact verified hand binary, never head/body/camera services.
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            parts=(p/'cmdline').read_bytes().split(b'\\0')
            if hand.encode() not in parts: continue
            os.kill(int(p.name),signal.SIGINT)
        except ProcessLookupError: pass
        except FileNotFoundError: pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and running(hand): time.sleep(.05)
    if running(hand): raise RuntimeError('旧手部服务未正常退出，不启动第二个实例')
    print('已退出旧手部服务；重新发现双手，不发送默认开合指令。',flush=True)
if not running(hand):
    env=dict(os.environ)
    for side in ('left','right'):
        serial=config['hand_bindings'].get(side,{}).get('serial')
        if isinstance(serial,str) and serial:
            env['ZETA0_BRAINCO_'+side.upper()+'_SERIAL']=serial
    launch([hand,'-n',interface],str(pathlib.Path(hand).parent/'service.log'),
        cwd=str(pathlib.Path(hand).parent),env=env)
head_serial=str(config.get('head_camera_serial','REPLACE_HEAD_SENSOR_SERIAL'))
head_usb_serial=str(config.get('head_camera_usb_serial',head_serial))
head_present=False
head_identity=dict(expected_sdk_serial=head_serial,expected_usb_serial=head_usb_serial,
    observed_sdk_serial=None,observed_usb_serial=None,usb_path=None,
    source='unverified',sdk_serial_verified=False,ros_device_info_status='not_queried')
# D455 SDK serial and USB descriptor serial can differ. Only accept the
# configured USB identity, including when there are no video4linux nodes.
# Avoid product descriptor reads on an unhealthy hub.
for device in pathlib.Path('/sys/bus/usb/devices').glob('*'):
    try:
        if ((device/'idVendor').read_text().strip().lower()=='8086' and
            (device/'idProduct').read_text().strip().lower()=='0b5c' and
            (device/'serial').read_text().strip()==head_usb_serial):
            head_present=True
            head_identity.update(observed_usb_serial=head_usb_serial,usb_path=device.name,
                source='configured_usb_mapping' if 'head_camera_usb_serial' in config else 'matching_usb_serial')
            break
    except OSError:
        pass
head_active=False
head_problem='头部 RealSense D455 当前未识别（SDK serial='+head_serial+'，USB serial='+head_usb_serial+'）'
if head_present:
    head_active=not subprocess.run(['systemctl','is-active','--quiet','sonic-g1-camera.service']).returncode
    if head_active:
        # Query the running driver, not the bridge's constant payload serial.
        # This master is dedicated to sonic-g1-camera; do not use factory ROS.
        head_identity['ros_device_info_status']='unavailable'
        try:
            result=subprocess.run(['/bin/bash','-c',
                'source /opt/ros/noetic/setup.bash && '
                'export ROS_MASTER_URI=http://127.0.0.1:11319 ROS_HOSTNAME=127.0.0.1 && '
                'exec rosservice call /sonic_camera/realsense2_camera/device_info'],
                capture_output=True,text=True,timeout=3)
            info=parse_ros_device_info(result.stdout) if not result.returncode else {}
            observed=info.get('serial_number')
            if observed:
                head_identity.update(observed_sdk_serial=observed,source='ros_device_info',
                    observed_firmware_update_id=info.get('firmware_update_id'),
                    sdk_serial_verified=observed==head_serial,
                    ros_device_info_status='verified' if observed==head_serial else 'mismatch')
                if observed!=head_serial:
                    head_present=False
                    head_problem='头部相机运行身份不匹配（SDK serial='+observed+'，预期='+head_serial+'）'
        except (OSError,subprocess.TimeoutExpired):
            pass
if not head_present:
    camera_problem('ego_view',head_problem)
elif not head_active:
    # Password arrives on stdin, never embedded in process argv.
    result=subprocess.run(['sudo','-S','-p','','systemctl','start','sonic-g1-camera.service'],input=password,text=True)
    if result.returncode: camera_problem('ego_view','头部相机服务启动失败')
for name in ('video6','video8'):
    source=config['camera_sources'][name]
    script=str(camera_root/'scripts/wrist_publish.py')
    # Identify a matching instance by both the script and logical camera key.
    exists=False
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            parts=(p/'cmdline').read_bytes().split(b'\\0')
            if script.encode() in parts and name.encode() in parts: exists=True
        except OSError: pass
    if not exists:
        # D415 depth and infrared also support capture. Require its verified
        # RGB interface, USB identity and capture index, not the lowest number.
        try:
            device=select_wrist_capture_device(source)
        except RuntimeError as error:
            camera_problem(name,str(error))
            continue
        if device is None:
            camera_problem(name,'腕部 RGB USB 不可用: '+source.get('usb_serial',source['usb_path']))
            continue
        try:
            release_factory_wrist_capture(device,password)
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            camera_problem(name,str(error))
            continue
        env=dict(os.environ,PYTHONPATH=str(camera_root/'deps'))
        try:
            launch(wrist_publisher_args(script,name,source,config['robot_host'],device),
                str(camera_root/('wrist-'+name+'.log')),env=env)
        except OSError as error:
            camera_problem(name,'腕部发布器启动失败: '+str(error))
time.sleep(2)
if not running(hand): raise RuntimeError('手部服务没有保持运行，检查 service.log')
print(json.dumps(dict(interface=interface,hand_server_running=True,
    hand_service_restarted=config.get('restart_hands',False),
    hand_feedback_verified=False,hand_update_pending=version_outdated and config.get('body_control_active',False),
    head_servo_commands=False,head_camera_identity=head_identity,camera_errors=camera_errors)))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--require-cameras',action='store_true',help='可选：相机缺失时要求停止准备')
    parser.add_argument('--restart-hands',action='store_true',help='身体控制停止时恢复手部服务；双手反馈须另行预检')
    parser.add_argument('--wrist-only',choices=('video6','video8','all'),
        help='仅恢复指定腕部相机；按 USB 身份选择 RGB 节点并验证新帧')
    parser.add_argument('--output',type=Path,help='保存远端服务准备输出，默认 logs/robot-services-preparation.json')
    args=parser.parse_args()
    if args.wrist_only and args.restart_hands:
        parser.error('--wrist-only 不能与 --restart-hands 同用')
    if args.restart_hands:
        from start_sonic import processes
        if processes('g1_deploy_onnx_ref'):
            raise RuntimeError('身体控制仍在运行，不能重启手部服务')
    config=json.loads(args.config.read_text())
    config['require_cameras']=args.require_cameras
    config['restart_hands']=args.restart_hands
    config['wrist_only']=args.wrist_only
    if not args.wrist_only:
        hand_path = config.get('hand_server_path')
        if not isinstance(hand_path, str) or not hand_path.startswith(('/', '~/')) or 'REPLACE_' in hand_path:
            parser.error('Set hand_server_path to the installed compatible DDS service, or use --skip-remote on the launcher with each robot service already running')
        hands=json.loads((args.config.parent/'hand-hardware.json').read_text())
        from start_sonic import processes
        config['body_control_active']=bool(processes('g1_deploy_onnx_ref'))
        config['hand_binary_sha256']=hands.get('service_binary_sha256')
        config['hand_bindings']=hands['serial_bindings']
    import paramiko
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    known=Path('/tmp/sonic-g1-known-hosts')
    if known.exists():client.load_host_keys(str(known))
    # Persist the host key locally on first use, without storing a password.
    project_known=args.config.parent/'robot-known-hosts'
    if project_known.exists():client.load_host_keys(str(project_known))
    class SaveHostKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,ssh,hostname,key):
            print('首次记录内置电脑主机指纹:',key.get_fingerprint().hex(),flush=True)
            ssh.get_host_keys().add(hostname,key.get_name(),key)
            ssh.save_host_keys(str(project_known))
    client.set_missing_host_key_policy(SaveHostKey())
    password=os.environ.get('G1_SSH_PASSWORD')
    if password is None:
        password=getpass.getpass('G1 内置电脑 unitree 的 SSH 密码（不保存）: ')
    try:
        client.connect(config['robot_host'],username='unitree',password=password,timeout=8,
            look_for_keys=True,allow_agent=True)
        program=remote_wrist_program(config) if args.wrist_only else remote_program(config)
        command='python3 -c '+shlex.quote(program)+' '+shlex.quote(json.dumps(config))
        stdin,stdout,stderr=client.exec_command(command,timeout=20)
        stdin.write(password+'\n');stdin.flush();stdin.channel.shutdown_write()
        output=stdout.read().decode(errors='replace')
        error=stderr.read().decode(errors='replace')
        print(output,end='',flush=True)
        if error:print(error,file=sys.stderr,end='',flush=True)
        report_name='wrist-camera-preparation.json' if args.wrist_only else 'robot-services-preparation.json'
        report_preparation_result(args.output or args.config.parent.parent/'logs'/report_name,
            stdout.channel.recv_exit_status(),output,error)
    finally:client.close()


if __name__=='__main__':main()
