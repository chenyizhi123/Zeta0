#!/usr/bin/env python3
"""Launch components using the existing SONIC installation."""
import argparse
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='chenyz G1 PICO data collection')
parser.add_argument('component', choices=['check-pico', 'teleop', 'compose-cameras', 'preview', 'export', 'pc-service', 'deploy-real', 'brainco-hands', 'brainco-record', 'observe', 'adapt-model', 'infer-model', 'serve-model', 'deploy-model'])
args = parser.parse_args(sys.argv[1:2])
extra = sys.argv[2:]
config = json.loads((root/'config/local.json').read_text())
sonic = Path(config['sonic_root'])
if args.component == 'teleop':
    import fcntl
    with (root/'logs/online-model-control.lock').open('a+') as control_lock:
        try:
            fcntl.flock(control_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('模型正在在线控制，不能同时启动 PICO manager；先退出模型客户端。')
python_teleop = str(sonic/'.venv_teleop/bin/python')
python_data = str(sonic/'.venv_data_collection/bin/python')
if args.component in ('teleop', 'check-pico'):
    # The SDK talks only to the PC service on loopback. Connect directly.
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'grpc_proxy',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'GRPC_PROXY'):
        os.environ.pop(key, None)
    sdk_libraries = sonic/'external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib'
    os.environ['LD_LIBRARY_PATH'] = str(sdk_libraries)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
commands = {
    'deploy-model':[str(sonic/'.venv_sim/bin/python'),str(root/'scripts/deploy_online_model.py')],
    'serve-model':[str(root/'.venv_policy/bin/python'),str(root/'scripts/serve_local_policy.py')],
    'infer-model':[str(root/'.venv_policy/bin/python'),str(root/'scripts/local_policy_runtime.py')],
    'adapt-model':[python_data,str(root/'scripts/local_vla_adapter.py')],
    'observe':[python_data,str(root/'scripts/observe.py')],
    'check-pico':[python_teleop, str(root/'scripts/check_pico_connection.py')],
    'teleop':[python_teleop, str(root/'scripts/pico_manager.py'), '--port', '5556'],
    'compose-cameras':[str(sonic/'.venv_sim/bin/python'), str(root/'scripts/compose_camera_streams.py')],
    'preview':[str(sonic/'.venv_sim/bin/python'), str(root/'scripts/preview_cameras.py')],
    'export':['bash', str(root/'record_episode.sh')],
    'deploy-real':['bash', str(root/'scripts/deploy_g1.sh')],
    'brainco-hands':[str(sonic/'.venv_sim/bin/python'), str(root/'scripts/brainco_bridge.py')],
    'brainco-record':[str(sonic/'.venv_sim/bin/python'), str(root/'scripts/brainco_record_source.py')],
}
if args.component in ('brainco-hands', 'brainco-record'):
    sdk_python = str(sonic/'external_dependencies/unitree_sdk2_python')
    os.environ['PYTHONPATH'] = sdk_python+(':'+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
if args.component=='pc-service':
    directory = Path(config['pc_service_root'])
    os.environ['LD_LIBRARY_PATH'] = ':'.join([str(directory), str(directory/'lib'), str(directory/'SDK/x64'), os.environ.get('LD_LIBRARY_PATH','')])
    os.environ['QT_PLUGIN_PATH'] = str(directory/'plugins')
    os.environ['QT_QML_PATH'] = str(directory/'qml')
    os.chdir(directory)
    command = [str(directory/'RoboticsServiceProcess')]
else:
    os.chdir(sonic if args.component=='teleop' else root)
    command = commands[args.component]
os.execvp(command[0], command+extra)
