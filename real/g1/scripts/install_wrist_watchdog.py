#!/usr/bin/env python3
"""Install the camera-only robot recovery service over authenticated SSH."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/local.json')
    parser.add_argument('--service-unit', type=Path,
                        default=ROOT/'config/systemd/sonic-g1-wrist-recovery.service',
                        help='Local systemd service matching robot_camera_root')
    args=parser.parse_args()
    local=json.loads(args.config.read_text())
    config=dict(robot_host=local['robot_host'],camera_sources={k:local['camera_sources'][k] for k in ('video6','video8')})
    files={'scripts/'+name:(ROOT/'scripts'/name).read_text() for name in
           ('wrist_camera.py','wrist_publish.py','wrist_watchdog.py')}
    files['wrist-config.json']=json.dumps(config,indent=2)+'\n'
    files['systemd/sonic-g1-wrist-recovery.service']=args.service_unit.read_text()
    root_program='''import pathlib,json,sys,time,shutil,subprocess,os
root=pathlib.Path(sys.argv[2])
backup=root/'backups'/('wrist-recovery-'+time.strftime('%Y%m%d-%H%M%S'))
files=json.loads(pathlib.Path(sys.argv[1]).read_text())
for name,content in files.items():
    path=root/name
    path.resolve().relative_to(root.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
    temporary=path.with_suffix(path.suffix+'.installing');temporary.write_text(content);os.replace(temporary,path)
unit=pathlib.Path('/etc/systemd/system/sonic-g1-wrist-recovery.service')
if unit.exists():
    backup.mkdir(parents=True,exist_ok=True);shutil.copy2(unit,backup/unit.name)
shutil.copyfile(root/'systemd/sonic-g1-wrist-recovery.service',unit)
os.chmod(unit,0o644)
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','enable','sonic-g1-wrist-recovery.service'],check=True)
subprocess.run(['systemctl','restart','sonic-g1-wrist-recovery.service'],check=True)
print('Remote backup: '+str(backup))
'''
    program='''import pathlib,json,sys,time,subprocess,getpass,tempfile
password=getpass.getpass('安装相机恢复服务的 sudo 密码（不保存）: ')+chr(10)
root=pathlib.Path(sys.argv[3]).expanduser()
with tempfile.TemporaryDirectory(prefix='sonic-wrist-install-') as folder:
    path=pathlib.Path(folder)
    (path/'payload.json').write_text(sys.argv[1])
    (path/'install.py').write_text(sys.argv[2])
    result=subprocess.run(['sudo','-S','-p','','python3',str(path/'install.py'),str(path/'payload.json'),str(root)],input=password,text=True,capture_output=True,timeout=30)
print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
if result.returncode:raise SystemExit(result.returncode)
time.sleep(5)
subprocess.run(['systemctl','show','sonic-g1-wrist-recovery.service','--no-pager','-p','ActiveState','-p','SubState','-p','UnitFileState'],check=True)
status=pathlib.Path('/tmp/sonic-wrist-watchdog-status.json')
print(status.read_text() if status.exists() else 'Waiting for first watchdog status')
'''
    command=shlex.join(['python3', '-c', program, json.dumps(files), root_program,
                        local.get('robot_camera_root', '~/sonic_camera')])
    ssh=['ssh','-tt','-o','ConnectTimeout=8','-o','NumberOfPasswordPrompts=1',
         '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'config/robot-known-hosts'),
         'unitree@'+local['robot_host'],command]
    raise SystemExit(subprocess.call(ssh))


if __name__=='__main__':main()
