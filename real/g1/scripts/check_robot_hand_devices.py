"""Inspect remote hand USB identities, without opening serial ports."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='''from pathlib import Path
import json,os,time,subprocess
def command(args):
    try:
        result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=5)
        return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except (OSError,subprocess.TimeoutExpired) as error:return dict(error=str(error))
def read(path):
    try:return path.read_text().strip()
    except OSError:return None
hands={}
for side in ('left','right'):
    path=Path('/dev/brainco')/side
    hands[side]=dict(path=str(path),exists=path.exists(),symlink=path.is_symlink(),
        target=str(path.resolve()),character_device=path.is_char_device(),
        accessible=os.access(path,os.R_OK|os.W_OK))
devices={}
for folder in ('/dev/serial/by-id','/dev/serial/by-path'):
    path=Path(folder)
    devices[folder]=[{"name":p.name,"target":str(p.resolve())} for p in path.iterdir()] if path.exists() else []
services=[]
for process in Path('/proc').glob('[0-9]*'):
    try:args=(process/'cmdline').read_bytes().split(b'\\0')
    except OSError:continue
    if any(Path(os.fsdecode(a)).name=='brainco_hand_service' for a in args if a):services.append(int(process.name))
nodes=[p for pattern in ('ttyUSB*','ttyACM*','ttyCH343USB*') for p in Path('/dev').glob(pattern)]
serial_info={}
for path in nodes:
    result=command(['udevadm','info','--query=property','--name='+str(path)])
    properties={}
    for line in result.get('stdout','').splitlines():
        key,separator,value=line.partition('=')
        if separator:properties[key]=value
    driver=Path('/sys/class/tty')/path.name/'device/driver'
    serial_info[str(path)]=dict(properties=properties,driver=str(driver.resolve()) if driver.exists() else None,
        owner_processes=command(['fuser',str(path)]))
usb=[]
for device in Path('/sys/bus/usb/devices').iterdir():
    if not (device/'idVendor').exists():continue
    # Cached descriptor numbers in uevent do not read USB string descriptors.
    # product/serial attributes can block in product_show on a stuck hub.
    values=dict(line.split('=',1) for line in (read(device/'uevent') or '').splitlines() if '=' in line)
    product=values.get('PRODUCT','').split('/')
    if len(product)>=2:
        usb.append(dict(sysfs_name=device.name,idVendor=format(int(product[0],16),'04x'),
            idProduct=format(int(product[1],16),'04x'),devpath=values.get('DEVNAME'),driver=values.get('DRIVER')))
rules={str(p):read(p) for p in Path('/etc/udev/rules.d').glob('*brainco*.rules')}
kernel=command(['dmesg','--color=never'])
if kernel.get('returncode')==0:
    words=('ttyacm','ch343','55ec','usb disconnect','device descriptor','unable to enumerate')
    kernel['stdout']='\\n'.join(line for line in kernel['stdout'].splitlines() if any(word in line.lower() for word in words))[-18000:]
print(json.dumps(dict(checked_at=time.time(),hands=hands,serial_devices=devices,
    serial_nodes=[str(p) for p in nodes],serial_info=serial_info,usb_devices=usb,udev_rules=rules,kernel_usb_log=kernel,
    expected_hand_adapter_present=any(x['idVendor']=='1a86' and x['idProduct']=='55ec' for x in usb),
    hand_service_pids=services,serial_ports_opened=False,robot_commands_sent=False)))
'''


def main():
    host=json.loads((ROOT/'config/local.json').read_text())['robot_host']
    output=ROOT/'logs/hand-usb-current-check.json'
    control=Path.home()/'.cache/zeta0-ssh'
    control.mkdir(mode=0o700,parents=True,exist_ok=True)
    control.chmod(0o700)
    print('只读检查左右手 USB 节点及手部服务；不会打开串口或发送动作指令。',flush=True)
    result=subprocess.run(['ssh','-o','ConnectTimeout=8','-o','NumberOfPasswordPrompts=2',
        '-o','ControlMaster=auto','-o','ControlPersist=10m','-o','ControlPath='+str(control/'%C'),
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'config/robot-known-hosts'),
        'unitree@'+host,'python3 -c '+shlex.quote(REMOTE)],stdout=subprocess.PIPE,text=True)
    if result.returncode:
        # A cancelled older terminal must not replace a newer successful report.
        failure=output.with_name('hand-usb-check-failure-'+str(os.getpid())+'.json')
        failure.write_text(json.dumps(dict(checked_at_client=time.time(),ready=False,ssh_exit_code=result.returncode))+'\n')
        print('SSH 登录未成功，设备情况尚未确认。',flush=True)
        return result.returncode
    report=json.loads(result.stdout)
    report['checked_at_client']=time.time()
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for side,label in (('left','左手'),('right','右手')):
        status=report['hands'][side]
        print(label+'：'+('USB 节点存在 → '+status['target'] if status['exists'] else 'USB 节点不存在或符号链接失效'),flush=True)
    print('手部服务 PID：'+str(report['hand_service_pids']),flush=True)
    print('预期双手 USB 转换器：'+('已枚举' if report['expected_hand_adapter_present'] else '未枚举'),flush=True)
    for device,info in report['serial_info'].items():
        props=info['properties']
        print(device+'：'+str(props.get('ID_VENDOR_ID'))+':'+str(props.get('ID_MODEL_ID'))+' '+str(props.get('ID_MODEL'))+' '+str(props.get('ID_SERIAL_SHORT','')),flush=True)
    print('报告：'+str(output),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
