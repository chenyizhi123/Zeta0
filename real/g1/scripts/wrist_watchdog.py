#!/usr/bin/env python3
"""Recover only identity-bound wrist publishers; never control robot motion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from wrist_camera import select_wrist_capture_device, wrist_publisher_args, wrist_status_matches


def publisher_pids(script, name):
    result=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(b'\0')
            if os.fsencode(script) in args and b'--source-key' in args:
                i=args.index(b'--source-key')
                if args[i+1]==name.encode():result.append(int(path.parent.name))
        except (OSError,IndexError):pass
    return result


def stop_publisher(script, name):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid in publisher_pids(script,name):
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        deadline=time.monotonic()+2
        while publisher_pids(script,name) and time.monotonic()<deadline:time.sleep(.05)
        if not publisher_pids(script,name):return
    raise RuntimeError(name+' publisher is blocked; refusing to start a duplicate')


class WristWatchdog:
    def __init__(self,config,root):
        self.config=config;self.root=Path(root)
        self.script=self.root/'scripts/wrist_publish.py'
        self.unhealthy_since={};self.last_attempt={};self.children={};self.restarts={}

    def check(self,name,now,monotonic):
        child=self.children.get(name)
        if child is not None:child.poll()  # Reap failed children.
        source=self.config['camera_sources'][name]
        pids=publisher_pids(self.script,name)
        try:status=json.loads(Path('/tmp/sonic-wrist-'+name+'-status.json').read_text())
        except (OSError,ValueError):status={}
        if wrist_status_matches(status,name,source,now,pids):
            self.unhealthy_since.pop(name,None)
            return dict(ready=True,pid=status['pid'],device=status['device'],
                        serial=source['usb_serial'],frames=status['frames'])
        since=self.unhealthy_since.setdefault(name,monotonic)
        if pids and monotonic-since<4:
            return dict(ready=False,reason='waiting for fresh publisher frames',pids=pids)
        if monotonic-self.last_attempt.get(name,-float('inf'))<10:
            return dict(ready=False,reason='recovery backoff')
        self.last_attempt[name]=monotonic
        if pids:stop_publisher(self.script,name)
        device=select_wrist_capture_device(source)
        if device is None:
            return dict(ready=False,reason='identity-bound RGB device unavailable',serial=source['usb_serial'])
        env=dict(os.environ,PYTHONPATH=str(self.root/'deps'))
        with (self.root/('wrist-'+name+'.log')).open('a') as log:
            self.children[name]=subprocess.Popen(wrist_publisher_args(str(self.script),name,source,
                self.config['robot_host'],device),env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        self.restarts[name]=self.restarts.get(name,0)+1
        self.unhealthy_since[name]=monotonic
        print(json.dumps(dict(event='wrist_publisher_started',camera=name,device=device,
            serial=source['usb_serial'],pid=self.children[name].pid)),flush=True)
        return dict(ready=False,reason='publisher starting',device=device,pid=self.children[name].pid)

    def poll(self):
        result={}
        for name in ('video6','video8'):
            try:result[name]=self.check(name,time.time(),time.monotonic())
            except Exception as error:result[name]=dict(ready=False,reason=str(error))
        return dict(updated_at=time.time(),cameras=result,restarts=self.restarts,robot_commands_sent=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,help='Default: directory containing the watchdog config')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    for name in ('video6','video8'):
        if not config['camera_sources'][name].get('usb_serial'):raise ValueError('Explicit wrist serial required')
    with Path('/tmp/sonic-wrist-watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=threading.Event()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
        watchdog=WristWatchdog(config,args.root or args.config.resolve().parent);last=None
        destination=Path('/tmp/sonic-wrist-watchdog-status.json')
        while not stop.is_set():
            report=watchdog.poll()
            temp=destination.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(destination)
            state={name:(value['ready'],value.get('reason')) for name,value in report['cameras'].items()}
            if state!=last:print(json.dumps(report),flush=True);last=state
            stop.wait(2)


if __name__=='__main__':main()
