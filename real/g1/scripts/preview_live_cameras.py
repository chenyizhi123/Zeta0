#!/usr/bin/env python3
"""Independent latest-frame camera preview in a normal OpenCV window."""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import time

import cv2
import msgpack
import numpy as np
import zmq


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=root/'config/local.json')
    parser.add_argument('--cameras',choices=('all','head'),default='all')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    selected=[('ego_view','Head')]
    if args.cameras=='all':
        reverse={label:key for key,label in config['wrist_mapping'].items()}
        selected += [(reverse['left_wrist'],'Left Wrist'),(reverse['right_wrist'],'Right Wrist')]
    context=zmq.Context();poller=zmq.Poller();sockets={};latest={};counts={key:0 for key,_ in selected}
    connected_at={};reconnects={key:0 for key,_ in selected}
    def connect(key):
        source=config['camera_sources'][key]
        sock=context.socket(zmq.SUB);sock.setsockopt(zmq.SUBSCRIBE,b'')
        sock.setsockopt(zmq.CONFLATE,1);sock.setsockopt(zmq.LINGER,0)
        sock.setsockopt(zmq.HEARTBEAT_IVL,1000)
        sock.setsockopt(zmq.HEARTBEAT_TIMEOUT,3000)
        sock.connect(f"tcp://{source['host']}:{source['port']}")
        poller.register(sock,zmq.POLLIN);sockets[sock]=key
        connected_at[key]=time.monotonic()
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    title='G1 Cameras - Head | Left Wrist | Right Wrist' if args.cameras=='all' else 'G1 Camera - Head'
    cv2.setNumThreads(1)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,640*len(selected),480)
    cv2.moveWindow(title,40,60)
    status_path=root/'logs/three-camera-live-window.json'
    last_report=0
    try:
        for key,label in selected:
            connect(key)
        while running:
            for sock in dict(poller.poll(5)):
                key=sockets[sock];source=config['camera_sources'][key]
                try:
                    payload=msgpack.unpackb(sock.recv(),raw=False)
                    expected=source.get('usb_serial')
                    if expected and payload.get('camera',{}).get('serial')!=expected:
                        continue
                    data=payload['images'][source['key']]
                    if isinstance(data,str):data=base64.b64decode(data)
                    frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                    if frame is not None:
                        latest[key]=(frame,time.monotonic());counts[key]+=1
                except (ValueError,KeyError,TypeError):
                    continue
            now=time.monotonic();panels=[];ages={}
            for sock,key in list(sockets.items()):
                last_frame=latest[key][1] if key in latest else connected_at[key]
                if now-max(last_frame,connected_at[key])>5:
                    # Replace only this preview subscription. A half-open TCP
                    # connection must not leave the panel frozen indefinitely.
                    poller.unregister(sock);sock.close();del sockets[sock]
                    reconnects[key]+=1
                    print(f'{key}: no decoded frame for 5s; reconnecting preview',flush=True)
                    connect(key)
            for key,label in selected:
                sample=latest.get(key)
                if sample:
                    frame,stamp=sample;panel=cv2.resize(frame,(640,480));age=now-stamp
                    caption=label if age<1 else label+' - STALE'
                else:
                    panel=np.zeros((480,640,3),np.uint8);age=None;caption=label+' - Waiting'
                ages[key]=None if age is None else age*1000
                color=(0,255,0) if age is not None and age<1 else (0,80,255)
                cv2.rectangle(panel,(0,0),(640,36),(0,0,0),-1)
                cv2.putText(panel,caption,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
                panels.append(panel)
            cv2.imshow(title,np.hstack(panels))
            if cv2.waitKey(10)&0xff in (27,ord('q'),ord('Q')):break
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if now-last_report>=1:
                report=dict(updated_at=time.time(),pid=os.getpid(),live=True,
                    recording=False,cameras=args.cameras,window_title=title,received=counts,camera_age_ms=ages,
                    reconnects=reconnects)
                temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(status_path)
                last_report=now
    finally:
        cv2.destroyAllWindows()
        for sock in sockets:sock.close()
        context.term()
        if status_path.exists():
            report=json.loads(status_path.read_text());report.update(live=False,stopped_at=time.time());status_path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
