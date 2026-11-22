"""Require fresh feedback from both Revo2 hands; creates no command writers."""
import argparse
import json
import math
from pathlib import Path
import time


def read_feedback(readers, timeout=6, *, clock=time.monotonic, sleep=time.sleep):
    if set(readers)!={'left','right'}:raise ValueError('Both hand readers are required')
    started=clock();latest={};counts={side:0 for side in readers}
    while clock()-started<timeout:
        for side,reader in readers.items():
            for sample in reader.take(100):
                if not hasattr(sample,'states'):continue
                q=[motor.q for motor in sample.states]
                if len(q)==6 and all(math.isfinite(v) and 0<=v<=1 for v in q):
                    latest[side]=dict(q=q,received=clock());counts[side]+=1
                else:
                    latest.pop(side,None);counts[side]=0
        now=clock()
        if all(counts[side]>=3 and side in latest and 0<=now-latest[side]['received']<=.5 for side in readers):
            return dict(ready=True,counts=counts,q={side:sample['q'] for side,sample in latest.items()},robot_commands_sent=False)
        sleep(.01)
    now=clock()
    missing=[side for side in readers if counts[side]<3 or side not in latest or now-latest[side]['received']>.5]
    return dict(ready=False,missing=missing,counts=counts,robot_commands_sent=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    from robot_network import LOCAL_CONFIG, local_dds_config
    parser.add_argument('--config', type=Path, default=LOCAL_CONFIG)
    args=parser.parse_args()
    from cyclonedds.domain import Domain,DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.core import Qos,Policy
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    cfg = local_dds_config(args.config)
    domain=Domain(0,cfg);participant=DomainParticipant(0)
    readers={side:DataReader(participant,Topic(participant,'rt/brainco/'+side+'/state',MotorStates_),
        qos=Qos(Policy.Reliability.BestEffort,Policy.History.KeepLast(1))) for side in ('left','right')}
    result=read_feedback(readers);result['checked_at']=time.time()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    if not result['ready']:
        names={'left':'左手','right':'右手'}
        raise RuntimeError('、'.join(names[side] for side in result['missing'])+'真实反馈未就绪；检查供电/RS485/手部服务，取消启动')
    print('双手真实反馈正常；检查期间未发送开合指令')


if __name__=='__main__':main()
