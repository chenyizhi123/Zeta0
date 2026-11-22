"""Read fresh CRC-validated temperatures and faults, with no DDS writers."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=3)
    from robot_network import LOCAL_CONFIG, local_dds_config
    parser.add_argument('--config', type=Path, default=LOCAL_CONFIG)
    args=parser.parse_args()
    if args.duration <= 0:parser.error('--duration must be positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(checked_at=time.time(),ready=False,
        phase='waiting_for_dds',robot_commands_sent=False),indent=2))
    from cyclonedds.domain import Domain,DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.core import Qos,Policy
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC
    cfg = local_dds_config(args.config, peer_key='body_dds_peer')
    domain=Domain(0,cfg);participant=DomainParticipant(0)
    reader=DataReader(participant,Topic(participant,'rt/lowstate',LowState_),
        qos=Qos(Policy.Reliability.BestEffort,Policy.History.KeepLast(1)))
    crc=CRC();start=time.monotonic();latest=None
    received=0;valid=0;crc_failed=0;invalid_samples=0
    while time.monotonic()-start<args.duration:
        for sample in reader.take(100):
            received+=1
            if not hasattr(sample,'motor_state'):
                invalid_samples+=1
                continue
            if crc.Crc(sample)!=sample.crc:
                crc_failed+=1
                continue
            valid+=1
            motors=sample.motor_state[:29]
            latest=dict(checked_at=time.time(),temperatures=[list(m.temperature) for m in motors],
                errors=[m.motorstate for m in motors],q=[m.q for m in motors],
                max_temperature=max(max(m.temperature) for m in motors),crc_verified=True)
        time.sleep(.01)
    fresh=latest is not None and time.time()-latest['checked_at']<=.5
    ready=fresh and not any(latest['errors'])
    report=dict(latest or {})
    report.update(checked_at=time.time(),ready=ready,
        phase='ready' if ready else 'body_fault' if fresh else 'no_fresh_state',
        received_samples=received,crc_valid_samples=valid,crc_failed_samples=crc_failed,
        invalid_samples=invalid_samples,duration_s=args.duration,robot_commands_sent=False,
        last_valid_received_at=latest['checked_at'] if latest else None)
    args.output.write_text(json.dumps(report,indent=2))
    print(f'身体 DDS 检查：收到 {received} 条，CRC 有效 {valid} 条，CRC 失败 {crc_failed} 条；报告：{args.output}',flush=True)
    if not fresh:
        raise RuntimeError('身体 DDS 没有新鲜有效状态；'+
            ('未收到 LowState 消息' if received==0 else
             '收到消息但没有 CRC 有效状态' if valid==0 else '有效状态已过期'))
    if any(latest['errors']):
        faults=[]
        for i,code in enumerate(latest['errors']):
            if code:
                name='腰部俯仰 WaistPitch' if i==14 else '电机'
                temps=latest['temperatures'][i]
                faults.append(f'{name}[{i}] 故障码 {code:#x}，温度读数 {temps}°C')
        raise RuntimeError('电机报告故障：'+'；'.join(faults)+f'；拒绝启动。完整状态：{args.output}')
    print('身体状态正常，最高温度 %d°C'%latest['max_temperature'])


if __name__=='__main__':main()
