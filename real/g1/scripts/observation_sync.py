"""Shared receive-time observation contract for recording and visual inference.

No robot commands, no clock-domain conversion, no claimed exposure synchronization.
Snapshots are causal at decision time; repeat tracking advances only on commit.
"""
from collections import deque
import copy
import json
import math
import time
import uuid
from pathlib import Path

from brainco_gripper_data import MissingGripperData, packet_features
from collection_metrics import EpisodeMetrics, SourceCounter

CAMERAS = ('ego_view', 'left_wrist', 'right_wrist')
DEFAULTS = dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20.,
                max_age_ms=200., body_max_offset_ms=40., hand_max_offset_ms=50.,
                history_size=128, image_shape=[480,640,3])
BOOT_FILE = Path('/proc/sys/kernel/random/boot_id')
WORKSTATION_BOOT_ID = BOOT_FILE.read_text().strip() if BOOT_FILE.exists() else 'unknown'


def camera_keys(config=None):
    mode = (config or {}).get('recording_cameras', 'all')
    if mode not in ('all', 'head'):
        raise ValueError('相机模式必须为 all 或 head')
    return ('ego_view',) if mode == 'head' else CAMERAS


def contract_cameras(sync_contract):
    keys = tuple(sync_contract.get('cameras', CAMERAS))
    if keys not in (CAMERAS, ('ego_view',)):
        raise ValueError('Unsupported observation camera set')
    return keys


def contract(config):
    result = {**DEFAULTS, **config.get('observation_sync', {})}
    keys = camera_keys(config)
    if 'cameras' in result and tuple(result['cameras']) != keys:
        raise ValueError('观察接口相机集合与采集模式不一致')
    if keys != CAMERAS:
        result['cameras'] = list(keys)
    if result['version'] != 1:
        raise ValueError('Unsupported observation contract version')
    for key in ('camera_max_skew_ms','camera_wait_ms','max_age_ms',
                'body_max_offset_ms','hand_max_offset_ms'):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError('Invalid observation timing: '+key)
    if result['history_size'] < 4:
        raise ValueError('Observation history is too small')
    return result


class CameraPairer:
    """Head anchor, nearest wrist reception, bounded wait; preserve identities."""
    def __init__(self, config):
        self.contract = contract(config)
        self.camera_keys = camera_keys(config)
        self.mapping = config.get('wrist_mapping', {})
        self.source_config = copy.deepcopy(config.get('camera_sources', {}))
        self.buffers = {k:deque(maxlen=self.contract['history_size'])
                        for k in (('ego_view',) if len(self.camera_keys) == 1 else ('ego_view','video6','video8'))}
        self.pending = deque(maxlen=self.contract['history_size'])
        self.session = uuid.uuid4().hex
        self.counts = {k:0 for k in self.buffers}
        self.sequence = self.rejected = 0
        self.source_metrics = {k:SourceCounter() for k in self.buffers}
        self.rejected_missing = self.rejected_skew = self.pending_evictions = 0
        self.publisher_drops = 0

    def push(self, name, key, payload, mono, wall):
        if key not in payload.get('images', {}):
            self.source_metrics[name].invalid_messages += 1
            return
        expected = self.source_config.get(name, {})
        if name != 'ego_view' and ('usb_path' in expected or 'usb_serial' in expected):
            identity = payload.get('camera')
            field, setting = ('serial', 'usb_serial') if 'usb_serial' in expected else ('usb_path', 'usb_path')
            if not isinstance(identity, dict) or identity.get(field) != expected[setting]:
                self.source_metrics[name].invalid_messages += 1
                return
            for field, setting in (('idVendor', 'usb_vendor_id'), ('idProduct', 'usb_product_id'),
                                   ('usb_interface', 'usb_interface')):
                if setting in expected and str(identity.get(field, '')).lower() != expected[setting].lower():
                    self.source_metrics[name].invalid_messages += 1
                    return
        self.counts[name] += 1
        metric = self.source_metrics[name]
        sequence = payload.get('sequence')
        source_id = sequence if sequence is not None else payload.get('timestamps',{}).get(key)
        identity = None if source_id is None else (payload.get('source_session'),source_id)
        metric.observe(mono,identity,sequence,payload.get('source_session'))
        metric.history_evictions += int(len(self.buffers[name]) == self.buffers[name].maxlen)
        sample = dict(arrived=mono, received_at=wall, payload=payload, key=key,
                      receive_sequence=self.counts[name])
        self.buffers[name].append(sample)
        if name == 'ego_view':
            self.pending_evictions += int(len(self.pending) == self.pending.maxlen)
            self.pending.append(sample)

    def ready(self, now):
        output = []
        wait_s = 0. if len(self.camera_keys) == 1 else self.contract['camera_wait_ms']/1000
        while self.pending and now-self.pending[0]['arrived'] >= wait_s:
            head = self.pending.popleft()
            chosen = {'ego_view':head}
            for name in self.buffers:
                if name == 'ego_view':
                    continue
                eligible = [s for s in self.buffers[name] if s['arrived'] <= now]
                if eligible:
                    chosen[name] = min(eligible,key=lambda s:abs(s['arrived']-head['arrived']))
            stamps = [s['arrived'] for s in chosen.values()]
            skew = max(stamps)-min(stamps)
            if len(chosen) != len(self.camera_keys) or skew > self.contract['camera_max_skew_ms']/1000:
                self.rejected += 1
                if len(chosen) != len(self.camera_keys):
                    self.rejected_missing += 1
                else:
                    self.rejected_skew += 1
                continue
            images, sources, timestamps = {}, {}, {}
            for name, sample in chosen.items():
                label = name if name == 'ego_view' else self.mapping[name]
                original = sample['payload']
                source_id = original.get('sequence')
                id_kind = 'source_sequence'
                if source_id is None:
                    source_id = original.get('timestamps',{}).get(sample['key'])
                    id_kind = 'source_timestamp'
                if source_id is None:
                    source_id = sample['receive_sequence']
                    id_kind = 'composer_receive_sequence'
                frame_id = f"{self.session}:{name}:{original.get('source_session','unknown')}:{source_id}"
                images[label] = original['images'][sample['key']]
                timestamps[label] = sample['received_at']
                sources[label] = dict(source_key=sample['key'],
                    frame_id=frame_id,frame_id_kind=id_kind,
                    sequence=original.get('sequence'), source_session=original.get('source_session'),
                    original_timestamp=original.get('timestamps',{}).get(sample['key']),
                    original_clock=original.get('timestamp_kind','unspecified_remote_clock'),
                    capture=original.get('capture'), camera=original.get('camera'),
                    received_at=sample['received_at'], received_monotonic=sample['arrived'])
            group_sequence = self.sequence
            self.sequence += 1
            output.append(dict(images=images,timestamps=timestamps,sources=sources,
                sequence=group_sequence,composer_session=self.session,
                camera_info=head['payload'].get('camera_info',{}),
                sync=dict(method='workstation_receive_time',skew_ms=skew*1000,
                    anchor_monotonic=head['arrived'], emitted_monotonic=now,
                    max_skew_ms=self.contract['camera_max_skew_ms'],
                    hardware_synchronized=False,wrist_mapping_verified=len(self.camera_keys) == 3,
                    contract=self.contract,source_receive_metrics=self.metrics(now))))
        return output

    def metrics(self, now=None):
        return dict(version=1,produced_groups=self.sequence,rejected_groups=self.rejected,
            rejected_missing_source=self.rejected_missing,rejected_skew=self.rejected_skew,
            pending_history_evictions=self.pending_evictions,publisher_send_drops=self.publisher_drops,
            sources={k:v.summary(now) for k,v in self.source_metrics.items()},
            hardware_synchronized=False)


class ObservationAssembler:
    def __init__(self, config):
        self.contract = contract(config)
        self.camera_keys = camera_keys(config)
        self.history = {k:deque(maxlen=self.contract['history_size'])
                        for k in ('camera','body','gripper')}
        self.last_ids = None
        self.last_decision = None
        self.episode_start = None
        self.ingress_metrics = {k:SourceCounter() for k in self.history}
        self.episode_metrics = EpisodeMetrics(self.camera_keys)
        self.sample_segment = 0
        self.last_source_identity = None
        self.last_composer_session = None

    def push(self, kind, payload, mono, wall):
        metric = self.ingress_metrics[kind]
        session = payload.get('composer_session') if kind == 'camera' else payload.get('source_session')
        sequence = payload.get('sequence')
        identity = (session, sequence) if sequence is not None else payload.get(
            'timestamp_monotonic',payload.get('ros_timestamp'))
        metric.observe(mono,identity,sequence,session)
        metric.history_evictions += int(len(self.history[kind]) == self.history[kind].maxlen)
        self.history[kind].append(dict(payload=payload,mono=mono,wall=wall))

    def reset_episode(self):
        """Reset commit state while preserving receive history and source counters."""
        self.last_ids = self.last_decision = self.episode_start = None
        self.last_source_identity = self.last_composer_session = None
        self.sample_segment = 0
        self.episode_metrics = EpisodeMetrics(self.camera_keys)

    def metrics(self, now=None):
        now = time.monotonic() if now is None else now
        latest_camera = self.history['camera'][-1]['payload'] if self.history['camera'] else {}
        return dict(version=1, ingress={k:v.summary(now) for k,v in self.ingress_metrics.items()},
                    history_sizes={k:len(v) for k,v in self.history.items()},
                    camera_composer=latest_camera.get('sync',{}).get('source_receive_metrics'),
                    committed_episode=self.episode_metrics.summary())

    def _eligible(self, kind, decision):
        return [s for s in self.history[kind]
                if 0 <= decision-s['mono'] <= self.contract['max_age_ms']/1000]

    def snapshot(self, decision=None, wall=None):
        decision = time.monotonic() if decision is None else decision
        wall = time.time() if wall is None else wall
        cameras = self._eligible('camera',decision)
        bodies = self._eligible('body',decision)
        hands = self._eligible('gripper',decision)
        if not cameras or not bodies or not hands:
            if not any(self.history.values()):
                raise MissingGripperData('正在等待首批图像、身体和手部遥测；刚建立订阅时短暂等待属于正常现象')
            statuses = []
            camera_label = '顶部图像' if len(self.camera_keys) == 1 else '三路图像合流'
            for kind, label, eligible in (('camera',camera_label,cameras),
                                          ('body','身体状态',bodies),
                                          ('gripper','灵巧手遥测',hands)):
                samples = self.history[kind]
                if not samples:
                    statuses.append(label+'缺失（接收 age=无，尚未收到）')
                    continue
                latest = max(eligible or samples,key=lambda s:s['mono'])
                age_ms = (decision-latest['mono'])*1000
                status = ('接收新鲜' if eligible else
                          '接收时间无效' if not math.isfinite(age_ms) else
                          '接收时间超前' if age_ms < 0 else '过期')
                statuses.append(f'{label}{status}（接收 age={age_ms:.0f}ms）')
            raise MissingGripperData('等待新鲜'+camera_label+'、身体和灵巧手反馈：'+
                '；'.join(statuses)+f'；接收新鲜阈值≤{self.contract["max_age_ms"]:g}ms')
        camera = max(cameras,key=lambda s:s['mono'])
        message = camera['payload']
        try:
            sources = message['sources']
            arrivals = [float(sources[k]['received_monotonic']) for k in self.camera_keys]
            ids = [str(sources[k]['frame_id']) for k in self.camera_keys]
            if set(message['images']) != set(self.camera_keys) or set(sources) != set(self.camera_keys):
                raise ValueError('图像或来源集合不匹配，要求 '+','.join(self.camera_keys))
            ages = [(decision-t)*1000 for t in arrivals]
            age_details = '、'.join(f'{k}={age:.1f}ms' for k,age in zip(self.camera_keys,ages))
            if not all(math.isfinite(t) for t in arrivals):
                raise ValueError('相机接收时间不是有限数值')
            if any(age < 0 for age in ages):
                raise ValueError('相机接收时间超前：'+age_details)
            if any(decision-t > self.contract['max_age_ms']/1000 for t in arrivals):
                raise ValueError('相机原始帧过期：'+age_details+
                    f'；阈值≤{self.contract["max_age_ms"]:g}ms')
            skew = max(arrivals)-min(arrivals)
            if skew > self.contract['camera_max_skew_ms']/1000:
                raise ValueError(f'三路时间差 {skew*1000:.1f}ms 超过配对窗口 '+
                    f'{self.contract["camera_max_skew_ms"]:g}ms')
            if message['sync']['method'] != 'workstation_receive_time':
                raise ValueError('合流时间方法不兼容，需要 workstation_receive_time')
            if message['sync']['hardware_synchronized'] is not False:
                raise ValueError('合流硬件同步声明与接收时间配对规则不兼容')
            if message['sync']['contract'] != self.contract:
                received_contract = message['sync']['contract']
                if not isinstance(received_contract,dict):
                    raise ValueError('合流配对配置不是字典')
                differences = [f'{k}: 合流={received_contract.get(k)!r}, 录制={self.contract.get(k)!r}'
                    for k in sorted(set(received_contract)|set(self.contract))
                    if received_contract.get(k) != self.contract.get(k)]
                raise ValueError('合流与录制配置不一致：'+'；'.join(differences))
        except (KeyError,ValueError,TypeError) as error:
            reason = '缺少字段 '+str(error) if isinstance(error,KeyError) else str(error)
            raise MissingGripperData('相机配对信息无效：'+reason) from error
        anchor = arrivals[0]
        body = min(bodies,key=lambda s:abs(s['mono']-anchor))
        def hand_stamps(sample):
            packet_features(sample['payload'],decision)
            return [sample['payload']['timestamp_monotonic']-a
                    for a in sample['payload']['receive_ages'][:2]]
        valid_hands = []
        for sample in hands:
            try:
                stamps = hand_stamps(sample)
            except MissingGripperData:
                continue
            valid_hands.append((sample,stamps))
        if not valid_hands:
            raise MissingGripperData('等待新鲜左右手实际反馈')
        hand, channel_stamps = min(valid_hands,key=lambda s:max(abs(t-anchor) for t in s[1]))
        if abs(body['mono']-anchor) > self.contract['body_max_offset_ms']/1000:
            raise MissingGripperData('图像与身体状态接收时间差超过配对窗口')
        if max(abs(t-anchor) for t in channel_stamps) > self.contract['hand_max_offset_ms']/1000:
            raise MissingGripperData('图像与手部状态接收时间差超过配对窗口')
        current_body = max(bodies,key=lambda s:s['mono'])
        current_hand = max(hands,key=lambda s:s['mono'])
        # Observation state comes from the image-time neighbour. Actions are the
        # latest actual published commands available at this decision, not history.
        proprio = copy.deepcopy(body['payload'])
        for key in ('last_action','last_left_hand_action','last_right_hand_action','token_state'):
            proprio[key] = copy.deepcopy(current_body['payload'].get(key,[]))
        if 'delta_heading' in current_body['payload']:
            proprio['delta_heading'] = copy.deepcopy(current_body['payload']['delta_heading'])
        else:
            proprio.pop('delta_heading',None)
        packet = copy.deepcopy(hand['payload'])
        packet_features(packet,decision)
        packet_features(current_hand['payload'],decision)
        packet['action'] = copy.deepcopy(current_hand['payload']['action'])
        packet['action_joints'] = copy.deepcopy(current_hand['payload'].get('action_joints'))
        for key in ('action_speed_joints','source_session','action_receive_monotonic',
                    'action_receive_sequence'):
            if key in current_hand['payload']:
                packet[key] = copy.deepcopy(current_hand['payload'][key])
            else:
                packet.pop(key,None)
        # Rebase both samples' channel ages onto the common decision timestamp.
        packet['receive_ages'] = [
            a+decision-hand['payload']['timestamp_monotonic']
            for a in hand['payload']['receive_ages'][:2]] + [
            a+decision-current_hand['payload']['timestamp_monotonic']
            for a in current_hand['payload']['receive_ages'][2:]]
        packet['timestamp_monotonic'] = decision
        repeated = [self.last_ids is not None and i == old
                    for i, old in zip(ids,self.last_ids or [None]*len(self.camera_keys))]
        timing = dict(version=1, decision_monotonic=decision,decision_wall=wall,
            decision_clock='workstation_monotonic',workstation_boot_id=WORKSTATION_BOOT_ID,
            sample_elapsed_s=0. if self.episode_start is None else decision-self.episode_start,
            sample_interval_s=None if self.last_decision is None else decision-self.last_decision,
            anchor_monotonic=anchor, camera_pair_sequence=message.get('sequence'),
            composer_session=message.get('composer_session'),
            cameras={k:dict(sources[k],repeated=repeated[i],
                offset_ms=(arrivals[i]-anchor)*1000,age_ms=(decision-arrivals[i])*1000)
                for i,k in enumerate(self.camera_keys)},
            camera_skew_ms=(max(arrivals)-min(arrivals))*1000,
            body=dict(received_monotonic=body['mono'],offset_ms=(body['mono']-anchor)*1000,
                original_timestamp=body['payload'].get('ros_timestamp'),
                original_clock='robot_wall_unconverted',
                source_session=body['payload'].get('source_session'),sequence=body['payload'].get('sequence')),
            hand_state=dict(received_monotonic=hand['mono'],
                offset_ms=(sum(channel_stamps)/2-anchor)*1000,
                channel_received_monotonic=channel_stamps,
                channel_offset_ms=[(t-anchor)*1000 for t in channel_stamps],
                source_session=hand['payload'].get('source_session'),sequence=hand['payload'].get('sequence')),
            action=dict(body_received_monotonic=current_body['mono'],
                hand_received_monotonic=current_hand['mono'],
                body_source_session=current_body['payload'].get('source_session'),
                body_sequence=current_body['payload'].get('sequence'),
                hand_source_session=current_hand['payload'].get('source_session'),
                hand_receive_monotonic=copy.deepcopy(current_hand['payload'].get('action_receive_monotonic')),
                hand_receive_sequence=copy.deepcopy(current_hand['payload'].get('action_receive_sequence'))),
            gripper_receive_ages=packet['receive_ages'],
            hardware_synchronized=False,contract=self.contract)
        return dict(image_message=message,proprio=proprio,packet=packet,timing=timing)

    def commit(self, snapshot):
        timing = copy.deepcopy(snapshot['timing'])
        decision = timing['decision_monotonic']
        if self.last_decision is not None and decision <= self.last_decision:
            raise ValueError('Observation decisions must increase')
        if self.episode_start is None:
            self.episode_start = decision
        timing['sample_elapsed_s'] = decision-self.episode_start
        timing['sample_interval_s'] = None if self.last_decision is None else decision-self.last_decision
        source_identity = {k:(timing['cameras'][k].get('source_session'),
                              timing['cameras'][k].get('sequence')) for k in self.camera_keys}
        source_identity.update(body=(timing['body'].get('source_session'),timing['body'].get('sequence')),
            gripper=(timing['hand_state'].get('source_session'),timing['hand_state'].get('sequence')),
            body_action=(timing['action'].get('body_source_session'),timing['action'].get('body_sequence')),
            gripper_action=(timing['action'].get('hand_source_session'),timing['action'].get('hand_receive_sequence')))
        reasons = []
        if self.last_source_identity is not None:
            if timing.get('composer_session') != self.last_composer_session:
                reasons.append('composer_session_changed')
            for key,(session,sequence) in source_identity.items():
                previous_session,previous_sequence = self.last_source_identity[key]
                if session is not None and previous_session is not None and session != previous_session:
                    reasons.append(key+':source_session_changed')
                elif session == previous_session:
                    current_values = sequence if isinstance(sequence,(tuple,list)) else [sequence]
                    previous_values = previous_sequence if isinstance(previous_sequence,(tuple,list)) else [previous_sequence]
                    if len(current_values) == len(previous_values) and any(
                            isinstance(current,int) and isinstance(previous,int) and current < previous
                            for current,previous in zip(current_values,previous_values)):
                        reasons.append(key+':source_sequence_regressed')
        if reasons:
            self.sample_segment += 1
        timing['sample_segment'] = self.sample_segment
        timing['segment_reasons'] = reasons
        self.last_decision = decision
        self.last_ids = [timing['cameras'][k]['frame_id'] for k in self.camera_keys]
        self.last_source_identity = source_identity
        self.last_composer_session = timing.get('composer_session')
        self.episode_metrics.observe(timing)
        return timing


def timing_features(config=None):
    cameras = camera_keys(config)
    names = {'ego_view':'ego', 'left_wrist':'left', 'right_wrist':'right'}
    return {
        'observation.sync_offset':dict(dtype='float32',shape=(len(cameras)+2,),
            names=[names[k]+'_receive_offset_s' for k in cameras]+['body_receive_offset_s','hand_receive_offset_s']),
        'observation.camera_repeated':dict(dtype='int64',shape=(len(cameras),),names=list(cameras)),
        'observation.sample_time':dict(dtype='float64',shape=(1,),names=['elapsed_monotonic_s']),
    }


def use_timing_features(existing):
    if existing is None:
        return True
    present = [k in existing['features'] for k in timing_features()]
    if any(present) and not all(present):
        raise ValueError('已有数据集时间配对字段不完整')
    return all(present)


def timing_values(snapshot, start):
    import numpy as np
    t = snapshot['timing']
    cameras = contract_cameras(t['contract'])
    return {
        'observation.sync_offset':np.asarray(
            [t['cameras'][k]['offset_ms']/1000 for k in cameras] +
            [t['body']['offset_ms']/1000,t['hand_state']['offset_ms']/1000],dtype=np.float32),
        'observation.camera_repeated':np.asarray([t['cameras'][k]['repeated'] for k in cameras],dtype=np.int64),
        'observation.sample_time':np.asarray([t['decision_monotonic']-start],dtype=np.float64),
    }


def save_timing_sidecar(destination, index, samples, sync_contract):
    path = destination/'meta'/f'chenyz_timing_{index:06d}.jsonl'
    with path.open('x') as output:
        for frame, sample in enumerate(samples):
            output.write(json.dumps(dict(frame_index=frame,**sample),allow_nan=False)+'\n')
    return dict(timing_file=str(path.relative_to(destination)),timing_frames=len(samples),
                observation_contract=sync_contract,hardware_synchronized=False)
