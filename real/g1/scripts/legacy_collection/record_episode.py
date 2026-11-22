#!/usr/bin/env python3
"""Record one or several task episodes; contains only telemetry subscriptions."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty

from episode_session import collection_lock, dataset_plan, task_path, verify_saved_episode
from episode_metadata import ensure_empty_episode_indexes, install_empty_metadata_indexes
from grasp_recording import (grasp_features, use_grasp_features, grasp_sample,
                             sample_features, save_grasp_sidecar)
from observation_sync import (contract, timing_features, use_timing_features,
                              timing_values, save_timing_sidecar, camera_keys)
from episode_quality import EpisodeLifecycle, build_episode_quality
from collection_metrics import EpisodeMetrics
from recording_runtime import capture_recording_runtime, runtime_sample_matches
from training_actions import (training_action_features, use_training_action_features,
                              training_action_values)
from recording_voice import RecordingVoice, announce

ROOT = Path(__file__).resolve().parents[2]


def terminal_command(key, recording):
    if key in ('x','q'):
        return 'discard'
    if key in ('eof','\x03'):
        return 'exit'
    if key in (' ','\n','\r'):
        return 'save' if recording else 'start'
    if key == 's':
        return 'start'
    if key == 'r':
        return 'rerecord'
    if recording and key in ('y', 'f'):
        return 'success' if key == 'y' else 'failure'
    return None


def operator_command(collector, key, recording):
    """Consume PICO edge events and terminal input into one episode state."""
    save = collector._manager_toggle_dc
    discard = collector._manager_toggle_da
    collector._manager_toggle_dc = collector._manager_toggle_da = False
    terminal = terminal_command(key,recording)
    command = source = None
    # An explicit terminal save must never become a destructive PICO discard.
    # Conflicting remote events also preserve the buffer for another decision.
    if terminal is not None:
        command, source = terminal, 'terminal'
    elif save and discard:
        source = 'pico_conflict'
        print('PICO 保存与丢弃信号冲突，保留当前片段；请重新按保存键或使用终端。',flush=True)
    elif discard:
        command, source = 'discard', 'pico'
    elif save:
        command, source = ('save' if recording else 'start'), 'pico'
    if command == 'discard':
        collector.discard_requires_start = source == 'terminal' and key == 'q'
    if source is not None:
        names = {'\n':'Enter','\r':'Enter',' ':'Space','\x03':'Ctrl+C','eof':'EOF'}
        event = dict(command=command or 'conflict',source=source,
            terminal_key=names.get(key,key) if source == 'terminal' else None,
            pico_save=bool(save),pico_discard=bool(discard),
            received_monotonic=time.monotonic(),received_at=time.time())
        if command == 'discard':
            event['requires_start_after_discard'] = collector.discard_requires_start
        collector.last_operator_event = event
        audit = getattr(collector,'operator_audit',None)
        if callable(audit):
            audit(event)
    return command


def collect_until_operator_decision(collector, exporter, index, keys, clock=time,
                                    recovery_wait_s=1.0, mark_success=False,
                                    capture_runtime=None, initial_recording=False):
    """Retain a valid prefix on lost input; operator keys decide start and end."""
    from brainco_gripper_data import MissingGripperData
    recording = False
    # A discarded active fragment rotates only its writers/transaction. Keep
    # the recording intent, but admit the new fragment's first frame only when
    # its own inputs and runtime capture are ready.
    resume_pending = bool(initial_recording)
    collector.recording_active = resume_pending
    frozen = False
    announced_ready = False
    ready_spoken = False
    last_notice = float('-inf')
    interruptions = []
    if not math.isfinite(recovery_wait_s) or not 0 <= recovery_wait_s <= 5:
        raise ValueError('采集恢复等待窗口必须在0–5秒之间')
    collector.recording_gaps = []
    collector.operator_events = []
    lifecycle = EpisodeLifecycle(True if mark_success else None)
    lifecycle.success_label_source = 'cli_mark_success' if mark_success else 'unknown'
    collector.episode_lifecycle = lifecycle
    missing_since = None
    missing_reason = None
    missing_after_frames = None
    source_segment = None
    next_tick = clock.monotonic()
    if resume_pending:
        print(f'RECORDING episode {index}：已清空旧片段，保持录制；有效数据就绪后从第 0 帧继续，无需再按开始。',flush=True)

    def finish_gap(recovered):
        nonlocal missing_since
        if missing_since is not None:
            gap = dict(reason=missing_reason,
                after_frames=missing_after_frames,duration_s=clock.monotonic()-missing_since,
                recovered=recovered, kind='missing_data')
            collector.recording_gaps.append(gap)
            lifecycle.gap(gap)
            missing_since = None

    def freeze(error):
        nonlocal frozen
        if frozen:
            return
        frozen = True
        lifecycle.freeze_reason=('video_backpressure' if
            error.__class__.__name__ == 'VideoBackpressureError' else 'data_timeout')
        interruptions.append(dict(reason=str(error),after_frames=exporter.episode_buffer['size']))
        announce(collector, '录制已暂停，请检查数据，保存或丢弃当前片段。')
        print(f'数据中断，保留 episode {index} 的 {exporter.episode_buffer["size"]} 帧，停止追加：{error}',flush=True)
        print('Enter/空格或 PICO 左Grip+A 保存有效帧；Q 舍弃后等待开始，X/左Grip+B 舍弃后继续录制，Ctrl+C 结束采集。不会自动退出或自动续接。',flush=True)

    while True:
        loop = clock.monotonic()
        collector.poll()
        command = operator_command(collector,keys.read(),recording or resume_pending)
        if command in ('discard', 'rerecord', 'exit'):
            lifecycle.end(command, frozen=frozen, unresolved_gap=missing_since is not None)
            finish_gap(False)
            return command,interruptions
        if command in ('success', 'failure'):
            lifecycle.task_success = command == 'success'
            lifecycle.success_label_source = 'operator_terminal'
            print('任务结果已标记：'+('成功' if lifecycle.task_success else '失败')+'；保存键仍需另按。',flush=True)
        if command == 'save':
            if exporter.episode_buffer['size'] >= 2:
                if not frozen and missing_since is None:
                    try:
                        collector.readiness()
                    except MissingGripperData as error:
                        # The save key can arrive on the very first failed probe.
                        # Classify the end without appending a synthetic frame.
                        missing_since=clock.monotonic()
                        missing_reason=str(error)
                        missing_after_frames=exporter.episode_buffer['size']
                lifecycle.end('save', frozen=frozen, unresolved_gap=missing_since is not None)
                finish_gap(False)
                return 'save',interruptions
            print('帧数不足2帧，暂不能保存；可继续采集或按 X 丢弃。',flush=True)
        if frozen:
            if command == 'start':
                print('当前条已停止追加；请保存，Q 舍弃后等待开始，X/左Grip+B 舍弃后继续，R 人工复位重录，Ctrl+C 结束采集。',flush=True)
        else:
            try:
                collector.readiness()
                if missing_since is not None:
                    if clock.monotonic()-missing_since >= recovery_wait_s:
                        freeze(missing_reason)
                        finish_gap(False)
                        continue
                    duration = clock.monotonic()-missing_since
                    finish_gap(True)
                    print(f'数据已恢复，继续 episode {index}；等待 {duration:.2f} 秒期间未写入帧。',flush=True)
                if not recording:
                    if not announced_ready and not resume_pending:
                        print(f'READY episode {index}：当前输入检查通过，尚未录制，等待你确认开始。',flush=True)
                        print('按 S / Enter / 空格或 PICO 左Grip+A；开始时会再次检查数据，看到 RECORDING 后再执行任务。',flush=True)
                        announced_ready = True
                        if not ready_spoken:
                            announce(collector, f'第{index}条已就绪，请按开始键录制。')
                            ready_spoken = True
                    if command == 'start' or resume_pending:
                        if capture_runtime is not None:
                            lifecycle.runtime_capture = capture_runtime(collector)
                        recording = True
                        resume_pending = False
                        collector.recording_active = True
                        collector._episode_state.change_state()
                        announce(collector, f'第{index}条开始录制。')
                        print(f'RECORDING episode {index}：Y 成功 / F 失败；Enter/空格或左Grip+A 保存，Q 舍弃后等待开始，X/左Grip+B 舍弃后继续，R 重录，Ctrl+C 结束采集。',flush=True)
                if recording:
                    if collector._add_data_frame() is False:
                        raise MissingGripperData('当前帧未成功采集')
                    samples = getattr(collector, 'timing_samples', [])
                    sample = samples[-1] if samples else {}
                    decision = sample.get('decision_monotonic')
                    current_segment = sample.get('sample_segment')
                    if (source_segment is not None and current_segment is not None and
                        current_segment != source_segment):
                        lifecycle.gap(dict(reason='source_restart', kind='source_restart',
                            source_reasons=sample.get('segment_reasons', []),
                            after_frames=exporter.episode_buffer['size']-1,
                            duration_s=0., recovered=True))
                    source_segment = current_segment
                    lifecycle.frame(exporter.episode_buffer['size']-1, decision,
                                    collector.loop_period if decision is not None else None)
                    if loop-last_notice > 1:
                        metrics=getattr(collector,'collection_metrics',None)
                        status=''
                        if metrics is not None:
                            summary=metrics.summary()
                            hz=summary['effective_sample_hz']
                            camera_labels=(('ego_view','头'),('left_wrist','左'),('right_wrist','右'))
                            rgb=[]
                            for name,label in camera_labels:
                                if name not in summary['cameras']:
                                    continue
                                camera=summary['cameras'][name]
                                rate=camera['effective_unique_hz']
                                repeated=camera['repeated_fraction']
                                rgb.append(label+':'+('--' if rate is None else f'{rate:.1f}')+
                                    'Hz/'+('--' if repeated is None else f'{repeated:.0%}')+'重复')
                            status=' | 采样 '+('--' if hz is None else f'{hz:.1f}')+'Hz | '+', '.join(rgb)
                        print(f'录制中 episode {index}: {exporter.episode_buffer["size"]} 帧'+status,flush=True)
                        last_notice=loop
            except Exception as error:
                if not isinstance(error, MissingGripperData) and error.__class__.__name__ != 'VideoBackpressureError':
                    raise
                announced_ready=False
                if recording:
                    if missing_since is None:
                        missing_since = clock.monotonic()
                        missing_reason = str(error)
                        missing_after_frames = exporter.episode_buffer['size']
                        print(f'录制数据暂不可用，暂停追加 episode {index}：{error}；等待恢复最多 {recovery_wait_s:g} 秒，仍可保存或丢弃。',flush=True)
                    if (error.__class__.__name__ == 'VideoBackpressureError' or
                        clock.monotonic()-missing_since >= recovery_wait_s):
                        freeze(error)
                        finish_gap(False)
                elif command == 'start' or loop-last_notice > 2:
                    if resume_pending:
                        print(f'RECORDING episode {index}：等待有效数据：{error}；恢复后继续录制，无需按开始。',flush=True)
                    else:
                        request_notice = '开始请求未执行；' if command == 'start' else ''
                        print(f'WAIT episode {index}：尚未录制；'+request_notice+str(error)+
                            '。输入就绪后会显示 READY，再按 S / Enter / 空格或 PICO 左Grip+A 开始。',flush=True)
                    last_notice=loop
        duration = clock.monotonic()-loop
        metrics = getattr(collector, 'collection_metrics', None)
        if metrics is not None:
            metrics.observe_loop(duration, collector.loop_period)
        # Keep a monotonic sampling phase rather than accumulating OS sleep
        # latency over long episodes. Never emit catch-up rows after overruns.
        next_tick += collector.loop_period
        now=clock.monotonic()
        if next_tick <= now:
            next_tick += (int((now-next_tick)/collector.loop_period)+1)*collector.loop_period
        clock.sleep(max(0,next_tick-now))


def wait_for_manual_reset(collector, index, keys, clock=time):
    """Unrecorded operator reset; does not command or automatically start a robot."""
    announce(collector, '请人工复位任务，完成后确认。')
    print(f'RESET episode {index}：人工复位后按 Enter/S/空格或 PICO 左Grip+A 确认；再按开始键录制。Ctrl+C 结束采集；Q/X/左Grip+B 继续等待。',flush=True)
    while True:
        collector.poll()
        command = operator_command(collector, keys.read(), False)
        if command == 'exit':
            return False
        if command == 'discard':
            print(f'当前没有未保存片段，继续等待复位 episode {index}；Ctrl+C 才结束采集。',flush=True)
        if command == 'start':
            return True
        clock.sleep(collector.loop_period)


class ConsoleKeys:
    def __enter__(self):
        self.original = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
        if self.original:
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            # TextIO may prefetch the following key into its private buffer,
            # making select(fd) miss it forever (e.g. rapid Y then Space).
            key = os.read(sys.stdin.fileno(),1)
            # EOF must remain a real exit now that Q only clears a fragment.
            return 'eof' if key == b'' else key.decode('ascii',errors='ignore').lower()
        return None

    def __exit__(self, *args):
        if self.original:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original)


def load_sonic(local, backend='legacy'):
    sonic_root = Path(local['sonic_root'])
    sys.path.insert(0, str(sonic_root))
    spec = importlib.util.spec_from_file_location('episode_sonic_exporter',
        sonic_root/'gear_sonic/scripts/run_data_exporter.py')
    sonic = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sonic
    spec.loader.exec_module(sonic)
    # The upstream writer can flush while its last frame is still encoding.
    # Use a joined worker for this entry point and verify every saved video.
    if backend == 'legacy':
        from gear_sonic.data import exporter as exporter_module
        from episode_video_writer import EpisodeVideoWriter
        exporter_module.VideoWriter = EpisodeVideoWriter
        install_empty_metadata_indexes(exporter_module.Gr00tDatasetMetadata)
        sonic.Gr00tDataExporter = exporter_module.Gr00tDataExporter
    return sonic


def make_collector(sonic, exporter, model, local, mode):
    import numpy as np
    from brainco_gripper_data import adapt_frame, MissingGripperData
    from observation_stream import ObservationStream
    from episode_operator_controls import PicoEpisodeGesture
    head_source = local['camera_sources']['ego_view'] if camera_keys(local) == ('ego_view',) else None
    camera_host = head_source['host'] if head_source else '127.0.0.1'
    camera_port = head_source['port'] if head_source else local['camera_output_port']

    class EpisodeCollector(sonic.GrootDataCollector):
        def __init__(self):
            super().__init__(camera_host=camera_host, camera_port=camera_port,
                data_exporter=exporter, robot_model=model, frequency=local['dataset_frequency'],
                sonic_data_zmq_host='127.0.0.1', state_zmq_host='127.0.0.1',
                sonic_data_zmq_port=local.get('pico_port',5556),
                state_zmq_port=local.get('state_port',5557))
            # Replace the upstream clients that conflate body history and strip
            # camera timing metadata. Only the manager/PICO reader stays upstream.
            self._state_subscriber.close()
            self._image_subscriber.close()
            self.observations = ObservationStream(local)
            self.snapshot = None
            self.manager = {}
            self.recording_gesture = PicoEpisodeGesture()
            self.legacy_recording_notice = False
            self.body_seen = self.manager_seen = float('-inf')
            self.packet = {}
            self.ages = None
            self.recording_voice = RecordingVoice(local, enabled=local.get('recording_voice', True))
            self.bind_exporter(exporter)

        def bind_exporter(self, exporter):
            """Rotate staged writers while the subscriptions stay connected."""
            self.data_exporter = exporter
            if hasattr(self, '_episode_state'):
                self._episode_state.reset_state()
            self._initial_yaw = None
            self.grasp_samples = []
            self.timing_samples = []
            self.collection_metrics = EpisodeMetrics(camera_keys(local))
            if hasattr(self.observations, 'reset_episode'):
                self.observations.reset_episode()
            else:
                # Older receiver fallback; the receive history remains intact.
                with self.observations.lock:
                    assembler = self.observations.assembler
                    assembler.last_ids = assembler.last_decision = assembler.episode_start = None
            self.record_grasp_features = 'teleop.grasp_profile' in getattr(exporter,'features',{})
            self.record_timing_features = 'observation.sync_offset' in getattr(exporter,'features',{})
            self.record_training_actions = 'action.hand_q' in getattr(exporter,'features',{})
            original_add = exporter.add_frame

            def add_frame(frame):
                decision = self.snapshot['timing']['decision_monotonic']
                self.snapshot['timing']['operator'] = dict(manager_received_monotonic=self.manager_seen,
                    pico_input_age_s=self.ages[2],stream_mode=self.current_stream_mode,
                    pose_received_wall=(self.latest_sonic_msg or {}).get('receive_timestamp'))
                frame = adapt_frame(frame, model, self.packet, decision)
                frame['observation.record_receive_age'] = np.asarray(self.ages,dtype=np.float32)
                sample = grasp_sample(self.manager,self.packet,required=self.record_grasp_features)
                lifecycle = getattr(self, 'episode_lifecycle', None)
                if lifecycle is not None and lifecycle.runtime_capture is not None:
                    sample['runtime_matches_capture'] = runtime_sample_matches(sample, lifecycle.runtime_capture)
                if self.record_grasp_features:
                    frame.update(sample_features(sample))
                if self.record_training_actions:
                    frame.update(training_action_values(sample, lifecycle.runtime_capture['grasp_config']))
                if self.record_timing_features:
                    start = self.observations.assembler.episode_start
                    frame.update(timing_values(self.snapshot,decision if start is None else start))
                original_add(frame)
                timing = self.observations.commit(self.snapshot)
                self.timing_samples.append(timing)
                self.collection_metrics.observe(timing)
                self.grasp_samples.append(sample)  # Append only after exporter accepts the frame.
                if sample.get('runtime_matches_capture') is False:
                    lifecycle.runtime_sample_mismatches.append(len(self.grasp_samples)-1)
            exporter.add_frame = add_frame

        def _poll_state_zmq(self):
            pass  # The common receiver continuously buffers timestamped states.

        def _handle_manager_state(self, raw):
            # Upstream rising edges overlap A+X/B+Y and latch old abort packets.
            # Preserve only events already accepted by our complete-gesture gate.
            pending = self._manager_toggle_dc,self._manager_toggle_da
            super()._handle_manager_state(raw)
            self._manager_toggle_dc,self._manager_toggle_da = pending
            self.manager = sonic.unpack_pose_message(raw, topic='manager_state')
            self.manager_seen = time.monotonic()
            event = self.recording_gesture.feed(self.manager,self.manager_seen)
            if event == 'toggle_save':
                self._manager_toggle_dc = True
            elif event == 'discard':
                self._manager_toggle_da = True
            elif self.recording_gesture.diagnostics.get('status') == 'legacy' and not self.legacy_recording_notice:
                print('当前 PICO 录制按键版本较旧，请用终端开始/保存/丢弃；下次完整启动 SONIC 后加载安全组合键。',flush=True)
                self.legacy_recording_notice = True
            if hasattr(self.observations, 'append_archive'):
                self.observations.append_archive('manager', self.manager, self.manager_seen, time.time())

        def _archive_sonic_message(self, raw, topic):
            try:
                payload=sonic.unpack_pose_message(raw,topic=topic)
            except Exception:
                return
            self.observations.append_archive(topic,payload,time.monotonic(),time.time())

        def _handle_pose_message(self, raw):
            super()._handle_pose_message(raw)
            self._archive_sonic_message(raw,'pose')

        def _handle_planner_message(self, raw):
            super()._handle_planner_message(raw)
            self._archive_sonic_message(raw,'planner')

        def _finalize_frame(self, start):
            return True  # Only this process's save key commits an episode.

        def poll(self):
            self._poll_sonic_zmq_messages()

        def clear_pending_operator_events(self):
            # Encoding/commit deliberately does not poll PICO. Discard queued
            # pre-READY button edges rather than letting them acknowledge RESET
            # or start a fresh episode without another intentional button press.
            if self._sonic_zmq_socket is not None:
                import zmq
                deadline=time.monotonic()+.1
                for _ in range(100):
                    if not self._sonic_zmq_socket.poll(0,zmq.POLLIN):
                        break
                    self._poll_sonic_zmq_messages()
                    if time.monotonic() >= deadline:
                        break
            self._manager_toggle_dc=self._manager_toggle_da=False
            self.recording_gesture.reset()

        def readiness(self):
            self.snapshot = self.observations.snapshot()
            self.latest_proprio_msg = self.snapshot['proprio']
            self.latest_image_msg = self.snapshot['image_message']
            self.packet = self.snapshot['packet']
            self.body_seen = self.snapshot['timing']['body']['received_monotonic']
            now = time.monotonic()
            if now-self.body_seen > .2:
                raise MissingGripperData('等待 SONIC 实际状态/动作；需先启动策略')
            for key in ('body_q','last_action'):
                value=np.asarray(self.latest_proprio_msg.get(key,[]))
                if value.shape != (29,) or not np.isfinite(value).all():
                    raise MissingGripperData('身体状态/动作无效: '+key)
            if now-self.manager_seen > .5 or not self._extract_bool(self.manager,'brainco_input_valid'):
                raise MissingGripperData('PICO 输入缺失或过期')
            if self.current_stream_mode not in ((1,) if mode == 'pose' else (1,2,3,5)):
                raise MissingGripperData('等待 POSE 全身模式（A+X 切换）' if mode == 'pose' else '等待活动控制模式')
            if self.current_stream_mode == 1:
                pose=self.latest_sonic_msg
                if pose is None or not 0 <= time.time()-pose.get('receive_timestamp',0) <= .1:
                    raise MissingGripperData('全身 POSE 数据缺失或过期')
                for key,shape in [('smpl_joints',(24,3)),('smpl_pose',(63,))]:
                    value=np.asarray(pose.get(key,[]))
                    if value.shape != shape or not np.isfinite(value).all():
                        raise MissingGripperData('全身 POSE 数据无效: '+key)
            image = self.latest_image_msg or {}
            camera_ages = [item['age_ms']/1000 for item in self.snapshot['timing']['cameras'].values()]
            pico_age = float(self.manager.get('pico_input_age_ms',np.array([float('inf')])).flat[0])/1000
            if not 0 <= pico_age <= .5:
                raise MissingGripperData('PICO 跟踪帧过期')
            if self.record_grasp_features:
                sample=grasp_sample(self.manager,self.packet,required=True)
                if self.record_training_actions:
                    for field in ('state_joints','action_joints','action_speed_joints'):
                        if sample.get(field) is None:
                            raise MissingGripperData('等待新版实际手部遥测：'+field+'；下次完整启动 SONIC 后再采新格式任务')
            self.ages = [now-self.body_seen,max(camera_ages),pico_age]

        def close(self):
            self.recording_voice.close()
            self.observations.close()
            self._keyboard_listener.close()
            if self._sonic_zmq_socket is not None:
                self._sonic_zmq_socket.close(linger=0)
                self._sonic_zmq_ctx.term()

    return EpisodeCollector()


def cancel_buffer(exporter):
    # Cancel only the unsaved buffer. No old episode or metadata is touched.
    episode = int(exporter.meta.total_episodes)
    if getattr(exporter,'backend',None) == 'lerobot':
        exporter.cancel()
        return episode
    for writer in exporter.video_writers.values():
        writer.cancel()
    images = Path(exporter.root)/'images'
    # Images are placeholder paths for direct video writers; no image files written.
    if images.exists():
        import shutil
        shutil.rmtree(images)
    exporter.episode_buffer = exporter.create_episode_buffer()
    return episode


def record(args, local, destination, index):
    from brainco_gripper_data import adapt_schema
    from episode_transaction import EpisodeTransaction
    from episode_video_writer import install_atomic_frame_admission
    backend = getattr(args,'backend','legacy')
    sonic = load_sonic(local,backend)
    model = sonic.get_g1_robot_model()
    features, modalities = adapt_schema(sonic.get_features_sonic_vla(model),
        sonic.get_modality_config_sonic_vla(model),model)
    selected_cameras = camera_keys(local)
    if len(selected_cameras) == 3:
        features.update(sonic.get_wrist_camera_features())
        modalities['video'].update(sonic.get_wrist_camera_modality_config()['video'])
    features['observation.record_receive_age'] = dict(dtype='float32',shape=(3,),
        names=['body_receive_age','oldest_camera_receive_age','pico_input_age'])
    info_path = destination/'meta/info.json'
    existing = json.loads(info_path.read_text()) if info_path.exists() else None
    if use_timing_features(existing):
        features.update(timing_features(local))
    if use_grasp_features(existing):
        features.update(grasp_features())
        modalities.setdefault('teleop',{}).update({
            'grasp_profile':dict(start=0,end=2,original_key='teleop.grasp_profile'),
            'grasp_closure':dict(start=0,end=2,original_key='teleop.grasp_closure')})
    if use_training_action_features(existing):
        features.update(training_action_features())
    if info_path.exists():
        from episode_session import validate_dataset_cameras
        validate_dataset_cameras(destination, selected_cameras)
        expected = {k:(v['dtype'],list(v['shape'])) for k,v in features.items()}
        actual = {k:(v['dtype'],list(v['shape'])) for k,v in existing['features'].items() if k in features}
        if actual != expected or existing['fps'] != local['dataset_frequency']:
            raise ValueError('已有数据集字段/FPS 不同，请用新任务名')
    repaired = ensure_empty_episode_indexes(destination) if backend == 'legacy' else []
    if repaired:
        print('已补齐空数据集索引，episode 仍从0开始: '+', '.join(repaired),flush=True)
    config = sonic.poll_robot_config_zmq('127.0.0.1',local.get('state_port',5557),args.wait_timeout)
    collector = None
    exporter = None
    transaction = None
    committing = False
    saved_count = 0
    resume_recording = False
    # The limit belongs only to this invocation, never to the task dataset.
    # Omitted/zero means a persistent session that the operator ends manually.
    num_episodes = getattr(args, 'num_episodes', None) or None
    recovery_wait_s=float(local.get('recording_input_wait_s',1.0))

    def create_staged_exporter(current):
        script_config={**config,'chenyz':dict(task=args.task,entrypoint='record_episode.sh',
                hardware_synchronized=False,timestamp_kind='workstation_receive_time',
                observation_contract=contract(local),
                cameras=list(selected_cameras),camera_mode=local.get('recording_cameras','all'),
                gripper_mean_semantics='legacy six-channel arithmetic mean, not trigger coordinate',
                wrist_mapping=local.get('wrist_mapping',{}) if len(selected_cameras) == 3 else {})}
        if backend == 'lerobot':
            from lerobot_backend import LeRobotExporter
            staged = LeRobotExporter(current.staging_root,local['dataset_frequency'],
                features,modalities,args.prompt,script_config)
        else:
            staged = sonic.Gr00tDataExporter.create(save_root=current.staging_root,
                fps=local['dataset_frequency'],features=features,modality_config=modalities,
                task=args.prompt,script_config=script_config)
            install_atomic_frame_admission(staged)
        (current.staging_root/'task.json').write_text(json.dumps(dict(task=args.task,prompt=args.prompt,
            codebase_version='v3.0' if backend=='lerobot' else 'v2.1',
            backend=backend,episode_index_base=0,cameras=list(selected_cameras)),ensure_ascii=False,indent=2))
        return staged

    def begin_recording(current):
        capture = capture_recording_runtime(local, current.manager, root=ROOT)
        if getattr(current,'record_training_actions',False) and capture['grasp_config'] is None:
            from brainco_gripper_data import MissingGripperData
            raise MissingGripperData('等待运行中的抓姿配置快照；下次完整启动 SONIC 后再采新格式任务')
        if getattr(args,'raw_archive',False) and hasattr(current, 'observations'):
            current.observations.start_archive(transaction.staging_root/'meta'/f'chenyz_raw_{index:06d}.jsonl')
        if not capture['runtime_verified']:
            print('运行配置尚未验证，本条只可作诊断记录：'+', '.join(capture['blockers']),flush=True)
        return capture

    def audit_operator(event):
        if transaction is None:
            return  # RESET has no unsaved episode transaction.
        item = {**event,'task':args.task,'episode':index,
                'frames':int(collector.data_exporter.episode_buffer['size'])}
        collector.operator_events.append(item)
        # Keep the action's source even if an intentional discard removes media.
        with (transaction.directory/'operator-events.jsonl').open('a') as output:
            output.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
        if item['command'] in ('save','discard','rerecord','exit'):
            label = ('终端 '+str(item['terminal_key']) if item['source']=='terminal' else 'PICO 完整组合键')
            print(f'episode {index} 收到 {item["command"]} 请求，来源：{label}。',flush=True)

    def quarantine_active(reason):
        nonlocal transaction, exporter
        if collector is not None and hasattr(collector, 'observations'):
            collector.observations.stop_archive()
        if exporter is not None:
            # Drain a bounded writer before relocating its open file. Preserve
            # failure artifacts; do not remove the fragment during diagnostics.
            for writer in getattr(exporter,'video_writers',{}).values():
                try:
                    writer.stop()
                except Exception as error:
                    print('关闭诊断视频时发生错误：'+str(error),file=sys.stderr,flush=True)
            if getattr(exporter,'backend',None)=='lerobot':
                try:
                    exporter.close()
                except Exception as error:
                    print('关闭官方 LeRobot 写入器时发生错误：'+str(error),file=sys.stderr,flush=True)
            exporter=None
        if transaction is not None:
            transaction.abort(reason)
            transaction=None

    try:
        print(f'任务: {args.task} | episode {index} | {destination}',flush=True)
        if backend == 'lerobot':
            print('保存后端：官方 LeRobot 0.6.1 / v3.0；AV1、CRF 30、GOP 2；录制时写 PNG，保存时编码并完整校验。',flush=True)
        print('手动控制：S 开始；Enter/空格或 PICO 左Grip+A 开始/保存；Y 成功、F 失败；Q 舍弃片段并等待开始，X/左Grip+B 舍弃后继续录制，R 人工复位重录；Ctrl+C 结束采集。',flush=True)
        if len(selected_cameras) == 1:
            print('相机：仅顶部 ego_view，直接读取头部 RGB；不等待腕部相机。身体/手部状态贴近图像，动作取当前实际命令。时间信息逐帧保存。',flush=True)
        else:
            print('三路 RGB 按本机接收时间配对≤25ms；身体/手部状态贴近图像，动作取当前实际命令。时间信息逐帧保存。',flush=True)
        print('状态：WAIT 等待数据；READY 当前输入检查通过、等待开始；RECORDING 正在录制。启动时等待首批消息属于正常现象。',flush=True)
        print(f'录制中缺数据时暂停写入，等待恢复最多 {recovery_wait_s:g} 秒；超时后保留已有帧，等待你保存或丢弃。',flush=True)
        print('训练质量：仅含数据缺口或其他质量问题的片段默认排除训练；开始前等待不计入片段缺口。保存时会报告本条训练准入结果。',flush=True)
        session_plan=f'计划保存 {num_episodes} 条' if num_episodes is not None else '持续采集，不限条数；由你主动退出'
        print(f'SESSION：{session_plan}；保存不会自动标记任务成功，也不会停止 SONIC。',flush=True)
        with ConsoleKeys() as keys:
            while num_episodes is None or saved_count < num_episodes:
                transaction = EpisodeTransaction(destination,index)
                exporter = create_staged_exporter(transaction)
                if collector is None:
                    collector = make_collector(sonic,exporter,model,local,args.mode)
                    collector.operator_audit = audit_operator
                    if hasattr(collector,'clear_pending_operator_events'):
                        collector.clear_pending_operator_events()
                else:
                    collector.bind_exporter(exporter)
                collection_started=time.monotonic()
                decision,interruptions=collect_until_operator_decision(collector,exporter,index,keys,
                    recovery_wait_s=recovery_wait_s,mark_success=getattr(args,'mark_success',False),
                    capture_runtime=begin_recording,initial_recording=resume_recording)
                collection_elapsed=time.monotonic()-collection_started
                archive = (collector.observations.stop_archive() if getattr(args,'raw_archive',False) and
                    hasattr(collector, 'observations') and collector.episode_lifecycle.runtime_capture is not None else None)
                if decision in ('discard', 'rerecord', 'exit'):
                    from episode_transaction import atomic_json
                    transaction.journal['operator_events'] = list(collector.operator_events)
                    transaction.journal['discarded_frames'] = int(exporter.episode_buffer['size'])
                    atomic_json(transaction.journal_path,transaction.journal)
                    cancel_buffer(exporter)
                    exporter = None
                    transaction.abort('operator_'+decision)
                    transaction = None
                    print(f'已丢弃未提交片段，episode {index} 编号未增加；SONIC 继续运行。',flush=True)
                    announce(collector, '当前片段已丢弃。')
                    if decision == 'exit':
                        print(f'SESSION 已结束，保存 {saved_count} 条；SONIC 继续运行。',flush=True)
                        return 2
                    if hasattr(collector,'clear_pending_operator_events'):
                        collector.clear_pending_operator_events()
                    if decision == 'discard':
                        resume_recording = collector.recording_active and not collector.discard_requires_start
                        if collector.discard_requires_start:
                            print(f'已舍弃 episode {index}，等待确认开始：按 S / Enter / 空格或 PICO 左Grip+A 录制新片段；采集进程保持运行。',flush=True)
                        elif not resume_recording:
                            print(f'episode {index} 尚未开始，继续等待开始键；采集进程保持运行。',flush=True)
                        continue
                    resume_recording = False
                    if not wait_for_manual_reset(collector,index,keys):
                        return 2
                    continue
                frames=exporter.episode_buffer['size']
                announce(collector, f'正在保存第{index}条，请稍候。')
                committing=True
                recording_writers=dict(getattr(exporter,'video_writers',{}))
                phase_started=time.monotonic()
                exporter.save_episode()
                save_elapsed=time.monotonic()-phase_started
                phase_started=time.monotonic()
                result=verify_saved_episode(exporter,index,frames)
                verify_elapsed=time.monotonic()-phase_started
                phase_started=time.monotonic()
                staged_root=transaction.staging_root
                if len(collector.timing_samples) != frames:
                    raise ValueError('时间信息帧数与 episode 不一致')
                result.update(save_timing_sidecar(staged_root,index,collector.timing_samples,contract(local)))
                capture=collector.episode_lifecycle.runtime_capture
                grasp_samples = getattr(collector,'grasp_samples',None)
                if grasp_samples is not None:
                    if len(grasp_samples) != frames:
                        raise ValueError('抓姿输入帧数与 episode 不一致')
                    result.update(save_grasp_sidecar(staged_root,index,grasp_samples,
                        (capture or {}).get('grasp_config')))
                archive_exclusions=[]
                if archive is not None and not archive.get('complete',False):
                    archive_exclusions.append('raw_metadata_archive_incomplete')
                if grasp_samples is None or any(sample.get('action_speed_joints') is None for sample in grasp_samples):
                    archive_exclusions.append('actual_hand_speed_telemetry_missing')
                quality=build_episode_quality(collector.episode_lifecycle,
                    storage_valid=result.get('storage_valid',result.get('verified',False)),
                    additional_exclusions=archive_exclusions)
                result.update(quality)
                result['quality']=quality
                result.update(task=args.task,prompt=args.prompt,recorded_at=time.time(),
                    cameras=list(selected_cameras),camera_mode=local.get('recording_cameras','all'),
                    recording_control='terminal_and_pico',data_interruptions=interruptions,
                    recording_input_wait_s=recovery_wait_s,runtime_capture=capture,
                    recording_session=dict(requested_episodes=num_episodes,continuous=num_episodes is None,saved_ordinal=saved_count+1,
                        started_after_discard=resume_recording,
                        raw_metadata_archive_enabled=getattr(args,'raw_archive',False)),
                    operator_events=list(collector.operator_events))
                if hasattr(collector,'observations'):
                    result['stream_metrics']=collector.observations.metrics()
                    result['stream_metrics']['scope']='receiver lifetime; collection_metrics covers this episode'
                if archive is not None:
                    # Absolute staging paths are not stable after publication.
                    archive['path']=str(Path(archive['path']).relative_to(staged_root))
                    result['raw_metadata_archive']=archive
                    result['stream_metrics']['archive']=archive
                metrics=getattr(collector,'collection_metrics',None)
                if metrics is not None:
                    result['collection_metrics']=metrics.summary()
                result['video_writer_metrics']={key:writer.metrics_snapshot()
                    for key,writer in recording_writers.items()
                    if hasattr(writer,'metrics_snapshot')}
                if getattr(exporter,'backend',None)=='lerobot':
                    result['dataset_backend']=exporter.metrics_snapshot()
                result['phase_metrics']=dict(operator_wait_and_recording_s=collection_elapsed,
                    video_and_parquet_save_s=save_elapsed,storage_verify_s=verify_elapsed,
                    sidecar_and_quality_finalize_s=time.monotonic()-phase_started)
                (staged_root/'meta'/f'chenyz_episode_{index:06d}.json').write_text(
                    json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
                # save_episode starts empty next-episode writers; close them before
                # the transaction publishes only a complete, verified episode.
                cancel_buffer(exporter)
                exporter=None
                transaction.commit(result)
                transaction=None
                committing=False
                saved_count+=1
                announce(collector, f'第{index}条已保存。' +
                         ('请查看质量报告。' if result['quality_exclusions'] else ''))
                session_progress=(f'{saved_count}/{num_episodes}' if num_episodes is not None else
                    f'本次已保存 {saved_count} 条，不限条数')
                print(f'已完整提交 episode {index}，{frames} 帧，任务结果：'+
                    ('成功' if result['task_success'] is True else '失败' if result['task_success'] is False else '未知')+
                    f'；训练准入：{result["training_eligible"]}。SESSION {session_progress}；下一条 {index+1}。SONIC 继续运行。',flush=True)
                index+=1
                resume_recording = False
                if hasattr(collector,'clear_pending_operator_events'):
                    collector.clear_pending_operator_events()
                if (num_episodes is None or saved_count < num_episodes) and not wait_for_manual_reset(collector,index,keys):
                    print(f'SESSION 已结束，保存 {saved_count} 条；SONIC 继续运行。',flush=True)
                    return 0
        return 0
    except KeyboardInterrupt:
        print('采集收到中断，已退出；SONIC 继续运行。Q 舍弃片段后等待开始，X/左Grip+B 舍弃后继续录制；Ctrl+C/停止信号会结束采集。',flush=True)
        if transaction is not None:
            quarantine_active('interrupted_during_commit' if committing else 'operator_interrupt')
        return 2
    except BaseException as error:
        if transaction is not None:
            quarantine_active(str(error))
        raise
    finally:
        if collector is not None:
            collector.close()
        if exporter is not None and not committing:
            cancel_buffer(exporter)
        elif exporter is not None:
            # Preserve failed save artifacts for inspection; never overwrite them.
            for writer in getattr(exporter,'video_writers',{}).values():
                writer.stop()
            if getattr(exporter,'backend',None)=='lerobot':
                exporter.close()


def main():
    parser = argparse.ArgumentParser(description='按任务累计录制官方 LeRobot v3.0；默认持续采到主动退出，不停止 SONIC')
    parser.add_argument('task',help='任务名；每个任务一个独立数据集')
    parser.add_argument('episode',type=int,nargs='?',help='可选预期编号，从 0 开始；默认自动续写')
    parser.add_argument('--prompt',help='任务描述；默认使用任务名，已有任务不可更改')
    parser.add_argument('--root',type=Path,default=ROOT/'data/tasks')
    parser.add_argument('--config',type=Path,default=ROOT/'config/local.json')
    parser.add_argument('--backend',choices=('lerobot','legacy'),default='lerobot',
                        help='默认官方 LeRobot 0.6.1/v3；legacy 仅用于续采旧 v2.1 任务')
    parser.add_argument('--mode',choices=('pose','any'),default='any',help='默认允许活动控制模式；pose 可限制只采全身模式')
    parser.add_argument('--cameras',choices=('all','head'),default='all',help='all：顶部+左右腕；head：只采顶部 RGB，不等待腕部；同一任务不可混用')
    parser.add_argument('--wait-timeout',type=float,default=120,help='初始化时等待机器人配置的时限；就绪后手动控制等待和开始')
    parser.add_argument('--countdown',type=float,default=0,help='兼容旧命令，已停用自动倒计时；始终由终端或PICO手动开始')
    parser.add_argument('--num-episodes',type=int,default=None,help='可选本次保存上限；默认不限，0也不限，1为单条；任务数据始终累计续号')
    parser.add_argument('--mark-success',action='store_true',help='操作者明确声明本次保存的任务已成功；不会覆盖断流/配置问题造成的训练排除')
    parser.add_argument('--raw-archive',action='store_true',help='额外归档原始状态/手部/PICO遥测及相机时间元数据，含操作者姿态；不额外复制RGB，默认关闭')
    parser.add_argument('--dry-run',action='store_true',help='只显示保存位置和下一编号，无网络、无录制')
    args = parser.parse_args()
    if not math.isfinite(args.wait_timeout) or not math.isfinite(args.countdown) or args.wait_timeout <= 0 or args.countdown < 0:
        parser.error('等待时间必须为正，倒计时不可为负')
    if args.num_episodes is not None and args.num_episodes < 0:
        parser.error('--num-episodes 不可为负；省略或0不限条数，正整数为本次上限')
    if args.num_episodes == 0:
        args.num_episodes = None
    destination = task_path(args.root,args.task)
    if args.prompt is None:
        manifest = destination/'task.json'
        args.prompt = json.loads(manifest.read_text())['prompt'] if manifest.exists() else args.task
    if args.dry_run:
        index = dataset_plan(destination,args.task,args.prompt,args.episode,
            camera_keys=camera_keys({'recording_cameras':args.cameras}),
            codebase_version='v3.0' if args.backend=='lerobot' else 'v2.1')
        print(json.dumps(dict(task=args.task,prompt=args.prompt,episode=index,
            cameras=list(camera_keys({'recording_cameras':args.cameras})),camera_mode=args.cameras,
            num_episodes=args.num_episodes,continuous=args.num_episodes is None,
            task_success=True if args.mark_success else None,
            backend=args.backend,codebase_version='v3.0' if args.backend=='lerobot' else 'v2.1',
            destination=str(destination),robot_commands_sent=False),ensure_ascii=False,indent=2))
        return 0
    local = json.loads(args.config.read_text())
    local['recording_cameras']=args.cameras
    local['recording_raw_archive']=bool(args.raw_archive)
    if local.get('pico_port',5556)==5556 and local.get('state_port',5557)==5557:
        from start_sonic import processes
        if processes('export_dataset.py'):
            raise RuntimeError('旧时间戳 exporter 仍在运行；先运行 bash start_sonic.sh，它会关闭旧采集入口且不重启正在运行的 SONIC')
    if args.cameras == 'all' and local['wrist_mapping'] != {'video6':'left_wrist','video8':'right_wrist'}:
        raise ValueError('左右腕相机身份未确认')
    with collection_lock(ROOT/'logs/task-collection.lock'):
        from episode_transaction import recover_dataset
        recover_dataset(destination)
        # Recheck after acquiring the lock, never race two appenders.
        index = dataset_plan(destination,args.task,args.prompt,args.episode,camera_keys=camera_keys(local),
            codebase_version='v3.0' if args.backend=='lerobot' else 'v2.1')
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        return record(args,local,destination,index)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('采集失败: '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(1)
