#!/usr/bin/env python3
"""Run SONIC's manager with the locally extracted XRoboToolkit service."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from dex3_controls import install_continuous_hands
from brainco_controls import install_brainco_hands
from recording_runtime import write_runtime_manifest


class ManagerStateRateLimiter:
    """Bound manager iterations at 50 Hz, including early-return input stalls.

    The injected clock and sleeper allow checking timing without loading the SDK.
    The first status publish is immediate, and missed periods are never replayed.
    """

    def __init__(self, hz=50, *, clock=time.monotonic, sleep=time.sleep):
        self.period = 1 / hz
        self.clock = clock
        self.sleep = sleep
        self.last_publish = None

    def wait(self):
        now = self.clock()
        if self.last_publish is not None:
            remaining = self.last_publish + self.period - now
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self.last_publish = now


class PoseTimingDiagnostics:
    """Measure publishing stalls without changing the pose or its pacing."""

    def __init__(self, *, clock=time.monotonic, report=print):
        self.clock = clock
        self.report = report
        self.reset()

    def reset(self):
        self.started = self.clock()
        self.last_publish = None
        self.last_stamp = None
        self.published = self.input_changes = self.iterations = 0
        self.max_publish_gap_ms = self.max_step_ms = self.max_compute_ms = 0.0
        self.max_input_age_ms = 0.0
        self.reader_fps = None

    def observe_input(self, sample):
        if sample is None:
            return
        stamp = sample.get('timestamp_ns')
        if stamp is not None and stamp != self.last_stamp:
            self.input_changes += 1
            self.last_stamp = stamp
        self.max_input_age_ms = max(self.max_input_age_ms,
            max(0.0, self.clock()-sample['timestamp_monotonic'])*1000)
        self.reader_fps = float(sample.get('fps', 0.0))

    def observe_publish(self):
        now = self.clock()
        if self.last_publish is not None:
            self.max_publish_gap_ms = max(self.max_publish_gap_ms,
                (now-self.last_publish)*1000)
        self.last_publish = now
        self.published += 1

    def observe_step(self, duration_ms):
        self.iterations += 1
        self.max_step_ms = max(self.max_step_ms, duration_ms)
        now = self.clock()
        elapsed = now-self.started
        if elapsed < 5.0:
            return
        # Include an unfinished gap, even if no new packet has arrived yet.
        gap = max(self.max_publish_gap_ms,
            (now-self.last_publish)*1000 if self.last_publish is not None else elapsed*1000)
        self.report('[PoseTiming] '+json.dumps(dict(
            window_s=round(elapsed, 3), publish_hz=round(self.published/elapsed, 1),
            observed_input_hz=round(self.input_changes/elapsed, 1),
            reader_timestamp_fps=None if self.reader_fps is None else round(self.reader_fps, 1),
            max_publish_gap_ms=round(gap, 1), max_input_age_ms=round(self.max_input_age_ms, 1),
            max_compute_ms=round(self.max_compute_ms, 1), max_step_ms=round(self.max_step_ms, 1),
            exceeds_collection_pose_age=gap > 100.0,
            input_clock_scope='SDK reader samples; not independently verified body timestamps'),
            ensure_ascii=False), flush=True)
        # Keep continuity across reporting windows; mode changes call reset().
        last_publish, last_stamp = self.last_publish, self.last_stamp
        self.reset()
        self.last_publish, self.last_stamp = last_publish, last_stamp


root = Path(__file__).resolve().parents[1]
config = json.loads((root/'config/local.json').read_text())
sonic = Path(config['sonic_root'])
os.chdir(sonic)
path = sonic/'gear_sonic/scripts/pico_manager_thread_server.py'
spec = importlib.util.spec_from_file_location('chenyz_pico_manager', path)
pico = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pico
spec.loader.exec_module(pico)
pose_timing = PoseTimingDiagnostics()
original_compute_from_body_poses = pico.compute_from_body_poses
original_pose_run_once = pico.PoseStreamer.run_once
original_pose_reset_yaw = pico.PoseStreamer.reset_yaw


def timed_compute_from_body_poses(*args, **kwargs):
    started = time.monotonic()
    try:
        return original_compute_from_body_poses(*args, **kwargs)
    finally:
        pose_timing.max_compute_ms = max(pose_timing.max_compute_ms,
            (time.monotonic()-started)*1000)


def timed_pose_run_once(streamer):
    pose_timing.observe_input(streamer.reader.get_latest())
    started = time.monotonic()
    try:
        return original_pose_run_once(streamer)
    finally:
        pose_timing.observe_step((time.monotonic()-started)*1000)


def timed_pose_reset_yaw(streamer):
    pose_timing.reset()
    return original_pose_reset_yaw(streamer)


pico.compute_from_body_poses = timed_compute_from_body_poses
pico.PoseStreamer.run_once = timed_pose_run_once
pico.PoseStreamer.reset_yaw = timed_pose_reset_yaw
original_input_source = pico._init_input_source
original_pack_pose_message = pico.pack_pose_message
manager_state_limiter = ManagerStateRateLimiter()
active_reader = None
sdk_initialized = False
hand_hardware = json.loads((root/'config/hand-hardware.json').read_text())
brainco_status_fields = None
if hand_hardware.get('brand') == 'BrainCo' and hand_hardware.get('model') == 'Revo2':
    hand_controller, brainco_status_fields = install_brainco_hands(pico, lambda: active_reader,
        time.monotonic, grasp_config=hand_hardware.get('grasp_input'))
    write_runtime_manifest(root/'logs/pico-hand-runtime.json','pico_manager',
        hand_controller.grasp_config,session=hand_controller.runtime_session,
        trigger_rate_per_s=hand_controller.rate)
else:
    hand_controller = install_continuous_hands(pico, lambda: active_reader)
controller_buttons_abxy = (False, False, False, False)
buttons_armed = False
last_reported_buttons = None
last_hand_report = None
original_get_abxy_buttons = pico.get_abxy_buttons
original_get_controller_inputs = pico.get_controller_inputs
recording_control_snapshot = threading.local()


def observed_controller_inputs(reader=None):
    values = original_get_controller_inputs(reader)
    recording_control_snapshot.left_grip = float(values[3])
    recording_control_snapshot.grip_monotonic = time.monotonic()
    return values


pico.get_controller_inputs = observed_controller_inputs


def observed_abxy_buttons(reader=None):
    global controller_buttons_abxy, buttons_armed, last_reported_buttons
    sampled_buttons = tuple(bool(v) for v in original_get_abxy_buttons(reader))
    controller_buttons_abxy = sampled_buttons
    recording_control_snapshot.buttons = sampled_buttons
    recording_control_snapshot.buttons_monotonic = time.monotonic()
    if controller_buttons_abxy != last_reported_buttons:
        print('[Controller] A/B/X/Y: '+str(tuple(int(v) for v in controller_buttons_abxy)), flush=True)
        last_reported_buttons = controller_buttons_abxy
    # A restart must not interpret an already-held chord as a new press.
    if not buttons_armed:
        buttons_armed = not any(controller_buttons_abxy)
        return False, False, False, False
    return controller_buttons_abxy


pico.get_abxy_buttons = observed_abxy_buttons


def paced_pack_pose_message(pose_data, topic='pose', version=3):
    global last_hand_report
    if topic == 'pose':
        pose_timing.observe_publish()
    # OFF, duplicate timestamps and disconnects bypass the streamers' sleeps.
    # Bound every manager iteration; a paced 20 Hz planner adds no extra wait.
    if topic == 'manager_state':
        # Use this publishing thread's manager-loop inputs, before hand status
        # performs any later controller reads. Pose workers cannot overwrite them.
        recording_buttons = getattr(recording_control_snapshot,'buttons',(False,)*4)
        recording_left_grip = getattr(recording_control_snapshot,'left_grip',0.0)
        recording_stamp = min(getattr(recording_control_snapshot,'buttons_monotonic',0.0),
                              getattr(recording_control_snapshot,'grip_monotonic',0.0))
        stream_mode = pose_data.get('stream_mode')
        manager_state_limiter.wait()
        pose_data = dict(pose_data)
        if brainco_status_fields is not None:
            hand_status=brainco_status_fields(int(stream_mode[0]))
            pose_data.update(hand_status)
            report=(bool(hand_status['brainco_hand_enabled'][0]),
                    tuple(bool(v) for v in hand_status['brainco_stick_clicks']),
                    bool(hand_status['brainco_input_valid'][0]),
                    tuple(round(float(v),1) for v in hand_status['brainco_triggers']),
                    tuple(int(v) for v in hand_status['brainco_grasp_profile']))
            if report!=last_hand_report:
                print('[Hands] '+('ENABLED' if report[0] else 'DISABLED')+
                    '; stick clicks='+str(report[1])+'; input valid='+str(report[2])+
                    '; triggers='+str(hand_status['brainco_triggers'].tolist())+
                    '; grasp L/R='+str(report[4])+' (0=tripod, 1=power)'+
                    '; mapping v'+str(hand_controller.mapping_revision),flush=True)
                last_hand_report=report
            for side, status in zip(('left','right'), hand_controller.selection_status):
                if status == 'release_trigger_and_wait_for_open':
                    print('[Grasp] '+side+': 松开扳机，等待张开目标归零后再按摇杆切换抓姿。',flush=True)
        pose_data['controller_buttons_abxy'] = pico.np.asarray(recording_buttons, dtype=pico.np.uint8)
        # The collector decides recording gestures only after button release.
        # Keep raw inputs separate from the upstream edges shared with mode keys.
        pose_data['recording_left_grip'] = pico.np.asarray([recording_left_grip], dtype=pico.np.float32)
        pose_data['recording_controls_monotonic'] = pico.np.asarray([recording_stamp], dtype=pico.np.float64)
        pose_data['recording_controls_protocol'] = pico.np.asarray([2], dtype=pico.np.int32)
        sample = active_reader.get_latest() if active_reader is not None else None
        age_ms = -1.0 if sample is None else max(0.0, time.monotonic()-sample['timestamp_monotonic'])*1000
        pose_data['pico_input_age_ms'] = pico.np.asarray([age_ms], dtype=pico.np.float32)
    return original_pack_pose_message(pose_data, topic=topic, version=version)


pico.pack_pose_message = paced_pack_pose_message

def portable_input_source(input_source, buffer_size):
    global active_reader, sdk_initialized
    if input_source != 'xrt':
        active_reader = original_input_source(input_source, buffer_size)
        return active_reader
    try:
        with socket.create_connection(('127.0.0.1', 60061), timeout=1):
            pass
    except OSError:
        subprocess.Popen([sys.executable, str(root/'run.py'), 'pc-service'], start_new_session=True)
        deadline = time.monotonic()+15
        while True:
            try:
                with socket.create_connection(('127.0.0.1', 60061), timeout=.5):
                    break
            except OSError:
                if time.monotonic()>=deadline:
                    raise RuntimeError('XRoboToolkit PC service did not start')
                time.sleep(.25)
    if pico.xrt is None:
        raise ImportError('XRoboToolkit SDK is unavailable')
    pico.xrt.init()
    sdk_initialized = True
    print('PC service connected. Waiting for PICO full-body tracking; policy starts OFF.', flush=True)
    while not pico.xrt.is_body_data_available():
        print('Waiting for PICO full-body: timestamp_ns='+str(pico.xrt.get_time_stamp_ns())+
              ', head_pose='+str(pico.xrt.get_headset_pose())+
              '; PC IP '+config['pc_wifi_ip']+'; enable Fullbody and Send.', flush=True)
        time.sleep(2)
    reader = pico.input_readers.PicoReader(max_queue_size=buffer_size)
    reader.start()
    active_reader = reader
    print('SONIC arm policy is unchanged. B+Y selects joystick walking with head/hand tracking. BrainCo triggers directly control fingers; release and wait for open, then click the matching stick to select tripod/power (default power). OFF/initialization sends no finger commands.'
          if brainco_status_fields is not None else
          'Dex3: left/right triggers continuously close matching hands; release opens. Joystick locomotion is unchanged.', flush=True)
    return reader

pico._init_input_source = portable_input_source
parser = argparse.ArgumentParser(description='SONIC PICO manager using the portable PC service')
parser.add_argument('--port', type=int, default=5556)
parser.add_argument('--target-fps', type=int, default=50)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--vis-vr3pt', action='store_true')
parser.add_argument('--vis-smpl', action='store_true')
args = parser.parse_args()
try:
    pico.run_pico_manager(port=args.port, target_fps=args.target_fps, use_cuda=args.cuda,
                          enable_vis_vr3pt=args.vis_vr3pt, enable_smpl_vis=args.vis_smpl)
finally:
    # PicoReader.stop only joins the Python reader. Release the native SDK's
    # stream/watch threads too, including exit while waiting for full-body data.
    if sdk_initialized:
        if active_reader is not None:active_reader.stop()
        pico.xrt.close()
