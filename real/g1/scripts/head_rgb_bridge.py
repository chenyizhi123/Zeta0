#!/usr/bin/python3
"""D455 ROS Noetic RGB -> SONIC msgpack/JPEG, without depth processing."""
import json
import time
import uuid
from pathlib import Path
from threading import Lock

import cv2
import msgpack
import rospy
import zmq
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


def main():
    rospy.init_node("sonic_camera_bridge")
    serial = str(rospy.get_param("~serial"))
    if not serial or serial.startswith("REPLACE_"):
        raise ValueError("Set the verified ROS ~serial parameter before publishing")
    cv2.setNumThreads(1)
    output = Path(rospy.get_param("~output_dir", str(Path.home() / "sonic_camera/output")))
    output.mkdir(parents=True, exist_ok=True)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = rospy.get_param("~endpoint")
    socket.bind(endpoint)
    bridge = CvBridge()
    lock = Lock()
    latest = [None]
    info = {}

    def save_info(message, name):
        info[name] = {
            "width": message.width, "height": message.height,
            "frame_id": message.header.frame_id,
            "distortion_model": message.distortion_model,
            "D": list(message.D), "K": list(message.K),
            "R": list(message.R), "P": list(message.P),
        }

    def receive_color(color):
        with lock:
            latest[0] = (color, time.monotonic())

    subscribers = [
        rospy.Subscriber("/sonic_camera/color/camera_info", CameraInfo, save_info,
                         callback_args="color", queue_size=1),
        rospy.Subscriber("/sonic_camera/color/image_raw", Image, receive_color,
                         queue_size=1, buff_size=2**24),
    ]
    source_session = uuid.uuid4().hex
    count = 0
    started = time.monotonic()
    last_frame = started
    last_status = 0.0
    rospy.loginfo("SONIC D455 camera endpoint: %s", endpoint)
    try:
        while not rospy.is_shutdown():
            with lock:
                frames = latest[0]
                latest[0] = None
            if frames is None:
                if time.monotonic() - last_frame > 40:
                    raise RuntimeError("No RGB frames for 40 seconds")
                time.sleep(0.003)
                continue
            color_msg, received = frames
            if time.monotonic() - received > 0.5:
                continue
            color = bridge.imgmsg_to_cv2(color_msg, "bgr8")
            ok_color, jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok_color:
                raise RuntimeError("Camera image encoding failed")
            # Official SONIC client accepts raw JPEG bytes and converts BGR to RGB.
            stamp = color_msg.header.stamp.to_sec()
            payload = {
                "timestamps": {"ego_view": stamp},
                "images": {"ego_view": jpg.tobytes()},
                "camera_info": info.copy(),
                "serial": serial, "sequence": count,
                "source_session": source_session,
                "timestamp_kind": "robot_ros_header_stamp",
                "capture": dict(ros_header_sequence=color_msg.header.seq,
                                ros_header_stamp=stamp, callback_monotonic=received,
                                hardware_exposure_timestamp=None),
            }
            socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)
            count += 1
            last_frame = time.monotonic()
            if count == 30:
                cv2.imwrite(str(output / "sample_rgb.jpg"), color)
            if last_frame - last_status >= 5:
                status = {"serial": serial, "endpoint": endpoint,
                          "mode": "rgb_only", "width": color_msg.width,
                          "height": color_msg.height,
                          "frames": count, "uptime_s": last_frame-started,
                          "fps_since_start": count/(last_frame-started),
                          "last_capture_timestamp": stamp}
                (output / "status.tmp").write_text(json.dumps(status, indent=2))
                (output / "status.tmp").replace(output / "status.json")
                (output / "camera_calibration.json").write_text(json.dumps(info.copy(), indent=2))
                rospy.loginfo("Published %d RGB frames", count)
                last_status = last_frame
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
