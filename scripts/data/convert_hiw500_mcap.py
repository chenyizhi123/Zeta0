#!/usr/bin/env python3
"""Restartable HIW MCAP -> LeRobot 2.1 conversion; never modifies raw inputs.

Head acquisition time is the row clock. CFR video position is stored separately
as video_timestamp: missing camera frames must not compress physical time.
Wrist/state/action samples are causal, freshness checked, and their original
nanosecond timestamps are retained. Completed episodes are immutable; a release
is published only after all workers finish, with contiguous LeRobot indices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "hiw_mcap_rgb3_causal_v1"
CAMERAS = ("head", "left_wrist", "right_wrist")
IMAGE_TOPICS = {f"/camera/{key}/image/compressed": key for key in CAMERAS}
STATE_TOPIC = "/stamped/lowstate"
ACTION_TOPIC = "/wbc_lerobot"
GRIPPERS = ("left_trigger", "left_squeeze", "right_trigger", "right_squeeze")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def causal_indices(timestamps, queries, max_age_ns):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("source acquisition timestamps must be strictly increasing")
    indices = np.searchsorted(timestamps, queries, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("query precedes first causal sample")
    ages = queries - timestamps[indices]
    if np.any(ages > max_age_ns):
        raise ValueError(f"stale stream: maximum age {ages.max()/1e6:.3f} ms > {max_age_ns/1e6:g} ms")
    return indices, ages


def wbc_vectors(data):
    pivot = list(data["pivot"])
    state = list(data["ee_state"])
    action = list(data["ee_action"])
    grippers = [data["gripper_controls"][name] for name in GRIPPERS]
    if (len(pivot), len(state), len(action)) != (7, 12, 12):
        raise ValueError("unsupported WBC action/state dimensions")
    result = np.asarray([pivot + state + grippers, pivot + action + grippers], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite WBC action/state")
    return result[0], result[1]


def read_streams(path):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    streams = {key: [] for key in (*CAMERAS, "state", "wbc")}
    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        required = set(IMAGE_TOPICS) | {STATE_TOPIC, ACTION_TOPIC}
        if summary is None or not required <= {c.topic for c in summary.channels.values()}:
            raise ValueError("missing MCAP summary or required RGB/state/action channels")
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=sorted(required)):
            if channel.topic == ACTION_TOPIC:
                # WBC String has no acquisition header; receipt time is the
                # explicit clock, never inferred to be earlier than receipt.
                streams["wbc"].append((message.log_time, wbc_vectors(json.loads(decoded.data))))
                continue
            stamp = decoded.header.stamp
            timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if timestamp <= 0:
                raise ValueError("missing acquisition timestamp")
            if abs(timestamp - message.log_time) > 5_000_000_000:
                raise ValueError("acquisition and MCAP clocks disagree by over 5 seconds")
            if channel.topic in IMAGE_TOPICS:
                if "jpeg" not in decoded.format.lower() and "jpg" not in decoded.format.lower():
                    raise ValueError(f"unsupported image compression {decoded.format!r}")
                streams[IMAGE_TOPICS[channel.topic]].append((timestamp, bytes(decoded.data)))
            else:
                values = np.asarray([motor.q for motor in decoded.data.motor_state[:29]], dtype=np.float32)
                if values.shape != (29,) or not np.isfinite(values).all():
                    raise ValueError("invalid 29-joint observation")
                streams["state"].append((timestamp, values))
    for key, values in streams.items():
        if not values:
            raise ValueError(f"empty {key} stream")
        # Transport ordering need not equal acquisition ordering. Ambiguous
        # duplicate acquisition timestamps are rejected, not silently dropped.
        values.sort(key=lambda item: item[0])
        if np.any(np.diff([item[0] for item in values]) <= 0):
            raise ValueError(f"duplicate {key} acquisition timestamps")
    return streams


def encode_video(path, samples, indices, *, crf, threads, codec="h264"):
    import cv2
    first = cv2.imdecode(np.frombuffer(samples[int(indices[0])][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("invalid JPEG")
    height, width = first.shape[:2]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_threads", "1",
               "-threads", "1", "-f", "image2pipe", "-framerate", "30", "-vcodec", "mjpeg",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(crf), "-threads", str(threads), "-pix_fmt", "yuv420p",
               "-g", "30", "-bf", "0", "-movflags", "+faststart", "-n", str(path)]
    if codec == "mjpeg":
        command = command[:command.index("-an")] + [
            "-an", "-c:v", "copy", "-movflags", "+faststart", "-n", str(path)]
    elif codec != "h264":
        raise ValueError(f"unsupported video codec: {codec}")
    with path.with_suffix(".ffmpeg.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in indices:
                process.stdin.write(samples[int(index)][1])
            process.stdin.close()
            code = process.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited {code}: {path.with_suffix('.ffmpeg.log')}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    capture = cv2.VideoCapture(str(path))
    try:
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(indices):
            raise ValueError("encoded video frame count mismatch")
        for frame in (0, len(indices) // 2, len(indices) - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok or image.shape[:2] != (height, width):
                raise ValueError("encoded video failed seek/decode/shape validation")
    finally:
        capture.release()
    return [height, width, 3]


def convert_episode(job):
    raw, relative, output, crf, threads, image_age_ms, state_age_ms, *options = job
    codec = options[0] if options else "h264"
    raw, output = Path(raw), Path(output)
    path = raw / relative
    metadata_path = path.with_name("info.json")
    fingerprint = {"relative_path": relative, "size": path.stat().st_size,
                   "mtime_ns": path.stat().st_mtime_ns, "info_sha256": digest_file(metadata_path),
                   "version": VERSION, "crf": crf,
                   "image_age_ms": image_age_ms, "state_age_ms": state_age_ms}
    if codec != "h264":
        fingerprint["video_codec"] = codec
    identity = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    destination = output / "work" / identity
    marker = destination / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        for filename, size in record["file_sizes"].items():
            if (destination / filename).stat().st_size != size:
                raise ValueError(f"completed artifact was modified: {destination / filename}")
            if digest_file(destination / filename) != record["file_sha256"][filename]:
                raise ValueError(f"completed artifact hash mismatch: {destination / filename}")
        return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed attempt remains inspectable; retries never publish its files.
    import tempfile
    temporary = Path(tempfile.mkdtemp(prefix=identity[:16] + ".partial-", dir=destination.parent))
    started = time.monotonic()
    metadata = json.loads(metadata_path.read_text())
    streams = read_streams(path)
    clocks = {key: np.asarray([item[0] for item in values], dtype=np.int64) for key, values in streams.items()}
    start = max(values[0] for values in clocks.values())
    stop = min(values[-1] for values in clocks.values())
    head_indices = np.flatnonzero((clocks["head"] >= start) & (clocks["head"] <= stop))
    head = clocks["head"][head_indices]
    if len(head) < 20 or (head[-1] - head[0]) < 600_000_000:
        raise ValueError("insufficient common stream coverage")
    selections, ages = {}, {}
    for key in streams:
        selections[key], ages[key] = causal_indices(
            clocks[key], head, round((image_age_ms if key in CAMERAS else state_age_ms) * 1e6))
    n = len(head)
    columns = {
        "timestamp": (head - head[0]).astype(np.float64) / 1e9,
        "video_timestamp": np.arange(n, dtype=np.float64) / 30,
        "source_timestamp_ns": head,
        "observation.state": np.stack([streams["state"][int(i)][1] for i in selections["state"]]).tolist(),
        "observation.state.wbc": np.stack([streams["wbc"][int(i)][1][0] for i in selections["wbc"]]).tolist(),
        "action": np.stack([streams["wbc"][int(i)][1][1] for i in selections["wbc"]]).tolist(),
    }
    for key in streams:
        columns[f"source_timestamp_ns.{key}"] = clocks[key][selections[key]]
        columns[f"timestamp.{key}"] = (clocks[key][selections[key]] - head[0]).astype(np.float64) / 1e9
    table = pa.table(columns)
    for key in ("observation.state", "observation.state.wbc", "action"):
        dim = 29 if key == "observation.state" else 23
        table = table.set_column(table.schema.get_field_index(key), key,
                                pa.array(columns[key], type=pa.list_(pa.float32(), dim)))
    pq.write_table(table, temporary / "data.parquet", compression="zstd")
    shapes = {key: encode_video(temporary / f"{key}.mp4", streams[key], selections[key],
                                crf=crf, threads=threads, codec=codec) for key in CAMERAS}
    json_atomic(temporary / "source_info.json", metadata)
    files = ["data.parquet", "source_info.json", *[f"{key}.mp4" for key in CAMERAS]]
    record = {
        "identity": identity, "source": fingerprint, "source_sha256": digest_file(path),
        "video_codec": codec,
        "task": metadata["task"], "capture_group": str(Path(relative).parent.parent),
        "scene": metadata.get("scene"), "length": n, "shapes": shapes,
        "duration_seconds": float((head[-1] - head[0]) / 1e9),
        "head_frames_trimmed_at_boundaries": len(clocks["head"]) - n,
        "head_max_interval_ms": float(np.diff(head).max() / 1e6),
        "max_age_ms": {key: float(value.max() / 1e6) for key, value in ages.items()},
        "repeated_samples": {key: n - len(np.unique(value)) for key, value in selections.items()},
        "file_sizes": {name: (temporary / name).stat().st_size for name in files},
        "file_sha256": {name: digest_file(temporary / name) for name in files},
        "conversion_seconds": time.monotonic() - started,
    }
    if fingerprint["size"] != path.stat().st_size or fingerprint["mtime_ns"] != path.stat().st_mtime_ns:
        raise ValueError("raw source changed during conversion")
    json_atomic(temporary / "complete.json", record)
    os.rename(temporary, destination)
    return record


def publish(output, records, inventory, failures):
    records = sorted(records, key=lambda row: row["source"]["relative_path"])
    release_id = hashlib.sha256(json.dumps([r["identity"] for r in records]).encode()).hexdigest()
    release = output / "releases" / release_id
    if (release / "meta/conversion.json").exists():
        return release
    if not records:
        raise ValueError("no successful episodes to publish")
    shapes = records[0]["shapes"]
    codecs = {record.get("video_codec", "h264") for record in records}
    if len(codecs) != 1:
        raise ValueError("mixed video codecs require separate releases")
    codec = codecs.pop()
    if any(record["shapes"] != shapes for record in records):
        raise ValueError("camera shapes differ across episodes; separate dataset required")
    import tempfile
    release.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=release.parent))
    (temporary / "meta").mkdir()
    tasks = sorted({record["task"] for record in records})
    episodes, episode_stats, global_index = [], [], 0
    for episode_index, record in enumerate(records):
        episode_dir = temporary / "episodes" / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True)
        work = output / "work" / record["identity"]
        table = pq.read_table(work / "data.parquet")
        n = table.num_rows
        for key, values in {
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index+n, dtype=np.int64),
            "task_index": np.full(n, tasks.index(record["task"]), dtype=np.int64),
        }.items():
            table = table.append_column(key, pa.array(values))
        pq.write_table(table, episode_dir / "data.parquet", compression="zstd")
        stats = {}
        for key in ("observation.state", "observation.state.wbc", "action"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)
            stats[key] = {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
                          "mean": values.mean(0).tolist(), "std": values.std(0).tolist(), "count": [n]}
        episode_stats.append({"episode_index": episode_index, "stats": stats})
        (episode_dir / "source_info.json").symlink_to(work / "source_info.json")
        for key in CAMERAS:
            (episode_dir / f"observation.images.{key}.mp4").symlink_to(work / f"{key}.mp4")
        episodes.append({"episode_index": episode_index, "tasks": [record["task"]], "length": n,
                         "capture_group": record["capture_group"], "scene": record["scene"],
                         "source_relative_path": record["source"]["relative_path"],
                         "source_sha256": record["source_sha256"], "robot_type": "unitree_g1"})
        global_index += n
    features = {f"observation.images.{key}": {
        "dtype": "video", "shape": shapes[key], "names": ["height", "width", "channels"],
        "video_info": {"video.fps": 30, "video.codec": codec,
                       **({"video.pix_fmt": "yuv420p"} if codec == "h264" else {})},
    } for key in CAMERAS}
    schema = pq.read_schema(temporary / "episodes/episode_000000/data.parquet")
    for field in schema:
        if pa.types.is_fixed_size_list(field.type):
            features[field.name] = {"dtype": "float32", "shape": [field.type.list_size], "names": None}
        else:
            features[field.name] = {"dtype": "int64" if pa.types.is_integer(field.type) else "float64", "shape": [1], "names": None}
    json_atomic(temporary / "meta/info.json", {
        "codebase_version": "v2.1", "robot_type": "unitree_g1", "fps": 30,
        "total_episodes": len(records), "total_frames": global_index, "total_tasks": len(tasks),
        "total_videos": 3*len(records), "total_chunks": (len(records)+999)//1000,
        "chunks_size": 1000, "splits": {"train": f"0:{len(records)}"},
        "features": features, "video_timestamp_column": "video_timestamp",
        "state_timestamp_column": "timestamp.state",
        "action_timestamp_column": "timestamp.wbc",
        "image_timestamp_columns": {f"observation.images.{key}": f"timestamp.{key}" for key in CAMERAS},
        "data_path": "episodes/episode_{episode_index:06d}/data.parquet",
        "video_path": "episodes/episode_{episode_index:06d}/{video_key}.mp4",
        "revision": release_id, "conversion_version": VERSION,
    })
    for name, rows in (("episodes", episodes), ("episodes_stats", episode_stats), ("tasks", [
        {"task_index": index, "task": task} for index, task in enumerate(tasks)])):
        with (temporary / f"meta/{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    json_atomic(temporary / "meta/conversion.json", {
        "version": VERSION, "revision": release_id, "episodes": len(records),
        "frames": global_index, "inventory": inventory, "failures": failures,
        "image_clock": "head_acquisition_ns; wrists causal <= image_age_ms",
        "video_clock": "CFR frame_index/30; use video_timestamp_column, NOT physical timestamp",
        "state_clock": "lowstate acquisition header; causal",
        "action_clock": "WBC log_time (String has no header); causal",
        "action_layout": "pivot[7] + ee_action[12] + left_trigger,left_squeeze,right_trigger,right_squeeze",
        "state_layout": "first 29 lowstate motor_state.q in Unitree G1 order",
        "dropped_modalities": ["IMU", "odometry", "infrared"],
        "source_records": records,
    })
    os.rename(temporary, release)
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--encode-threads", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--image-age-ms", type=float, default=70)
    parser.add_argument("--state-age-ms", type=float, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--episode", help="one raw-root-relative .mcap for a smoke conversion")
    args = parser.parse_args()
    if args.workers < 1 or args.encode_threads < 1 or not 0 <= args.crf <= 51:
        parser.error("invalid workers/threads/crf")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    raw, output = args.raw_root.resolve(), args.output_root.resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        parser.error("raw and output roots must be separate non-nested directories")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".conversion.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        incomplete = 0
        if args.episode:
            selected = (raw / args.episode).resolve()
            if raw not in selected.parents or selected.suffix != ".mcap" or not selected.is_file():
                parser.error("episode must identify a completed .mcap within raw-root")
            paths = [selected]
        else:
            paths = []
            for directory, _, filenames in os.walk(raw):
                for name in filenames:
                    if name.endswith(".mcap"):
                        paths.append(Path(directory) / name)
                    elif name.endswith(".incomplete"):
                        incomplete += 1
            paths.sort()
        if not paths:
            raise ValueError("no completed .mcap files found")
        inventory = {"complete_mcap_files": len(paths),
                     "incomplete_files": incomplete, "scope": "single_episode" if args.episode else "local_snapshot",
                     "selected": min(len(paths), args.limit) if args.limit else len(paths)}
        if args.limit:
            paths = paths[:args.limit]
        print(json.dumps({"inventory": inventory, "workers": args.workers}), flush=True)
        jobs = [(str(raw), str(path.relative_to(raw)), str(output), args.crf, args.encode_threads,
                 args.image_age_ms, args.state_age_ms) for path in paths]
        records, failures = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(convert_episode, job): job[1] for job in jobs}
            for future in as_completed(pending):
                source = pending[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(json.dumps({"completed": len(records), "failed": len(failures), "source": source,
                                      "frames": record["length"], "seconds": record["conversion_seconds"]}), flush=True)
                except Exception as error:
                    failure = {"source": source, "error": f"{type(error).__name__}: {error}"}
                    failures.append(failure)
                    print(json.dumps({"failed": failure}), flush=True)
                json_atomic(output / "progress.json", {"inventory": inventory, "completed": len(records),
                                                        "failed": len(failures), "failures": failures})
        release = publish(output, records, inventory, failures)
        json_atomic(output / "latest_release.json", {"root": str(release), "episodes": len(records),
                                                     "failed": len(failures), "revision": release.name})
        print(json.dumps({"release": str(release), "episodes": len(records), "failed": len(failures)}), flush=True)
        if failures:
            raise SystemExit(1)  # partial release is explicit, never a silent full-success


if __name__ == "__main__":
    main()
