#!/usr/bin/env python3
"""Audit every recorded real-SONIC row and its independently timed sensors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from psi.data.predictive_world.real_sonic_post_training import inspect_dataset


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), min=float(values.min()),
                p50=float(np.quantile(values, .5)), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)), max=float(values.max())) if len(values) else dict(count=0)


def decode_video(item):
    import av
    path, expected = item
    count, errors = 0, []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (640, 480):
                errors.append(f"geometry at frame {count}")
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / 30) > 1e-5:
                errors.append(f"storage PTS at frame {count}")
            count += 1
    if count != expected:
        errors.append(f"frame count {count} != {expected}")
    return dict(path=path, frames=count, errors=errors[:10], error_count=len(errors))


def audit(root, *, decode_videos=False, workers=4):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info, instruction, segments, release, _ = inspect_dataset(root, task="real/place_doll")
    samples = defaultdict(list)
    counters = Counter()
    exclusions = Counter()
    examples = []
    episodes = []
    videos = {}
    meta = {int(row["episode_index"]): row for p in sorted((root / "meta/episodes").rglob("*.parquet"))
            for row in pq.read_table(p).to_pylist()}
    for record in release["episodes"]:
        ep = record["episode"]
        quality = json.loads((root / f"meta/chenyz_episode_{ep:06d}.json").read_text())
        timing = [json.loads(line) for line in (root / quality["timing_file"]).read_text().splitlines() if line]
        table = pq.read_table(record["parquet"], columns=["frame_index", "observation.sample_time",
            "observation.sync_offset", "observation.camera_repeated", "observation.state",
            "observation.hand_q", "action.motion_token", "action.hand_q"])
        decision = np.asarray([row["decision_monotonic"] for row in timing])
        cameras = sorted(timing[0]["cameras"])
        camera_times = np.asarray([[row["cameras"][key]["received_monotonic"] for key in cameras] for row in timing])
        body = np.asarray([row["body"]["received_monotonic"] for row in timing])
        hand = np.asarray([row["hand_state"]["received_monotonic"] for row in timing])
        body_action = np.asarray([row["action"]["body_received_monotonic"] for row in timing])
        hand_action = np.asarray([row["action"]["hand_received_monotonic"] for row in timing])
        anchor = np.asarray([row["anchor_monotonic"] for row in timing])
        expected_offset = np.asarray([[*(row["cameras"][key]["offset_ms"] for key in cameras),
            row["body"]["offset_ms"], row["hand_state"]["offset_ms"]] for row in timing]) / 1000
        actual_offset = np.asarray(table["observation.sync_offset"].to_pylist())
        # Camera names in sync_offset have the producer's fixed ego,left,right order.
        if cameras != ["ego_view", "left_wrist", "right_wrist"]:
            raise ValueError("unexpected camera layout")
        bad_offset = np.any(np.abs(actual_offset - expected_offset) > 1e-7, axis=1)
        counters["parquet_sidecar_offset_mismatches"] += int(bad_offset.sum())
        for key, times in [("body", body), ("hand", hand), ("body_action", body_action), ("hand_action", hand_action)]:
            ages = (decision - times) * 1000
            samples[key + "_age_ms"].extend(ages.tolist())
            counters[key + "_future_rows"] += int((ages < -1e-6).sum())
        combined_age = (decision - np.maximum(body, hand)) * 1000
        samples["combined_state_age_ms"].extend(combined_age.tolist())
        counters["current_state_older_than_training_40ms"] += int((combined_age > 40 + 1e-6).sum())
        counters["current_state_component_older_than_40ms"] += int(((decision - np.minimum(body, hand)) * 1000 > 40 + 1e-6).sum())
        skew = np.ptp(camera_times, axis=1) * 1000
        samples["camera_receive_skew_ms"].extend(skew.tolist())
        counters["camera_skew_over_recording_25ms"] += int((skew > 25 + 1e-6).sum())
        samples["body_hand_receive_skew_ms"].extend((np.abs(body-hand)*1000).tolist())
        samples["body_action_hand_action_skew_ms"].extend((np.abs(body_action-hand_action)*1000).tolist())
        repeated_flags = np.asarray(table["observation.camera_repeated"].to_pylist())
        expected_repeated = np.asarray([[int(bool(row["cameras"][key]["repeated"])) for key in cameras] for row in timing])
        counters["parquet_sidecar_repeat_mismatches"] += int(np.any(repeated_flags != expected_repeated, axis=1).sum())
        for i, key in enumerate(cameras):
            age = (decision - camera_times[:, i]) * 1000
            samples[key + "_age_ms"].extend(age.tolist())
            counters[key + "_future_rows"] += int((age < -1e-6).sum())
            counters[key + "_older_than_training_80ms"] += int((age > 80 + 1e-6).sum())
            counters[key + "_reported_repeated_rows"] += int(expected_repeated[:, i].sum())
            ids = [row["cameras"][key]["frame_id"] for row in timing]
            counters[key + "_duplicate_frame_id_rows"] += len(ids) - len(set(ids))
        for key in ["observation.hand_q", "action.hand_q"]:
            values = np.asarray(table[key].to_pylist())
            counters[key + "_out_of_0_1_rows"] += int(np.any((values < 0) | (values > 1), axis=1).sum())
        token = np.asarray(table["action.motion_token"].to_pylist())
        counters["motion_token_out_of_fsq_bounds_rows"] += int(np.any(np.abs(token) > .625 + 1e-6, axis=1).sum())
        counters["motion_token_off_fsq_grid_rows"] += int(np.any(np.abs(token / .0625 - np.rint(token / .0625)) > 1e-5, axis=1).sum())
        intervals = np.diff(decision)
        samples["sample_interval_ms"].extend((intervals * 1000).tolist())
        counters["sample_intervals_over_1_75_periods"] += int((intervals > 1.75 / 30).sum())
        drift = np.abs(decision - decision[0] - np.arange(len(decision))/30)
        samples["nominal_storage_clock_drift_ms"].extend((drift * 1000).tolist())
        counters["frames"] += len(timing)
        counters["episodes_with_gaps"] += bool(quality.get("recording_gaps"))
        counters["episodes_with_unknown_success"] += quality.get("task_success") is None
        counters["episodes_marked_training_ineligible_by_recorder"] += not bool(quality.get("training_eligible"))
        exclusions.update(quality.get("quality_exclusions", []))
        counters["runtime_sample_mismatches"] += len(quality.get("runtime_sample_mismatches", []))
        for path in record["video_paths"].values():
            if path in videos:
                raise ValueError("audit currently requires per-episode video files")
            videos[path] = len(timing)
        episodes.append(dict(episode=ep, frames=len(timing), max_camera_skew_ms=float(skew.max()),
            max_storage_drift_ms=float(drift.max()*1000), segments=record["contiguous_segments"],
            task_success=quality.get("task_success"), recorder_training_eligible=quality.get("training_eligible")))
        if len(examples) < 3:
            examples.append(dict(episode=ep, frame=0, camera_age_ms=((decision[0]-camera_times[0])*1000).tolist(),
                body_age_ms=float((decision[0]-body[0])*1000), hand_age_ms=float((decision[0]-hand[0])*1000),
                camera_skew_ms=float(skew[0])))
    video_results = []
    if decode_videos:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(decode_video, sorted(videos.items())), 1):
                video_results.append(result)
                if i % 12 == 0:
                    print(f"decoded {i}/{len(videos)} videos", flush=True)
    return dict(format="zeta0.real_sonic_alignment_audit.v1", dataset=str(root),
        instruction=instruction, frames=counters["frames"], episodes=len(episodes),
        contiguous_segments=len(segments), structural_numeric_audit="passed",
        hardware_exposure_synchronized=False, counters=dict(counters),
        metrics={key: summarize(value) for key, value in samples.items()},
        quality_exclusion_episode_counts=dict(exclusions), episodes_detail=episodes, examples=examples,
        video_decode=dict(enabled=decode_videos, files=len(video_results),
            decoded_frames=sum(v["frames"] for v in video_results),
            errors=[v for v in video_results if v["error_count"]]),
        limitations=["Frame addresses and receive-time sidecars can be audited; simultaneous physical exposures cannot be established.",
            "Stored values cannot be independently matched back to unavailable original raw sensor packets.",
            "Recorder eligibility uses fixed-fps rules; the training adapter instead splits gaps and masks unavailable physical-time targets."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    report = audit(args.dataset, decode_videos=args.decode_videos, workers=args.workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(path), "frames": report["frames"], "episodes": report["episodes"],
        "counters": report["counters"], "video_decode": report["video_decode"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
