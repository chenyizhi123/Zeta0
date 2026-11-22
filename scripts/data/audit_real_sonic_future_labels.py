#!/usr/bin/env python3
"""CPU audit of every retained real-SONIC future label against its actual source clock."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from psi.data.predictive_world.action_labels import validate_action_component_index
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.index import verify_predictive_world_index
from psi.real_observation_contract import snapshot_timing_violations


INPUT_FIELDS = (
    "sample_id", "state_indices", "state_valid", "state_actual_times",
    "history_visual_indices_by_camera", "history_visual_valid_by_camera", "history_visual_actual_times_by_camera",
)


def input_digest(path):
    digest = hashlib.sha256()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            digest.update(json.dumps({key: row[key] for key in INPUT_FIELDS}, sort_keys=True).encode())
    return digest.hexdigest()


def audit(config_path, *, reference_index=None):
    config = json.loads(Path(config_path).read_text())
    data = config["data"]
    if data.get("future_component_alignment") != "independent_components_v1":
        raise ValueError("audit requires the explicit independent-component training config")
    index = Path(data["index_path"])
    meta = verify_predictive_world_index(index)
    inventory = json.loads((index.parent / "dataset_audit.json").read_text())
    receipts = {int(row["episode"]): row for row in inventory["episodes"]}
    dataset = Path(inventory["root"])
    cache, cache_episode = None, None
    counts = {"all": Counter(), "joint": Counter()}
    errors, maximum_age, prefix_histogram = Counter(), Counter(), Counter()
    examples = []
    input_hash = hashlib.sha256()
    snapshots = Counter()
    held_commands = Counter()
    snapshot_mode = data.get("real_observation_mode") == "collector_snapshot_v1"
    held_mode = data.get("real_action_label_mode") == "held_command_v1"
    with index.open() as stream:
        for line in stream:
            row = json.loads(line)
            input_hash.update(json.dumps({key: row[key] for key in INPUT_FIELDS}, sort_keys=True).encode())
            validate_action_component_index(row, data["action_horizon"])
            episode = int(re.search(r"/episode_(\d+)/", row["episode_id"]).group(1))
            if episode != cache_episode:
                quality_path = dataset / f"meta/chenyz_episode_{episode:06d}.json"
                quality = json.loads(quality_path.read_text())
                timing_path = dataset / quality["timing_file"]
                receipt = receipts[episode]
                if (file_sha256(quality_path) != receipt["quality_sha256"]
                        or file_sha256(timing_path) != receipt["timing_sha256"]
                        or file_sha256(Path(row["materialization"]["data_path"])) != receipt["parquet_sha256"]):
                    raise ValueError("source rows or sidecar changed after index publication")
                timing = [json.loads(line) for line in timing_path.open() if line.strip()]
                origin = timing[0]["decision_monotonic"]
                actions = np.array([[item["action"]["body_received_monotonic"],
                    *item["action"].get("hand_receive_monotonic", [item["action"]["hand_received_monotonic"]]*2)]
                    for item in timing]) - origin
                bodies = np.array([item["body"]["received_monotonic"] for item in timing]) - origin
                decisions = np.array([item["decision_monotonic"] for item in timing]) - origin
                segments = np.asarray([item["sample_segment"] for item in timing])
                cuts = np.flatnonzero((segments[1:] != segments[:-1]) | (np.diff(decisions) > 1.75/30)) + 1
                bounds = np.r_[0, cuts, len(timing)]
                cache, cache_episode = (actions, bodies, bounds, timing, origin), episode
            actions, bodies, bounds, timing, origin = cache
            segment = int(re.search(r"/segment_(\d+)$", row["episode_id"]).group(1))
            lo, hi = bounds[segment:segment+2]
            scopes = ["all", *(["joint"] if row["eligibility"]["world"] else [])]
            for scope in scopes:
                counts[scope]["rows"] += 1
            anchor = row["anchor_time"]

            if snapshot_mode:
                selected_snapshot = row["collector_snapshot_row_index"]
                item = timing[selected_snapshot]
                hand = item["hand_state"].get("channel_received_monotonic")
                if hand is None:
                    hand = [item["hand_state"]["received_monotonic"]] * 2
                camera_times = {"observation.images." + name: value["received_monotonic"]
                                for name, value in item["cameras"].items()}
                paired = snapshot_timing_violations(
                    decision_time=origin+anchor, camera_anchor_time=item["anchor_monotonic"],
                    camera_times=camera_times, body_time=item["body"]["received_monotonic"],
                    hand_times=hand, contract=data["real_observation_contract"],
                    snapshot_max_age_s=data.get("real_snapshot_max_age_s"))
                for violation in paired:
                    errors["current_" + violation["code"]] += 1
                offsets_by_camera = row["materialization"]["collector_video_offsets"]
                mismatch = row["state_indices"] != [selected_snapshot]
                mismatch |= not lo <= selected_snapshot < hi
                for name, actual in camera_times.items():
                    mismatch |= row["history_visual_indices_by_camera"][name][-1] != selected_snapshot + offsets_by_camera[name]
                    mismatch |= not np.isclose(row["history_visual_actual_times_by_camera"][name][-1],
                                              actual-origin-anchor, atol=1e-9, rtol=0)
                actual_state_time = max(item["body"]["received_monotonic"], *hand)-origin-anchor
                mismatch |= not np.isclose(row["state_actual_times"][-1], actual_state_time, atol=1e-9, rtol=0)
                errors["current_snapshot_source_mismatch"] += int(mismatch)
                snapshots["rows"] += 1
                snapshots["delayed_snapshot_rows"] += int(item["decision_monotonic"] < origin+anchor-1e-9)
                if (paired or mismatch) and len(examples) < 10:
                    examples.append(dict(sample_id=row["sample_id"], kind="current_snapshot", failure=paired,
                                         source_mismatch=bool(mismatch)))

            def check(kind, indices, valid, actual, source, offsets, max_age, *, held=False):
                mask = np.asarray(valid, dtype=bool)
                selected = np.asarray(indices, dtype=np.int64)
                times = source[selected] - anchor
                recorded = np.asarray(actual, dtype=np.float64)
                ages = np.asarray(offsets) - times
                failures = {
                    "timestamp_mismatch": mask & ~np.isclose(times, recorded, atol=1e-9, rtol=0),
                    "before_or_at_anchor": mask & (times <= 0) & (not held),
                    "future_of_query": mask & (ages < -1e-9),
                    "stale": mask & (ages > max_age + 1e-9),
                    "cross_segment": mask & ((selected < lo) | (selected >= hi)),
                }
                for scope in scopes:
                    counts[scope][kind + "_valid_slots"] += int(mask.sum())
                if held:
                    held_commands[kind + "_source_before_or_at_anchor"] += int((mask & (times <= 0)).sum())
                if mask.any():
                    maximum_age[kind] = max(maximum_age[kind], float(ages[mask].max()) * 1000)
                for name, failed in failures.items():
                    errors[kind + "_" + name] += int(failed.sum())
                    if failed.any() and len(examples) < 10:
                        examples.append(dict(sample_id=row["sample_id"], kind=kind, failure=name,
                            slots=np.flatnonzero(failed).tolist()))

            offsets = np.arange(1, data["action_horizon"]+1) / data["action_hz"]
            components = []
            for column, name in enumerate(("motion_token", "left_hand", "right_hand")):
                valid = row["action_valid_by_component"][name]
                components.append(valid)
                check(name, row["action_indices_by_component"][name], valid,
                    row["action_actual_times_by_component"][name], actions[:, column], offsets, data["action_max_age_s"],
                    held=held_mode)
            complete = np.asarray(components).all(axis=0)
            for scope in scopes:
                counts[scope]["complete_action_slots"] += int(complete.sum())
                counts[scope]["partial_action_slots"] += int((np.asarray(components).any(axis=0) & ~complete).sum())
            if row["eligibility"]["world"]:
                leading = int(complete.astype(int).cumprod().sum())
                supported = min(config["training"]["rtc_training_max_delay"], max(0, leading-1))
                prefix_histogram[supported] += 1
            measurement_offsets = meta["temporal_abi"]["future_measurement_offsets_s"]
            check("future_body", row["future_measurement_indices"], row["future_measurement_valid"],
                row["future_measurement_actual_times"], bodies, measurement_offsets,
                data["state_max_age_s"] if data.get("future_measurement_max_age_s") is None else data["future_measurement_max_age_s"])
            if snapshot_mode:
                ego = np.asarray([item["cameras"]["ego_view"]["received_monotonic"] for item in timing]) - origin
                camera_offset = row["materialization"]["collector_video_offsets"]["observation.images.ego_view"]
                indices = np.asarray(row["future_visual_indices"]) - camera_offset
                check("future_visual", indices, row["future_visual_valid"], row["future_visual_actual_times"],
                      ego, data["future_group_ends_s"], data["future_visual_max_age_s"])
    result = dict(format="zeta0.real_sonic_future_labels_audit.v3" if snapshot_mode else "zeta0.real_sonic_future_labels_audit.v2", index_sha256=meta["index_sha256"],
        counts={key: dict(value) for key, value in counts.items()}, violations=dict(errors),
        maximum_age_ms=dict(maximum_age), joint_supported_prefix_histogram=dict(sorted(prefix_histogram.items())),
        examples=examples, current_input_digest=input_hash.hexdigest(), original_data_modified=False)
    if snapshot_mode:
        result.update(real_observation_mode=data["real_observation_mode"],
                      real_action_label_mode=data.get("real_action_label_mode"),
                      current_snapshot=dict(snapshots), held_command_counts=dict(held_commands),
                      native_capture_jitter_preserved=True)
    if reference_index:
        result["current_inputs_unchanged"] = input_digest(reference_index) == input_hash.hexdigest()
    result["passed"] = not any(errors.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-index")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("audit output exists; choose a new path")
    result = audit(args.config, reference_index=args.reference_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("future label alignment audit failed")


if __name__ == "__main__":
    main()
