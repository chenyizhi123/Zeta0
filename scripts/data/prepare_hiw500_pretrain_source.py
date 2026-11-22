#!/usr/bin/env python3
"""Declare a completed HIW release for Stage A, without changing any mixture."""
import argparse
import json
from pathlib import Path

from psi.data.predictor_pretrain.lerobot_source import build_lerobot_episode_registry
from psi.data.predictor_pretrain.registry import write_manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-version", choices=("v2", "v3"), default="v3")
    args = parser.parse_args()
    release = args.release.resolve()
    info = json.loads((release / "meta/info.json").read_text())
    conversion = json.loads((release / "meta/conversion.json").read_text())
    if info["revision"] != conversion["revision"]:
        raise ValueError("release identity mismatch")
    cameras = [f"observation.images.{name}" for name in ("head", "left_wrist", "right_wrist")]
    roles = dict(zip(cameras, ("primary", "left_wrist", "right_wrist")))
    episodes = build_lerobot_episode_registry(
        release, source_id="HIW-500", revision=info["revision"],
        primary_image_key=cameras[0], input_image_keys=cameras,
        camera_roles=roles, group_fields=["capture_group"], protocol_version=args.protocol_version,
        action_timestamp_column="timestamp.wbc" if args.protocol_version == "v3" else None,
    )
    if len(episodes) != info["total_episodes"]:
        raise ValueError("episode coverage mismatch")
    for row in episodes:
        if not all(view["available"] for view in row["input_views"]):
            raise ValueError("HIW release must contain all three RGB views")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_rows(args.output_dir / "episodes.jsonl", episodes)
    source = {
        "source_id": "HIW-500", "root": str(release), "revision": info["revision"],
        "primary_image_key": cameras[0], "input_image_keys": cameras, "camera_roles": roles,
        "group_fields": ["capture_group"], "sampling_weight": 1,
        **({"action_timestamp_column": "timestamp.wbc"} if args.protocol_version == "v3" else {}),
    }
    spec = {
        "dataset_id": "hiw500_mcap_rgb3_v1", "protocol_version": args.protocol_version,
        "sampling_version": args.protocol_version,
        "candidate_phases": 1, "anchor_hz_per_phase": 2, "visual_history_seconds": 0,
        "future_seconds": 0.6, "sources": [source],
        "note": "Standalone source declaration, not HE/Psi0 mixture weights.",
    }
    runtime_entry = {
        "image_crop": "none", "video_backend": "opencv",
        "state_columns": ["observation.state", "observation.state.wbc"],
        "state_schema": "hiw_g1_q29_wbc23",
    }
    if args.protocol_version == "v3":
        runtime_entry.update({
            "action_columns": ["action"], "action_protocol": "hiw_g1_wbc23",
        })
    runtime = {"HIW-500": runtime_entry}
    action_protocols = {
        "format_version": "stage_a_action_protocols_v1", "horizon_seconds": 0.6,
        "action_steps": 30, "protocols": {"hiw_g1_wbc23": {
            "dim": 23, "representation": "mixed_pivot_and_end_effector_command",
            "coordinate_frame": "hiw_wbc_native", "units": "source_native_units",
            "components": ["pivot_7", "ee_action_12", "gripper_trigger_squeeze_4"],
            "provenance": "scripts/data/convert_hiw500_mcap.py:action_layout",
        }},
    }
    for name, value in (
        ("sources.json", spec), ("runtime.json", runtime),
        ("action_protocols.json", action_protocols),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
