#!/usr/bin/env python3
"""Build HIW/HE Stage-A candidates from a pinned local LeRobot snapshot."""

from __future__ import annotations

import argparse

from psi.data.predictor_pretrain.lerobot_source import (
    build_lerobot_anchor_rows,
    build_lerobot_episode_registry,
)
from psi.data.predictor_pretrain.registry import write_manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--revision", required=True, help="resolved immutable commit SHA")
    parser.add_argument("--primary-image-key", required=True)
    parser.add_argument("--input-image-key", action="append")
    parser.add_argument("--camera-role", action="append", default=[], help="IMAGE_KEY=ROLE")
    parser.add_argument("--protocol-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--candidate-phases", type=int, default=1)
    parser.add_argument("--group-field", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument(
        "--action-timestamp-column",
        help="optional physical action clock override declared by the source",
    )
    parser.add_argument(
        "--episodes-only",
        action="store_true",
        help="freeze source episode identities before data/video files finish downloading",
    )
    args = parser.parse_args()
    episodes = build_lerobot_episode_registry(
        args.root,
        source_id=args.source_id,
        revision=args.revision,
        primary_image_key=args.primary_image_key,
        group_fields=args.group_field,
        input_image_keys=args.input_image_key,
        camera_roles=dict(value.split("=", 1) for value in args.camera_role) or None,
        protocol_version=args.protocol_version,
        action_timestamp_column=args.action_timestamp_column,
    )
    rows = (
        episodes
        if args.episodes_only
        else build_lerobot_anchor_rows(episodes, timestamp_column=args.timestamp_column,
                                      protocol_version=args.protocol_version, candidate_phases=args.candidate_phases)
    )
    write_manifest_rows(args.output, rows)


if __name__ == "__main__":
    main()
