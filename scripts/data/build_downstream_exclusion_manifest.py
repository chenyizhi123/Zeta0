#!/usr/bin/env python3
"""Freeze every HumanoidArena FT split and controller-derived lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from psi.data.predictor_pretrain.registry import write_manifest_rows


def build_exclusion_rows(root: str | Path, *, seed: int = 101) -> list[dict]:
    root = Path(root).resolve()
    versions: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for info_path in sorted(root.glob("*/*/meta/info.json")):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        task = info_path.parents[2].name
        versions[task].append((info_path.parents[1], int(info["total_episodes"])))
    if not versions:
        raise ValueError(f"no HumanoidArena dataset versions found below {root}")
    rows = []
    for task, task_versions in sorted(versions.items()):
        episode_count = max(count for _, count in task_versions)
        ordering = sorted(
            range(episode_count),
            key=lambda episode: hashlib.sha256(f"{seed}|{task}|{episode}".encode()).digest(),
        )
        split_by_episode = {
            episode: (
                "FT_train"
                if rank < round(0.8 * episode_count)
                else "FT_val"
                if rank < round(0.9 * episode_count)
                else "FT_test"
            )
            for rank, episode in enumerate(ordering)
        }
        for episode in range(episode_count):
            common = f"HumanoidArena_raw:{task}:episode:{episode}"
            paths = [str(path) for path, count in task_versions if episode < count]
            lineage = [common]
            lineage.extend(
                f"HumanoidArena:{task}:{path.name}:episode:{episode}"
                for path, count in task_versions
                if episode < count
            )
            rows.append(
                {
                    "sample_id": common,
                    "source_id": "HumanoidArena",
                    "revision": "humanoidarena_v3.1",
                    "episode_id": f"{task}:{episode}",
                    "group_id": f"{task}:{episode}",
                    "split": split_by_episode[episode],
                    "lineage_ids": lineage,
                    "content_hashes": [],
                    "paths": paths,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    write_manifest_rows(args.output, build_exclusion_rows(args.root, seed=args.seed))


if __name__ == "__main__":
    main()
