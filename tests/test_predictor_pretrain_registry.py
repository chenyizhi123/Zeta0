import json
from pathlib import Path
import subprocess
import sys

from psi.data.predictor_pretrain.registry import (
    audit_cross_stage_overlap,
    build_grouped_split,
    canonical_sha256,
    load_manifest_rows,
    manifest_sha256,
)


def _row(sample, group, *, source="HIW-500", lineage=(), hashes=(), paths=()):
    return {
        "sample_id": sample,
        "episode_id": sample.split(":")[0],
        "group_id": group,
        "source_id": source,
        "revision": "frozen-revision",
        "lineage_ids": list(lineage),
        "content_hashes": list(hashes),
        "paths": list(paths),
        "split": "FT_train" if source == "HumanoidArena" else None,
    }


def test_grouped_split_is_stable_and_never_splits_a_group():
    rows = [_row(f"episode-{i}:0", f"session-{i // 2}") for i in range(20)]
    first = build_grouped_split(rows, seed=101)
    second = build_grouped_split(list(reversed(rows)), seed=101)
    assert first == second
    assert {r["sample_id"]: r["split"] for r in first} == {
        r["sample_id"]: r["split"] for r in second
    }
    group_splits = {}
    for row in first:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert canonical_sha256(first) == canonical_sha256(second)


def test_grouped_split_stratifies_each_source():
    rows = [
        _row(f"{source}-{index}:0", f"{source}-episode-{index}", source=source)
        for source, count in (("HE", 100), ("Psi-task", 20))
        for index in range(count)
    ]
    split = build_grouped_split(rows, seed=101, train_fraction=0.9, val_fraction=0.05)
    counts = {}
    for row in split:
        counts.setdefault(row["source_id"], {}).setdefault(row["split"], 0)
        counts[row["source_id"]][row["split"]] += 1
    assert counts["HE"] == {"PT_train": 90, "PT_val": 5, "PT_test": 5}
    assert counts["Psi-task"] == {"PT_train": 18, "PT_val": 1, "PT_test": 1}


def test_audit_rejects_moved_lineage_hash_forbidden_path_and_source(tmp_path):
    ft_root = tmp_path / "humanoidarena_v3.1"
    ft = [
        {
            **_row("ft-episode:0", "ft-session", source="HumanoidArena", lineage=["capture-42"], hashes=["abc"]),
            "split": "FT_train",
        }
    ]
    pt = build_grouped_split(
        [
            _row("renamed:0", "pt-a", lineage=["capture-42"]),
            _row("other:0", "pt-b", hashes=["abc"]),
            _row("path:0", "pt-c", paths=[str(ft_root / "derived.mp4")]),
            _row("psi:0", "pt-d", source="USC/psi-data"),
        ],
        seed=101,
        train_fraction=0.5,
        val_fraction=0.25,
    )
    report = audit_cross_stage_overlap(pt, ft, forbidden_pt_roots=[ft_root])
    assert not report.passed
    kinds = {finding.kind for finding in report.findings}
    assert {
        "ft_lineage_overlap",
        "ft_content_hash_overlap",
        "forbidden_path",
        "forbidden_source",
    } <= kinds


def test_audit_accepts_clean_external_pretraining_rows():
    pt = build_grouped_split(
        [_row(f"hiw-{i}:0", f"home-{i}") for i in range(10)], seed=101
    )
    ft = [_row("sofa-0:0", "sofa-capture", source="HumanoidArena")]
    report = audit_cross_stage_overlap(pt, ft)
    assert report.passed
    assert report.findings == []


def test_disk_backed_registry_matches_group_split_and_freeze_hash(tmp_path):
    rows = [
        {**_row(f"episode-{i}:0", f"session-{i // 2}"),
         "format_version": "predictor_pretrain_anchor_v3"}
        for i in range(20)
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    output = tmp_path / "split.jsonl"
    freeze = tmp_path / "freeze.json"
    script = Path(__file__).resolve().parents[1] / "scripts/data/build_pretrain_registry.py"
    result = subprocess.run(
        [sys.executable, str(script), str(source), "--output", str(output),
         "--dataset-freeze", str(freeze), "--seed", "101"],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = build_grouped_split(rows, seed=101)
    assert load_manifest_rows(output) == expected
    frozen = json.loads(freeze.read_text())
    assert frozen["format_version"] == "predictor_pretrain_dataset_freeze_v3"
    assert frozen["num_rows"] == len(rows)
    assert frozen["manifest_sha256"] == manifest_sha256(output)
