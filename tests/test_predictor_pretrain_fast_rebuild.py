import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from psi.data.predictor_pretrain.registry import manifest_sha256, write_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def _episode_fixture(tmp_path):
    data = tmp_path / "episode.parquet"
    pq.write_table(pa.table({"timestamp": np.arange(91, dtype=np.float64) / 30}), data)
    episode = {
        "format_version": "predictor_pretrain_episode_v3",
        "source_id": "robot", "revision": "fixed", "episode_id": "episode-0",
        "group_id": "capture-0", "sample_id": "episode-0", "split": "PT_train",
        "lineage_ids": ["robot:episode-0"], "content_hashes": [], "paths": [str(data)],
        "data_path": str(data), "video_path": str(tmp_path / "head.mp4"),
        "nominal_fps": 30, "primary_image_key": "head", "target_image_key": "head",
        "input_image_keys": ["head"],
        "input_views": [{
            "image_key": "head", "role": "primary", "available": True,
            "video_path": str(tmp_path / "head.mp4"), "video_from_timestamp": 0.0,
        }],
    }
    episodes = tmp_path / "episodes.jsonl"
    write_manifest_rows(episodes, [episode])
    freeze = tmp_path / "episode_freeze.json"
    freeze.write_text(json.dumps({
        "format_version": "predictor_pretrain_dataset_freeze_v3",
        "input_protocol": "v3", "manifest_sha256": manifest_sha256(episodes),
        "num_rows": 1, "split_seed": 101, "train_fraction": 0.9,
        "val_fraction": 0.05,
    }))
    return episodes, freeze


def test_fast_materialization_writes_final_frozen_manifest_once(tmp_path):
    episodes, episode_freeze = _episode_fixture(tmp_path)
    manifest = tmp_path / "anchors.jsonl"
    freeze = tmp_path / "freeze.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/data/materialize_pretrain_anchors.py"),
        "--episode-manifest", str(episodes), "--episode-freeze", str(episode_freeze),
        "--output", str(manifest), "--dataset-freeze", str(freeze),
        "--protocol-version", "v3", "--progress-every-seconds", "1",
    ], env={"PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    frozen = json.loads(freeze.read_text())
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert rows and {row["split"] for row in rows} == {"PT_train"}
    assert frozen["num_rows"] == len(rows)
    assert frozen["manifest_sha256"] == manifest_sha256(manifest)
    assert frozen["episode_manifest_sha256"] == manifest_sha256(episodes)


def test_index_and_overlap_audit_share_one_scan(tmp_path):
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, [{
        "format_version": "predictor_pretrain_anchor_v3",
        "sample_id": "pt-0", "episode_id": "pt-episode", "group_id": "pt-group",
        "source_id": "robot", "revision": "fixed", "split": "PT_train",
        "lineage_ids": ["pt-episode"], "content_hashes": [], "paths": [],
        "visual_history_seconds": 0.0, "future_seconds": 0.6, "action_steps": 30,
    }])
    downstream = tmp_path / "downstream.jsonl"
    write_manifest_rows(downstream, [{
        "sample_id": "ft-0", "episode_id": "ft-episode", "group_id": "ft-group",
        "source_id": "HumanoidArena", "revision": "fixed", "split": "FT_train",
        "lineage_ids": ["ft-episode"], "content_hashes": [], "paths": [],
    }])
    index = tmp_path / "anchors.sqlite"
    audit = tmp_path / "audit.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/data/build_pretrain_anchor_index.py"),
        "--anchor-manifest", str(manifest), "--output", str(index),
        "--audit-downstream-exclusion-manifest", str(downstream),
        "--audit-output", str(audit), "--progress-every-rows", "1",
    ], env={"PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(audit.read_text())
    with sqlite3.connect(index) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert report["passed"]
    assert report["pt_manifest_sha256"] == metadata["manifest_sha256"] == manifest_sha256(manifest)
    summary = json.loads(metadata["summary_json"])
    assert summary["protocols"] == ["predictor_pretrain_anchor_v3"]
    assert summary["split_counts"] == {"robot": {"PT_train": 1}}


def test_v3_sample_stream_reuses_verified_index_summary(tmp_path):
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, [{
        "format_version": "predictor_pretrain_anchor_v3",
        "sample_id": "pt-0", "episode_id": "pt-episode", "group_id": "pt-group",
        "source_id": "robot", "revision": "fixed", "split": "PT_train",
        "lineage_ids": [], "content_hashes": [], "paths": [],
        "visual_history_seconds": 0.0, "future_seconds": 0.6, "action_steps": 30,
    }])
    identity = manifest_sha256(manifest)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": identity}))
    index = ROOT / "scripts/data/build_pretrain_anchor_index.py"
    index_path = tmp_path / "anchors.sqlite"
    subprocess.run([
        sys.executable, str(index), "--anchor-manifest", str(manifest),
        "--output", str(index_path),
    ], env={"PYTHONPATH": str(ROOT / "src")}, check=True)
    manifest.unlink()  # The fast path must not reopen the JSONL.
    stream = tmp_path / "stream.jsonl"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/data/build_pretrain_sample_stream.py"),
        "--pretrain-manifest", str(manifest), "--dataset-freeze", str(freeze),
        "--anchor-index", str(index_path), "--output", str(stream),
        "--source-weight", "robot=1", "--updates", "2", "--global-batch-size", "4",
        "--sampling-version", "v3",
    ], env={"PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "sample_stream_manifest_reused_from_index" in result.stdout
    assert json.loads(stream.read_text())["dataset_manifest_sha256"] == identity


def test_hiw_wrapper_selects_fast_rebuild_without_touching_generic_pipeline():
    wrapper = (ROOT / "scripts/data/rebuild_he_psi0_hiw_v3_manifests.sh").read_text()
    fast = (ROOT / "scripts/data/rebuild_he_psi0_hiw_v3_fast.sh").read_text()
    assert 'rebuild_he_psi0_hiw_v3_fast.sh' in wrapper
    assert '--dataset-freeze "$freeze"' in fast
    assert 'mode=fused_single_scan' in fast
    assert 'mode=parallel' in fast
    assert '--trust-frozen-index' in fast
    assert 'rebuild_complete.json' in fast


def test_fast_rebuild_end_to_end_and_second_run_is_constant_time(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 30,
        "data_path": "episode_{episode_index}.parquet",
        "video_path": "{video_key}_{episode_index}.mp4",
        "features": {
            "head": {"dtype": "video"},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }))
    (source / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["move"]}) + "\n"
    )
    timestamps = np.arange(91, dtype=np.float64) / 30
    pq.write_table(pa.table({
        "timestamp": timestamps,
        "observation.state": [[float(value), float(value + 1)] for value in timestamps],
        "action": [[float(value + 2), float(value + 3)] for value in timestamps],
    }), source / "episode_0.parquet")
    (source / "head_0.mp4").touch()

    source_spec = tmp_path / "sources.json"
    source_spec.write_text(json.dumps({"sources": [{
        "source_id": "robot", "root": str(source), "revision": "fixed",
        "primary_image_key": "head", "input_image_keys": ["head"],
        "camera_roles": {"head": "primary"}, "sampling_weight": 1,
    }]}))
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"robot": {
        "state_schema": "robot_state2", "state_columns": ["observation.state"],
        "action_columns": ["action"], "action_protocol": "robot_action2",
        "image_crop": "none", "video_backend": "opencv",
    }}))
    action_protocols = tmp_path / "actions.json"
    action_protocols.write_text(json.dumps({
        "horizon_seconds": 0.6, "action_steps": 30,
        "protocols": {"robot_action2": {"dim": 2}},
    }))
    ft_root = tmp_path / "ft"
    (ft_root / "task/version/meta").mkdir(parents=True)
    (ft_root / "task/version/meta/info.json").write_text(json.dumps({"total_episodes": 1}))
    artifact = tmp_path / "artifacts"
    command = ["bash", str(ROOT / "scripts/data/rebuild_he_psi0_hiw_v3_fast.sh")]
    environment = {
        "PATH": "/usr/bin:/bin", "PYTHON_BIN": sys.executable,
        "ARTIFACT_ROOT": str(artifact), "FT_ROOT": str(ft_root),
        "SOURCE_SPEC": str(source_spec), "RUNTIME_SPEC": str(runtime),
        "ACTION_PROTOCOLS_SPEC": str(action_protocols), "PROGRESS_EVERY_SECONDS": "1",
        "PROGRESS_EVERY_ROWS": "1",
    }
    first = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert first.returncode == 0, first.stderr + first.stdout
    assert "mode=fused_single_scan" in first.stdout
    assert "state_and_action_statistics status=passed" in first.stdout
    frozen = json.loads((artifact / "dataset_freeze.json").read_text())
    assert frozen["manifest_sha256"] == manifest_sha256(artifact / "pretrain_split_manifest.jsonl")
    second = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert second.returncode == 0, second.stderr + second.stdout
    assert "fast_rebuild status=reused" in second.stdout
