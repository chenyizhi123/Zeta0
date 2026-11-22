"""Packed-file offsets, semantic provenance and normalization are audited together."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.index import verify_predictive_world_index
from psi.data.predictive_world.sonic_aggregate import prepare_aggregate
from psi.data.predictive_world.sonic_post_training import PROTOCOL, STATE, TASKS, fit_statistics


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def aggregate(tmp_path):
    root = tmp_path / "packed"
    info = dict(codebase_version="v3.0", fps=50, total_episodes=700, total_tasks=7,
        total_frames=1438, splits={"train": "0:700"},
        data_path="data/{file_index}.parquet", video_path="videos/{file_index}.mp4",
        features={"observation.state": {"dtype": "float32", "shape": [64], "names": [f"s{i}" for i in range(64)]},
                  "action": {"dtype": "float32", "shape": [40]}, "observation.images.front": {"dtype": "video"}})
    write(root / "meta/info.json", info)
    instructions = {task: f"Perform {task}." for task in TASKS}
    pq.write_table(pa.Table.from_pylist([dict(task_index=i, task=instructions[t]) for i, t in enumerate(TASKS)]),
                   root / "meta/tasks.parquet")
    rows, episodes, sources = [], [], []
    for task_id, task in enumerate(TASKS):
        original = {**info, "vla_protocol": {"schema": "unitree_g1_gmt_refpose_v3_1"}}
        source_info = tmp_path / "source_info" / (task + ".json")
        write(source_info, original)
        frames = 0
        for ep in range(task_id * 100, (task_id + 1) * 100):
            length, start = (40 if ep == 1 else 2), len(rows)
            rows.extend(dict(episode_index=ep, frame_index=i, timestamp=np.float32(i / 50), task_index=task_id,
                **{"observation.state": [float(ep + i / 100)] * 64, "action": [float(ep + i / 100)] * 40})
                for i in range(length))
            episodes.append({"episode_index": ep, "length": length, "tasks": [instructions[task]],
                "data/chunk_index": 0, "data/file_index": 0,
                "videos/observation.images.front/chunk_index": 0, "videos/observation.images.front/file_index": 0,
                "videos/observation.images.front/from_timestamp": start / 50,
                "videos/observation.images.front/to_timestamp": (start + length) / 50})
            frames += length
        sources.append(dict(task=task, episodes=100, episode_from=task_id * 100,
                            episode_to=(task_id + 1) * 100, frames=frames,
                            original_info_sha256=file_sha256(source_info)))
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "videos/0.mp4").write_bytes(b"video decoder is a separate boundary")
    (root / "meta/episodes").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), root / "data/0.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/0.parquet")
    write(tmp_path / "manifest.json", dict(native_aggregate=True, concatenate_data=False, concatenate_videos=False,
        frames=len(rows), sources=sources, sample_checks=[dict(task=t, task_text=v) for t, v in instructions.items()]))
    stats = fit_statistics(np.asarray([r["observation.state"] for r in rows], dtype=np.float32),
                           np.asarray([r["action"] for r in rows], dtype=np.float32))
    mid = "sonic_whole_body"
    canonical = dict(layout_id="g1_measured_arm14_leg12_waist3_v1", raw_indices=list(range(6, 35)),
        slots=[f"joint{i}" for i in range(29)], mean=stats["body_mean"][6:35], std=stats["body_std"][6:35],
        normalization_space="per_source_registry_zscore_v1")
    registry = dict(action_protocol_registry={PROTOCOL: stats["action"]}, state_schema_registry={STATE: stats["state"]},
        measurement_registry={mid: dict(width=64, world_action_eligible=True, canonical_body=canonical,
            fields=[dict(name="measured_joint_positions", start=6, stop=35, kind="measurement", unit="rad",
                         coordinate_frame="robot_native", time_key="timestamp", statistics_id=STATE)])},
        capability_registry={t: dict(measurement_schema_id=mid, action_protocol_id=PROTOCOL,
            supports_future_visual=True, supports_future_measurement=True, supports_world_action=True,
            visual_keys=["observation.images.front"], audit_revision="fixture") for t in TASKS})
    for name, payload in registry.items():
        write(tmp_path / "reference" / (name + ".json"), payload)
    return [root, tmp_path / "manifest.json", tmp_path / "source_info", tmp_path / "reference", tmp_path / "out"]


def test_aggregate_keeps_global_numeric_rows_and_packed_video_offsets(aggregate):
    result = prepare_aggregate(*aggregate)
    output = aggregate[-1]
    assert result["rows"] == 1438 and result["training_started"] is False
    assert verify_predictive_world_index(output / "predictive_world_index.jsonl")["rows"] == 1438
    rows = [json.loads(line) for line in (output / "predictive_world_index.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["episode_id"].endswith("000001"))
    assert row["state_indices"] == [2]
    assert row["future_visual_indices"] == [12, 22, 32]
    assert row["action_indices"][:3] == [3, 4, 5]
    assert row["materialization"]["instruction"] == f"Perform {TASKS[0]}."
    assert not (output / "feature_cache").exists()
    assert json.loads((output / "dataset_audit.json").read_text())["original_statistics_verified"]


@pytest.mark.parametrize("damage", ["task", "offset", "semantics", "statistics"])
def test_aggregate_rejects_misaligned_or_changed_source(aggregate, damage):
    root, _, source_info, reference, output = aggregate
    if damage in {"task", "offset"}:
        path = root / ("data/0.parquet" if damage == "task" else "meta/episodes/0.parquet")
        rows = pq.read_table(path).to_pylist()
        rows[0]["task_index" if damage == "task" else "videos/observation.images.front/from_timestamp"] = .01
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif damage == "semantics":
        path = source_info / (TASKS[0] + ".json")
        path.write_text(path.read_text() + " ")
    else:
        path = reference / "action_protocol_registry.json"
        values = json.loads(path.read_text())
        values[PROTOCOL]["mean"][0] += 10
        write(path, values)
    with pytest.raises((ValueError, AssertionError)):
        prepare_aggregate(*aggregate)
    assert not output.exists()
