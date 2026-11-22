import json

from psi.data.predictor_pretrain.lerobot_source import (
    build_lerobot_episode_registry,
    resolve_lerobot_path,
)


def test_lerobot_registry_pins_revision_paths_group_and_safe_instruction(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 3, "description": "put the cup away"}) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 1002, "tasks": [3], "home": "home-7"}) + "\n"
    )
    rows = build_lerobot_episode_registry(
        tmp_path,
        source_id="HIW",
        revision="a" * 40,
        primary_image_key="observation.images.head_left",
        group_fields=["home"],
    )
    assert len(rows) == 1
    assert rows[0]["group_id"] == "HIW|home=home-7"
    assert rows[0]["instruction"] == "put the cup away"
    assert "chunk-001" in rows[0]["data_path"]
    assert "observation.images.head_left" in rows[0]["video_path"]
    assert rows[0]["lineage_ids"] == [f"HIW@{'a' * 40}:episode:1002"]


def test_lerobot_template_resolution_uses_episode_chunk(tmp_path):
    path = resolve_lerobot_path(
        tmp_path,
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        episode_index=2345,
        chunks_size=1000,
    )
    assert path.name == "episode_002345.parquet"
    assert path.parent.name == "chunk-002"


def test_lerobot_v3_file_path_uses_storage_indices(tmp_path):
    path = resolve_lerobot_path(
        tmp_path,
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        episode_index=9999,
        chunks_size=1000,
        video_key="observation.images.head",
        chunk_index=3,
        file_index=7,
    )
    assert path.as_posix().endswith(
        "videos/observation.images.head/chunk-003/file-007.mp4"
    )
