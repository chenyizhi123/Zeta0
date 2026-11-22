import json
import sqlite3

from psi.data.predictor_pretrain.cache_index import (
    load_anchor_rows_by_sample_ids,
    load_source_anchor_shard,
    load_source_cache_identity,
)
from psi.data.predictor_pretrain.registry import canonical_sha256, write_manifest_rows
from psi.data.predictor_pretrain.runtime import ANCHOR_INDEX_VERSION, build_anchor_index


def _rows():
    result = []
    splits = ("PT_train", "PT_val", "PT_test")
    for index in range(49):
        result.append({
            "sample_id": f"he-{index:03d}",
            "source_id": "HE",
            "split": splits[index % len(splits)],
            "target_timestamp": index / 10,
            "video_path": f"video-{index // 7}.mp4",
            "target_video_timestamp": index / 10,
        })
    result.append({
        "sample_id": "psi-000",
        "source_id": "Psi",
        "split": "PT_train",
        "target_timestamp": 0.6,
        "video_path": "psi.mp4",
        "target_video_timestamp": 0.6,
    })
    return result


def test_v4_anchor_index_partitions_all_splits_into_23_stable_cache_shards(tmp_path):
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, _rows())
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_sha = canonical_sha256(normalized)
    index = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    connection = sqlite3.connect(index)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["format_version"] == ANCHOR_INDEX_VERSION
    positions = connection.execute(
        "SELECT cache_position FROM anchors WHERE source_id='HE' ORDER BY cache_position"
    ).fetchall()
    assert [value[0] for value in positions] == list(range(49))

    shards = [
        load_source_anchor_shard(
            index,
            expected_manifest_sha256=manifest_sha,
            source_id="HE",
            shard_index=shard,
            num_shards=23,
        )
        for shard in range(23)
    ]
    ids = [row["sample_id"] for shard in shards for row in shard.rows]
    assert ids == sorted(row["sample_id"] for row in normalized if row["source_id"] == "HE")
    assert len(ids) == len(set(ids)) == 49
    assert max(len(shard.rows) for shard in shards) - min(len(shard.rows) for shard in shards) <= 1
    assert {row["split"] for shard in shards for row in shard.rows} == {
        "PT_train", "PT_val", "PT_test"
    }
    identity = load_source_cache_identity(
        index, expected_manifest_sha256=manifest_sha, source_id="HE"
    )
    assert identity.sample_ids == ids
    assert identity.target_timestamps == [
        row["target_timestamp"]
        for row in sorted(
            (row for row in normalized if row["source_id"] == "HE"),
            key=lambda row: row["sample_id"],
        )
    ]


def test_cache_index_fetches_only_requested_legacy_sample_ids(tmp_path):
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, _rows())
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_sha = canonical_sha256(normalized)
    index = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    rows = load_anchor_rows_by_sample_ids(
        index,
        expected_manifest_sha256=manifest_sha,
        sample_ids=["he-003", "absent", "he-001", "he-003"],
        source_id="HE",
    )
    assert [row["sample_id"] for row in rows] == ["he-003", "he-001"]


def test_v3_anchor_index_fallback_is_bounded_and_stably_sharded(tmp_path):
    path = tmp_path / "legacy.sqlite"
    rows = _rows()
    manifest_sha = "f" * 64
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE anchors(sample_id TEXT PRIMARY KEY,source_id TEXT NOT NULL,"
        "split TEXT NOT NULL,source_position INTEGER,payload TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO anchors VALUES(?,?,?,?,?)",
        [
            (row["sample_id"], row["source_id"], row["split"], None, json.dumps(row))
            for row in reversed(rows)
        ],
    )
    connection.execute("CREATE INDEX anchors_split_source ON anchors(split,source_id)")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        (("format_version", "predictor_pretrain_anchor_index_v3"),
         ("manifest_sha256", manifest_sha), ("num_rows", str(len(rows)))),
    )
    connection.commit()
    connection.close()

    shards = [
        load_source_anchor_shard(
            path,
            expected_manifest_sha256=manifest_sha,
            source_id="HE",
            shard_index=index,
            num_shards=2,
        )
        for index in range(2)
    ]
    ids = [row["sample_id"] for shard in shards for row in shard.rows]
    identity = load_source_cache_identity(
        path, expected_manifest_sha256=manifest_sha, source_id="HE"
    )
    assert ids == identity.sample_ids
    assert len(ids) == len(set(ids)) == 49
    assert all(shard.selection_order == "legacy_split_then_sample_id_v1" for shard in shards)
