import numpy as np
import pytest

from psi.data.predictor_pretrain.sparse_cache import (
    SPARSE_CACHE_COMPOSITE_VERSION,
    SparseEndpointCache,
    SparseEndpointCacheWriter,
    publish_sparse_endpoint_composite,
    write_sparse_endpoint_cache,
)


def test_sparse_cache_round_trip_and_sample_id_lookup(tmp_path):
    fields = np.random.default_rng(0).normal(size=(3, 16, 2048)).astype(np.float32)
    manifest = write_sparse_endpoint_cache(
        tmp_path,
        sample_ids=["a", "b", "c"],
        target_timestamps=[0.6, 1.1, 1.6],
        fields=fields,
        provenance={"teacher_sha256": "x" * 64, "split": "PT_train"},
    )
    cache = SparseEndpointCache(manifest)
    value, timestamp = cache.get("b")
    np.testing.assert_array_equal(value, fields[1].astype(np.float16))
    assert timestamp == 1.1
    with pytest.raises(KeyError):
        cache.get("unknown")


def test_sparse_cache_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        write_sparse_endpoint_cache(
            tmp_path,
            sample_ids=["a", "a"],
            target_timestamps=[0.6, 1.1],
            fields=np.zeros((2, 16, 2048), dtype=np.float16),
            provenance={},
        )


def test_sparse_cache_streaming_writer_requires_every_row(tmp_path):
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a", "b"],
        target_timestamps=[0.6, 1.1],
        provenance={},
    )
    writer.write([1], np.ones((1, 16, 2048), dtype=np.float32))
    with pytest.raises(RuntimeError, match="missing rows"):
        writer.publish()
    writer.write([0], np.zeros((1, 16, 2048), dtype=np.float32))
    cache = SparseEndpointCache(writer.publish())
    np.testing.assert_array_equal(cache.get("a")[0], np.zeros((16, 2048), dtype=np.float16))


def test_sparse_cache_resumes_only_durable_rows(tmp_path):
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a", "b", "c"],
        target_timestamps=[0.6, 1.1, 1.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    writer.write([0], np.ones((1, 16, 2048), dtype=np.float32))
    writer.checkpoint()
    writer.write([1], np.full((1, 16, 2048), 2, dtype=np.float32))
    writer.close()

    resumed = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a", "b", "c"],
        target_timestamps=[0.6, 1.1, 1.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    assert resumed.missing_offsets().tolist() == [1, 2]
    resumed.write([1, 2], np.full((2, 16, 2048), 3, dtype=np.float32))
    cache = SparseEndpointCache(resumed.publish())
    np.testing.assert_array_equal(cache.get("a")[0], np.ones((16, 2048), dtype=np.float16))
    np.testing.assert_array_equal(cache.get("b")[0], np.full((16, 2048), 3, dtype=np.float16))
    assert not (tmp_path / ".sparse_endpoint_build").exists()


def test_sparse_cache_can_durably_invalidate_selected_rows(tmp_path):
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a", "b", "c"],
        target_timestamps=[0.6, 1.1, 1.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    writer.write([0, 1, 2], np.ones((3, 16, 2048), dtype=np.float32))
    assert writer.invalidate([1, 1]) == 1
    writer.checkpoint()
    writer.close()

    resumed = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a", "b", "c"],
        target_timestamps=[0.6, 1.1, 1.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    assert resumed.missing_offsets().tolist() == [1]
    resumed.close()


def test_sparse_cache_refuses_incompatible_resume(tmp_path):
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a"],
        target_timestamps=[0.6],
        provenance={"teacher": "old"},
        resume=True,
    )
    writer.close()
    with pytest.raises(RuntimeError, match="incompatible"):
        SparseEndpointCacheWriter(
            tmp_path,
            sample_ids=["a"],
            target_timestamps=[0.6],
            provenance={"teacher": "new"},
            resume=True,
        )


def test_sparse_cache_reuses_matching_published_generation(tmp_path):
    fields = np.ones((1, 16, 2048), dtype=np.float32)
    manifest = write_sparse_endpoint_cache(
        tmp_path,
        sample_ids=["a"],
        target_timestamps=[0.6],
        fields=fields,
        provenance={"teacher": "fixed"},
    )
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a"],
        target_timestamps=[0.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    assert writer.num_written == 1
    assert writer.missing_offsets().tolist() == []
    assert writer.publish() == manifest


def test_sparse_cache_writer_lock_rejects_concurrent_same_output(tmp_path):
    writer = SparseEndpointCacheWriter(
        tmp_path,
        sample_ids=["a"],
        target_timestamps=[0.6],
        provenance={"teacher": "fixed"},
        resume=True,
    )
    with pytest.raises(RuntimeError, match="another sparse cache builder"):
        SparseEndpointCacheWriter(
            tmp_path,
            sample_ids=["a"],
            target_timestamps=[0.6],
            provenance={"teacher": "fixed"},
            resume=True,
        )
    writer.close()


def test_composite_cache_publishes_one_source_manifest_without_copying_shards(tmp_path):
    base = {
        "teacher_sha256": "teacher",
        "teacher_processor_fingerprint": {"processor": "fixed"},
        "dataset_manifest_sha256": "dataset",
        "source_id": "HE",
        "image_crop": "none",
    }
    ids = ["a", "b", "c", "d"]
    timestamps = [0.6, 1.1, 1.6, 2.1]
    shard_paths = []
    # Deliberately use physical/decode order inside each shard.  The composite
    # partition contract is sample-id membership, not shard-local array order.
    for index, order in enumerate(((1, 0), (3, 2))):
        start, stop = index * 2, (index + 1) * 2
        shard_ids = [ids[value] for value in order]
        shard_timestamps = [timestamps[value] for value in order]
        fields = np.stack([
            np.full((16, 2048), value + 1, dtype=np.float16) for value in order
        ])
        shard_paths.append(write_sparse_endpoint_cache(
            tmp_path / "shards" / str(index),
            sample_ids=shard_ids,
            target_timestamps=shard_timestamps,
            fields=fields,
            provenance={
                **base,
                "cache_shard": {
                    "format_version": "predictor_endpoint_cache_shard_v1",
                    "index": index,
                    "count": 2,
                    "start": start,
                    "stop": stop,
                    "source_num_samples": 4,
                    "selection_order": "source_sample_id_cache_position_v1",
                    "anchor_index_format_version": "predictor_pretrain_anchor_index_v4",
                },
            },
        ))
    manifest = publish_sparse_endpoint_composite(
        tmp_path / "final",
        sample_ids=ids,
        target_timestamps=timestamps,
        provenance=base,
        shard_manifests=shard_paths,
    )
    cache = SparseEndpointCache(manifest)
    assert cache.manifest["format_version"] == SPARSE_CACHE_COMPOSITE_VERSION
    assert len(cache.manifest["shards"]) == 2
    for value, sample_id in enumerate(ids):
        field, timestamp = cache.get(sample_id)
        np.testing.assert_array_equal(
            field, np.full((16, 2048), value + 1, dtype=np.float16)
        )
        assert timestamp == timestamps[value]
    assert not list((tmp_path / "final").glob("endpoint_fields.*.npy"))


def test_composite_cache_refuses_wrong_shard_membership(tmp_path):
    base = {"source_id": "HE", "dataset_manifest_sha256": "dataset"}
    paths = []
    for index, shard_ids in enumerate((("a",), ("a",))):
        paths.append(write_sparse_endpoint_cache(
            tmp_path / "shards" / str(index),
            sample_ids=shard_ids,
            target_timestamps=[0.6],
            fields=np.zeros((1, 16, 2048), dtype=np.float16),
            provenance={
                **base,
                "cache_shard": {
                    "format_version": "predictor_endpoint_cache_shard_v1",
                    "index": index,
                    "count": 2,
                    "start": index,
                    "stop": index + 1,
                    "source_num_samples": 2,
                    "selection_order": "stable",
                },
            },
        ))
    with pytest.raises(ValueError, match="sample ids"):
        publish_sparse_endpoint_composite(
            tmp_path / "final",
            sample_ids=["a", "b"],
            target_timestamps=[0.6, 1.1],
            provenance=base,
            shard_manifests=paths,
        )
