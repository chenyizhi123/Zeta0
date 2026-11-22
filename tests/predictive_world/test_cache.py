import json
import hashlib

import pytest
import torch

from psi.data.predictive_world.cache import (
    LatentCache,
    RaggedFeatureStore,
    FEATURE_RECORDS_FORMAT,
    feature_records_metadata_path,
    feature_identity,
    verify_predictive_world_feature_manifest,
    verify_predictive_world_feature_records,
)


def _provenance():
    return {
        "encoder_id": "qwen-test", "processor_id": "processor-test",
        "image_preprocessing": "aspect_v1", "feature_layer": "post_merger",
        "dtype": "float32",
    }


def _binding():
    return {
        "format": "zeta0.predictive_world.index_binding.v1",
        "semantic_identity": "0" * 64,
        "index_sha256": "1" * 64,
        "lookup_sha256": "2" * 64,
        "temporal_abi_sha256": "3" * 64,
    }


def test_feature_identity_ignores_training_topology_and_store_reuses_object(tmp_path):
    first = feature_identity(
        frame_id="episode/3", actual_time=0.2, camera_id="head", grid_hw=(2, 2),
        provenance=_provenance(),
    )
    second = feature_identity(
        frame_id="episode/3", actual_time=0.4, camera_id="head", grid_hw=(2, 2),
        provenance=_provenance(),
    )
    assert first == second
    store = RaggedFeatureStore(tmp_path, provenance=_provenance())
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    key = store.put(
        frame_id="episode/3", values=values, grid_hw=(2, 2), actual_time=0.2,
        camera_id="head",
    )
    assert store.put(
        frame_id="episode/3", values=values, grid_hw=(2, 2), actual_time=0.2,
        camera_id="head",
    ) == key
    manifest = store.publish([
        {
            "sample_id": "sample", "group": 0, "key": key, "actual_time": 0.4,
            "camera_id": "head", "frame_id": "episode/3",
        }
    ], index_binding=_binding(), feature_records_sha256="4" * 64)
    record = json.loads(manifest.read_text())["records"][0]
    assert record["key"] == key
    assert record["actual_time"] == 0.4
    assert verify_predictive_world_feature_manifest(manifest)["index_binding"] == _binding()


def test_feature_release_is_immutable(tmp_path):
    store = RaggedFeatureStore(tmp_path, provenance=_provenance())
    key = store.put(
        frame_id="frame", values=torch.ones(1, 2), grid_hw=(1, 1), actual_time=0.1,
        camera_id="head",
    )
    first = {
        "sample_id": "a", "group": 0, "key": key, "camera_id": "head",
        "frame_id": "frame",
    }
    store.publish(
        [first], index_binding=_binding(), feature_records_sha256="4" * 64
    )
    with pytest.raises(ValueError, match="immutable"):
        store.publish(
            [{**first, "sample_id": "b"}],
            index_binding=_binding(), feature_records_sha256="4" * 64,
        )


def test_feature_records_sidecar_binds_index_and_content(tmp_path):
    records = tmp_path / "records.jsonl"
    encoded = '{"sample_id":"sample"}\n'
    records.write_text(encoded)
    feature_records_metadata_path(records).write_text(json.dumps({
        "format": FEATURE_RECORDS_FORMAT,
        "records": 1,
        "records_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "index_binding": _binding(),
    }))
    assert verify_predictive_world_feature_records(records)["index_binding"] == _binding()
    records.write_text(encoded + encoded)
    with pytest.raises(ValueError, match="content hash"):
        verify_predictive_world_feature_records(records)


def test_latent_cache_is_strictly_bound_to_reference(tmp_path):
    path = tmp_path / "latents.safetensors"
    latent = torch.randn(2, 3, 2, 4)
    valid = torch.ones(2, 3, dtype=torch.bool)
    LatentCache(
        path, reference_id="reference-a", latent_shape=(3, 2, 4), input_fingerprint="inputs-a"
    ).publish(
        ("a", "b"), latent, valid
    )
    ids, restored, restored_valid = LatentCache(
        path, reference_id="reference-a", latent_shape=(3, 2, 4), input_fingerprint="inputs-a"
    ).load()
    assert ids == ("a", "b")
    torch.testing.assert_close(restored, latent)
    assert torch.equal(restored_valid, valid)
    with pytest.raises(ValueError, match="reference"):
        LatentCache(
            path, reference_id="reference-b", latent_shape=(3, 2, 4), input_fingerprint="inputs-a"
        ).load()
