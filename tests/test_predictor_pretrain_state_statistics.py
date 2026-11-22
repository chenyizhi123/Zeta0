import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictor_pretrain.registry import (
    canonical_sha256, iter_manifest_rows, load_manifest_rows, manifest_sha256,
    write_manifest_rows,
)
from psi.data.predictor_pretrain.runtime import (
    LeRobotAnchorReader, LeRobotRuntimeSource, build_anchor_index,
)


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/data/fit_pretrain_state_statistics.py"
    spec = importlib.util.spec_from_file_location("fit_state_statistics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".parquet"])
def test_streaming_manifest_hash_matches_normalized_canonical(tmp_path, suffix):
    path = tmp_path / f"anchors{suffix}"
    rows = [
        {"source_id": " 机器人 ", "sample_id": " β ", "episode_id": 12,
         "lineage_ids": ["z", "a", "z"], "paths": ["数据/乙", "数据/甲"],
         "content_hashes": "[\"z\",\"a\"]", "nested": {"乙": 1.25, "a": [True, None]}},
        {"sample_id": "α", "source_id": "机器人", "lineage_ids": "单行", "paths": None},
    ]
    if suffix == ".jsonl":
        path.write_text("\n" + "\n \n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    elif suffix == ".json":
        path.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    else:
        # The parquet path remains compatible with load_manifest_rows' schema.
        write_manifest_rows(path, rows)
    expected = load_manifest_rows(path)
    assert list(iter_manifest_rows(path)) == expected
    assert manifest_sha256(path) == canonical_sha256(expected)


def test_jsonl_helpers_do_not_materialize_manifest_and_reject_non_objects(tmp_path, monkeypatch):
    from psi.data.predictor_pretrain import registry

    path = tmp_path / "anchors.jsonl"
    path.write_text('{"sample_id":"a","source_id":"s"}\n')
    expected = canonical_sha256(load_manifest_rows(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("JSONL streaming must not call the list loader")

    monkeypatch.setattr(registry, "load_manifest_rows", forbidden)
    assert manifest_sha256(path) == expected
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="non-object"):
        manifest_sha256(path)
    path.write_text("\n \n")
    assert manifest_sha256(path) == canonical_sha256([])


def _fixture(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({
        "arm": [[1 / 7, -2.5], [3.25, 4.75], [-1.0, 9.0], [10000.0, 10000.0]],
        "leg": [7.0, 7.0, 7.0, 10000.0],
    }), first, row_group_size=2)
    pq.write_table(pa.table({
        "state": [[2.0, 3.0, 7.0], [float("nan"), 3.0, 7.0], [4.0, 5.0, 7.0]],
    }), second, row_group_size=1)
    source_values = {
        "a": {"state_schema": "shared", "state_columns": ["arm", "leg"]},
        "reordered": {"state_schema": "reordered", "state_columns": ["leg", "arm"]},
        "b": {"state_schema": None, "state_schema_field": "robot", "state_columns": ["state"]},
        "missing": {"state_schema": None},
    }

    def row(sample, source, path, indices, mask, split="PT_train", **kwargs):
        return {"sample_id": sample, "source_id": source, "data_path": str(path),
                "state_row_indices": indices, "state_history_mask": mask, "split": split, **kwargs}

    rows = [
        row("a0", "a", first, [0, 0, 1, 3], [True, True, True, False]),
        row("a1", "a", first, [1, 2, 2, 3], [True, True, True, False]),
        row("a2", "a", first, [2], [True]),
        row("a-val", "a", first, [3], [True], split="PT_val"),
        row("a-test", "a", first, [3], [True], split="PT_test"),
        row("reordered", "reordered", first, [1, 2], [True, True]),
        row("b0", "b", second, [0, 0, 1, 2], [True, True, False, True], robot="shared"),
        row("missing", "missing", tmp_path / "absent.parquet", [0], [False]),
    ]
    sources = {key: LeRobotRuntimeSource(source_id=key, **value) for key, value in source_values.items()}
    return rows, sources, source_values


def _legacy_moments(rows, sources):
    reader = LeRobotAnchorReader(sources)
    result = {}
    for row in rows:
        if row["split"] != "PT_train":
            continue
        source = sources[row["source_id"]]
        schema = reader._schema(row, source)
        if schema is None:
            continue
        states, valid = reader._state(row, source)
        values = states[valid].astype(np.float64)
        if not len(values):
            continue
        current = result.setdefault(schema, {"count": 0, "sum": np.zeros(values.shape[1]),
                                             "sumsq": np.zeros(values.shape[1])})
        current["count"] += len(values)
        current["sum"] += values.sum(0)
        current["sumsq"] += np.square(values).sum(0)
    return result


def test_weighted_statistics_equal_legacy_and_read_each_parquet_once(tmp_path, monkeypatch):
    module = _module()
    rows, sources, _ = _fixture(tmp_path)
    expected = _legacy_moments(rows, sources)
    parquet_files = []
    original = pq.ParquetFile

    def recording(path, *args, **kwargs):
        parquet_files.append(str(path))
        return original(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("weighted statistics must not read scalar state per anchor")

    monkeypatch.setattr(pq, "ParquetFile", recording)
    monkeypatch.setattr(LeRobotAnchorReader, "_state", forbidden)
    actual = module.fit_weighted_moments(iter(rows), sources, parquet_batch_size=2)
    assert sorted(parquet_files) == sorted({row["data_path"] for row in rows if row["source_id"] != "missing"})
    assert actual.keys() == expected.keys()
    assert actual["shared"]["count"] == 10  # repeated history slots count repeatedly
    for schema in actual:
        assert actual[schema]["count"] == expected[schema]["count"]
        for field in ("sum", "sumsq"):
            np.testing.assert_allclose(actual[schema][field], expected[schema][field], rtol=1e-14)
    result = module.statistics_from_moments(actual, "manifest")
    legacy = module.statistics_from_moments(expected, "manifest")
    assert result["schemas"]["shared"]["constant_dimensions"] == [2]
    for schema in result["schemas"]:
        for field in ("mean", "std"):
            np.testing.assert_allclose(result["schemas"][schema][field], legacy["schemas"][schema][field], rtol=1e-14)


def _cli_fixture(tmp_path):
    rows, _, source_values = _fixture(tmp_path)
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, rows)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": manifest_sha256(manifest)}))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(source_values))
    index = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    return manifest, freeze, sources, index


def _argv(manifest, freeze, sources, output, index=None):
    args = ["fit", "--anchor-manifest", str(manifest), "--dataset-freeze", str(freeze),
            "--source-configs", str(sources), "--output", str(output)]
    return args if index is None else [*args, "--anchor-index", str(index)]


def test_index_and_legacy_cli_match(tmp_path, monkeypatch):
    module = _module()
    manifest, freeze, sources, index = _cli_fixture(tmp_path)
    outputs = [tmp_path / "legacy.json", tmp_path / "indexed.json"]
    for output, selected_index in zip(outputs, [None, index]):
        monkeypatch.setattr(sys, "argv", _argv(manifest, freeze, sources, output, selected_index))
        module.main()
    assert json.loads(outputs[0].read_text()) == json.loads(outputs[1].read_text())


@pytest.mark.parametrize("corrupt", ["freeze", "index", "manifest"])
def test_hash_mismatch_fails_before_reading_states_or_writing_output(tmp_path, monkeypatch, corrupt):
    module = _module()
    manifest, freeze, sources, index = _cli_fixture(tmp_path)
    output = tmp_path / "statistics.json"
    if corrupt == "freeze":
        freeze.write_text(json.dumps({"manifest_sha256": "0" * 64}))
    elif corrupt == "index":
        with sqlite3.connect(index) as connection:
            connection.execute("UPDATE metadata SET value=? WHERE key='manifest_sha256'", ("0" * 64,))
    else:
        manifest.write_text(manifest.read_text().replace('"a0"', '"changed"'))

    def forbidden(*args, **kwargs):
        raise AssertionError("must check frozen hashes before any state reads")

    monkeypatch.setattr(module, "fit_weighted_moments", forbidden)
    monkeypatch.setattr(sys, "argv", _argv(manifest, freeze, sources, output, index))
    with pytest.raises(ValueError, match="freeze|frozen manifest"):
        module.main()
    assert not output.exists()


def test_valid_nonfinite_states_and_schema_dimension_changes_fail(tmp_path):
    module = _module()
    rows, sources, _ = _fixture(tmp_path)
    rows[-2]["state_history_mask"] = [True, True, True, True]
    with pytest.raises(ValueError, match="non-finite PT_train state"):
        module.fit_weighted_moments(rows, sources, parquet_batch_size=2)
    rows[-2]["state_history_mask"] = [True, True, False, True]
    pq.write_table(pa.table({"state": [[1.0], [2.0], [3.0]]}), tmp_path / "second.parquet")
    with pytest.raises(ValueError, match="state dimension changed"):
        module.fit_weighted_moments(rows, sources, parquet_batch_size=2)


def test_rebuild_passes_frozen_index_to_statistics():
    script = Path(__file__).resolve().parents[1] / "scripts/data/rebuild_he_psi0_real_v1_manifests.sh"
    invocation = script.read_text().split("scripts/data/fit_pretrain_state_statistics.py", 1)[1]
    assert '--anchor-index "$ARTIFACT_ROOT/anchor_index.sqlite"' in invocation
