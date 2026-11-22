import json
from pathlib import Path
import subprocess
import sys

from psi.data.predictor_pretrain.registry import canonical_sha256, write_manifest_rows
from psi.data.predictor_pretrain.sample_stream import (
    CompactV2OccurrenceResolver,
    iter_sample_occurrences,
    occurrence_state_dropout,
    sample_stream_spec_sha256,
)


def _rows():
    return [
        {
            "sample_id": f"{source}:{index}",
            "source_id": source,
            "split": "PT_train",
        }
        for source in ("HIW", "HE", "EgoDex")
        for index in range(5)
    ]


def test_stream_is_global_deterministic_and_model_independent():
    kwargs = dict(
        source_weights={"HIW": 28, "HE": 2, "EgoDex": 70},
        total_updates=3,
        global_batch_size=8,
        seed=101,
    )
    first = list(iter_sample_occurrences(_rows(), **kwargs))
    second = list(iter_sample_occurrences(list(reversed(_rows())), **kwargs))
    assert first == second
    assert len(first) == 24
    assert [(row["update"], row["position_in_global_batch"]) for row in first] == [
        (index // 8, index % 8) for index in range(24)
    ]


def test_state_dropout_and_stream_identity_are_separate_and_stable():
    values = [occurrence_state_dropout(seed=7, occurrence_index=i, probability=0.2) for i in range(100)]
    assert values == [occurrence_state_dropout(seed=7, occurrence_index=i, probability=0.2) for i in range(100)]
    assert 0 < sum(values) < 100
    common = dict(
        dataset_manifest_sha256="a" * 64,
        source_weights={"HIW": 28, "HE": 2, "EgoDex": 70},
        total_updates=30_000,
        global_batch_size=512,
        state_dropout_probability=0.2,
    )
    assert sample_stream_spec_sha256(seed=101, **common) != sample_stream_spec_sha256(seed=202, **common)


def test_v2_source_cycles_without_replacement_and_is_order_independent():
    from collections import defaultdict
    rows = [{"source_id": source, "sample_id": f"{source}-{index}", "split": "PT_train"}
            for source in ("a", "b") for index in range(7)]
    kwargs = dict(source_weights={"a": 3, "b": 1}, total_updates=20,
                  global_batch_size=8, seed=101, sampling_version="v2")
    occurrences = list(iter_sample_occurrences(rows, **kwargs))
    assert occurrences == list(iter_sample_occurrences(list(reversed(rows)), **kwargs))
    by_source = defaultdict(list)
    for item in occurrences:
        by_source[item["source_id"]].append(item["sample_id"])
    for values in by_source.values():
        for start in range(0, len(values), 7):
            assert len(set(values[start:start+7])) == len(values[start:start+7])


def test_compact_v2_resolver_is_random_access_exact_weight_and_bijective():
    resolver = CompactV2OccurrenceResolver(
        source_weights={"a": 3, "b": 1},
        source_capacities={"a": 7, "b": 5},
        seed=101,
        state_dropout_probability=0.2,
    )
    first = [resolver.occurrence(index) for index in range(4)]
    assert [item["source_id"] for item in first].count("a") == 3
    assert [item["source_id"] for item in first].count("b") == 1
    assert resolver.occurrence(12345) == resolver.occurrence(12345)
    positions = {"a": [], "b": []}
    for index in range(100):
        item = resolver.occurrence(index)
        positions[item["source_id"]].append(item["source_position"])
    for source, capacity in (("a", 7), ("b", 5)):
        for start in range(0, len(positions[source]), capacity):
            chunk = positions[source][start:start + capacity]
            assert len(set(chunk)) == len(chunk)


def test_v2_builder_represents_full_30k_run_with_small_descriptor(tmp_path):
    rows = [
        {
            "format_version": "predictor_pretrain_anchor_v2",
            "sample_id": f"s{index}", "source_id": "source",
            "group_id": f"g{index}", "episode_id": str(index),
            "split": "PT_train",
        }
        for index in range(7)
    ]
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, rows)
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": canonical_sha256(normalized)}))
    output = tmp_path / "stream.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts/data/build_pretrain_sample_stream.py"
    subprocess.run(
        [
            sys.executable, str(script), "--pretrain-manifest", str(manifest),
            "--dataset-freeze", str(freeze), "--output", str(output),
            "--source-weight", "source=1", "--updates", "30000",
            "--global-batch-size", "16384", "--sampling-version", "v2",
        ],
        check=True,
    )
    meta = json.loads(Path(str(output) + ".meta.json").read_text())
    assert output.stat().st_size < 2048
    assert len(output.read_text().splitlines()) == 1
    assert meta["stream_encoding"] == "algorithmic_v2"
    assert meta["num_occurrences"] == 30_000 * 16_384


def test_v2_builder_rejects_silently_omitted_train_source(tmp_path):
    rows = [
        {
            "format_version": "predictor_pretrain_anchor_v2",
            "sample_id": source, "source_id": source,
            "group_id": source, "episode_id": "0", "split": "PT_train",
        }
        for source in ("a", "b")
    ]
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, rows)
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": canonical_sha256(normalized)}))
    script = Path(__file__).resolve().parents[1] / "scripts/data/build_pretrain_sample_stream.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--pretrain-manifest", str(manifest),
            "--dataset-freeze", str(freeze), "--output", str(tmp_path / "stream.jsonl"),
            "--source-weight", "a=1", "--updates", "1",
            "--global-batch-size", "1", "--sampling-version", "v2",
        ],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "PT_train sources differ" in result.stderr
