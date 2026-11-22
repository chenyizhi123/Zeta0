#!/usr/bin/env python3
"""Validate immutable cache shards and atomically publish one source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psi.data.predictor_pretrain.cache_index import load_source_cache_identity
from psi.data.predictor_pretrain.sparse_cache import publish_sparse_endpoint_composite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-index", required=True)
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--shard-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    freeze = json.loads(Path(args.dataset_freeze).read_text(encoding="utf-8"))
    manifest_sha = str(freeze.get("manifest_sha256", ""))
    if not manifest_sha:
        raise ValueError("dataset freeze lacks its canonical manifest hash")
    identity = load_source_cache_identity(
        args.anchor_index,
        expected_manifest_sha256=manifest_sha,
        source_id=args.source_id,
    )
    shard_manifests = [Path(value).resolve() for value in args.shard_manifest]
    first = json.loads(shard_manifests[0].read_text(encoding="utf-8"))
    provenance = dict(first.get("provenance", {}))
    if not isinstance(provenance.pop("cache_shard", None), dict):
        raise ValueError("first cache shard lacks partition provenance")
    if (
        provenance.get("source_id") != args.source_id
        or provenance.get("dataset_manifest_sha256") != manifest_sha
    ):
        raise ValueError("cache shard does not belong to the requested frozen source")
    result = publish_sparse_endpoint_composite(
        args.output_dir,
        sample_ids=identity.sample_ids,
        target_timestamps=identity.target_timestamps,
        provenance=provenance,
        shard_manifests=shard_manifests,
    )
    print(result)


if __name__ == "__main__":
    main()
