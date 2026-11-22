#!/usr/bin/env python3
"""Publish an explicitly approved source-to-runtime research transition, not weights."""
import argparse
import json
from pathlib import Path

from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path
from psi.data.predictive_world.processor_migration import (
    make_processor_migration, migration_target_provenance,
)
from psi.data.predictive_world.provenance import verify_qwen_provenance
from psi.models.predictive_world.initialization import read_initialization_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--asset-path-map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--accepted-by', required=True)
    parser.add_argument('--acceptance-record', required=True)
    parser.add_argument('--accept-non-equivalent-post-training', action='store_true')
    args = parser.parse_args()
    if not args.accept_non_equivalent_post_training:
        parser.error('explicit non-equivalent research post-training approval is required')
    output = Path(args.output).absolute()
    if output.exists():
        raise FileExistsError(output)
    contract = make_processor_migration(args.report, accepted_by=args.accepted_by,
                                       acceptance_record=args.acceptance_record)
    mapping = load_asset_path_map(args.asset_path_map)
    source, payload, _ = read_initialization_source(
        args.checkpoint, args.source_config, asset_path_map=mapping)
    manifest = resolve_asset_path(payload['data']['feature_cache'], mapping)
    if manifest.is_dir():
        manifest /= 'feature_manifest.json'
    target = migration_target_provenance(json.loads(manifest.read_text())['provenance'],
                                         contract, source, args.source_config)
    identity = verify_qwen_provenance(resolve_asset_path(payload['model']['qwen_model'], mapping), target)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: this is a separate immutable approval, never a source edit.
    with output.open('x') as handle:
        handle.write(json.dumps(contract, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'research_processor_migration_approved',
                      'path': str(output), 'actual_teacher': identity,
                      'historical_equivalence': False, 'production_accepted': False,
                      'optimizer_steps': 0, 'source_cache_reused': False}))


if __name__ == '__main__':
    main()
