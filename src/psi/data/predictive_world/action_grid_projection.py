"""Extend audited 50 Hz Arena action labels without changing visual targets."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from psi.data.predictive_world.alignment import causal_floor_indices
from psi.data.predictive_world.cache import file_sha256, verify_predictive_world_feature_manifest
from psi.data.predictive_world.current_state import _rebind_packed_features
from psi.data.predictive_world.index import (
    predictive_world_index_binding, publish_predictive_world_index, verify_predictive_world_index,
)
from psi.data.predictive_world.sonic_post_training import canonical_frame_timestamps


def extend_action_row(row, *, times, native_rows, horizon, max_age):
    """Keep existing slots exact, including invalid padding and relative times."""
    anchor = row['anchor_time']
    indices, valid, actual = causal_floor_indices(
        times, anchor + np.arange(1, horizon + 1) / 50,
        max_age_seconds=max_age, frame_grid_hz=50,
    )
    valid &= actual > anchor + 1e-12
    relative = np.rint((actual - anchor) * 50) / 50
    fields = dict(action_indices=np.asarray(native_rows)[indices].tolist(),
                  action_valid=valid.tolist(), action_actual_times=relative.tolist())
    old_horizon = len(row['action_indices'])
    if horizon <= old_horizon:
        raise ValueError('action projection requires an extended horizon')
    for key, values in fields.items():
        if not np.array_equal(np.asarray(values[:old_horizon]), np.asarray(row[key]), equal_nan=True):
            raise ValueError('original action labels disagree with audited episode: ' + key)
    return {**row, **fields}


def publish_action_grid_projection(source_index, output_index, *, horizon=50,
                                   source_feature_manifest=None, output_feature_cache=None):
    source_index, output_index = Path(source_index), Path(output_index)
    if output_index.exists():
        raise ValueError('action projection destination already exists')
    if (source_feature_manifest is None) != (output_feature_cache is None):
        raise ValueError('feature reuse requires source and destination')
    meta = verify_predictive_world_index(source_index)
    temporal = copy.deepcopy(meta['temporal_abi'])
    if temporal['action_hz'] != 50 or temporal['action_horizon'] != 30 or horizon != 50:
        raise ValueError('supported Arena action projection is 30 to 50 at 50 Hz')
    binding = predictive_world_index_binding(source_index, verify_content_hashes=False)
    projection = dict(format='zeta0.predictive_world.action_grid_projection.v1',
                      source_index_binding=binding, source_horizon=30, target_horizon=50,
                      unchanged_fields='all_except_action_indices_action_valid_action_actual_times',
                      projection_code_sha256=file_sha256(Path(__file__)))
    manifest = None
    if source_feature_manifest:
        manifest = verify_predictive_world_feature_manifest(source_feature_manifest)
        if manifest['index_binding'] != binding or manifest['rows'] != meta['rows']:
            raise ValueError('source feature index mismatch')
        projection['source_feature_manifest_sha256'] = file_sha256(Path(source_feature_manifest))
    temporal.update(action_horizon=horizon, action_offsets_s=(np.arange(1, horizon + 1)/50).tolist())
    hashes = {}

    def rows():
        current_path = current_episode = None
        with source_index.open() as stream:
            for line in stream:
                row = json.loads(line)
                path = row['materialization']['data_path']
                if path != current_path:
                    table = pq.read_table(path, columns=['episode_index', 'frame_index', 'timestamp'])
                    episode_ids = np.asarray(table['episode_index'])
                    current_path, current_episode = path, None
                    hashes[path] = file_sha256(Path(path))
                if row['episode_id'] != current_episode:
                    ep = int(row['episode_id'].rsplit('episode_', 1)[1])
                    native_rows = np.flatnonzero(episode_ids == ep)
                    episode = table.take(native_rows)
                    times = canonical_frame_timestamps(np.asarray(episode['timestamp']), np.asarray(episode['frame_index']))
                    current_episode = row['episode_id']
                ordinal = row['anchor_ordinal']
                if not 0 <= ordinal < len(times) or row['anchor_time'] != times[ordinal]:
                    raise ValueError('anchor disagrees with source episode')
                yield extend_action_row(row, times=times, native_rows=native_rows, horizon=horizon,
                                        max_age=temporal['selection']['demonstrated_action']['max_age_s'])
        for path, digest in hashes.items():
            if file_sha256(Path(path)) != digest:
                raise ValueError('source parquet changed during projection')

    publish_predictive_world_index(output_index, rows=rows(),
        provenance={**meta['provenance'], 'action_grid_projection': projection},
        temporal_abi=temporal, presorted=True)
    target = verify_predictive_world_index(output_index)
    if target['rows'] != meta['rows']:
        raise ValueError('projection changed sample count')
    if manifest is not None:
        _rebind_packed_features(Path(source_feature_manifest), manifest, Path(output_feature_cache),
                               predictive_world_index_binding(output_index, verify_content_hashes=False), projection)
    return dict(rows=target['rows'], parquet_sha256=hashes, projection=projection)
