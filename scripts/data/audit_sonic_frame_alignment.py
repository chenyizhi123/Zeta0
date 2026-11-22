#!/usr/bin/env python3
"""Read-only CPU audit of actual SONIC windows against an integer-frame oracle.

Checks every anchor, including padded boundaries. Publishes no index, cache or
training artifact and never decodes video. JSON progress/results go to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path
from psi.data.predictive_world.index import temporal_abi_from_data_config
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.sonic_post_training import (
    MEASUREMENT, TASKS, TIME_ALIGNMENT, episode_windows, extend_registries, inspect_dataset,
)


def check_episode(episode, windows, temporal):
    """Compare production output to independent integer arithmetic, not a lookup."""
    n = len(episode['times'])
    np.testing.assert_array_equal([row['anchor_ordinal'] for row in windows], np.arange(n))
    np.testing.assert_array_equal([row['anchor_time'] for row in windows], np.arange(n)/50)
    anchors = np.arange(n)[:, None]
    video_rows = np.arange(n) + episode['frame_offset']
    channels = (
        ('history_visual', 'history_visual_offsets_s', 'history_visual', video_rows, False),
        ('state', 'state_offsets_s', 'state_history', episode['row_indices'], False),
        ('future_visual', 'future_group_ends_s', 'future_visual', video_rows, True),
        ('action', 'action_offsets_s', 'demonstrated_action', episode['row_indices'], True),
        ('future_measurement', 'future_measurement_offsets_s', 'future_measurement', episode['row_indices'], True),
    )
    for channel, offset_key, selection_key, native_rows, future_only in channels:
        ticks = np.asarray(temporal[offset_key])*50
        np.testing.assert_allclose(ticks, np.rint(ticks), rtol=0, atol=1e-12)
        queries = anchors + np.rint(ticks).astype(np.int64)
        selected = np.clip(queries, 0, n-1)
        max_age_ticks = temporal['selection'][selection_key]['max_age_s']*50
        valid = (queries >= 0) & ((queries-selected) <= max_age_ticks+1e-10)
        if future_only:
            valid &= selected > anchors
        actual_valid = np.array([row[channel+'_valid'] for row in windows], dtype=bool)
        np.testing.assert_array_equal(actual_valid, valid, err_msg=channel+' validity')
        actual_indices = np.array([row[channel+'_indices'] for row in windows])
        np.testing.assert_array_equal(
            actual_indices[valid], native_rows[selected][valid], err_msg=channel+' frames'
        )
        if not np.isin(actual_indices, native_rows).all():
            raise ValueError(channel+' index crosses an episode boundary')
        actual_times = np.array([row[channel+'_actual_times'] for row in windows])
        np.testing.assert_array_equal(
            actual_times[valid], ((selected-anchors)/50)[valid], err_msg=channel+' relative times'
        )
    full_count = max(0, n-30)
    for row in windows[:full_count]:
        if not all(row['world_action_group_valid']) or not all(row['eligibility'].values()):
            raise ValueError('complete future window lost world/policy eligibility')
    return full_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--asset-path-map')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=['HSI_boxing', 'HOI_double_desk'])
    args = parser.parse_args()
    asset_map = load_asset_path_map(args.asset_path_map)
    config, payload = load_predictive_world_config(args.source_config, allow_existing_output_override=True)
    temporal = temporal_abi_from_data_config(config.data)
    # This diagnostic's complete-window count specifically describes 30/50 Hz.
    if (temporal['action_horizon'] != 30 or temporal['action_hz'] != 50
            or temporal['future_group_ends_s'] != [0.2, 0.4, 0.6]):
        raise ValueError('audit expects the SONIC 30-step/50-Hz source temporal ABI')
    source = {name: json.loads(resolve_asset_path(payload['data'][name], asset_map).read_text()) for name in (
        'measurement_registry', 'action_protocol_registry', 'state_schema_registry',
    )}
    for task in args.tasks:
        info, instruction, episodes, audit, statistics = inspect_dataset(
            args.data_root/task/'sonic_refpose_v3_1', task=task
        )
        registries = extend_registries(source, info, statistics, task=task, audit_identity=canonical_sha256(audit))
        schema = MeasurementSchema.from_mapping(MEASUREMENT, registries['measurement_registry'][MEASUREMENT])
        capability = SourceCapability.from_mapping(task, registries['capability_registry'][task])
        frames = full_windows = 0
        for episode in episodes:
            windows = episode_windows(episode, task=task, instruction=instruction, temporal=temporal,
                                      capability=capability, measurement_schema=schema)
            full_windows += check_episode(episode, windows, temporal)
            frames += len(windows)
            if (episode['episode']+1) % 10 == 0:
                print(json.dumps({'event': 'alignment_progress', 'task': task,
                                  'episodes': episode['episode']+1, 'frames': frames}), flush=True)
        # Recheck original numeric-file hashes: this audit must not mutate data.
        for path, expected in {ep['parquet']: ep['parquet_sha256'] for ep in episodes}.items():
            if file_sha256(Path(path)) != expected:
                raise ValueError('source parquet changed during alignment audit: '+path)
        print(json.dumps({
            'event': 'sonic_frame_alignment_passed', 'task': task, 'episodes': len(episodes),
            'frames': frames, 'full_30_step_windows': full_windows,
            'full_action_slots': full_windows*30,
            'frame_or_mask_mismatches': 0, 'invalid_first_action_in_full_windows': 0,
            'time_alignment': TIME_ALIGNMENT, 'dataset_audit_identity': canonical_sha256(audit),
            'source_config_sha256': file_sha256(args.source_config),
            'maximum_timestamp_rounding_error_s': max(ep['timestamp_max_rounding_error_s'] for ep in episodes),
            'source_parquet_unchanged': True,
        }), flush=True)


if __name__ == '__main__':
    main()
