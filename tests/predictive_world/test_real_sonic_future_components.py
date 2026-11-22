"""Real recorded-clock regression coverage for independently sampled labels."""

import copy
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from test_real_sonic_post_training import recording, registries

from psi.data.predictive_world.action_labels import materialize_action_labels
from psi.data.predictive_world.cache import canonical_sha256
from psi.data.predictive_world.contracts import NativeActionBatch
from psi.data.predictive_world.index import predictive_world_temporal_abi
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.real_sonic_post_training import (
    ACTION_COLUMNS, MEASUREMENT, PROTOCOL, SPACE,
    episode_windows, extend_registries, inspect_dataset,
)
from psi.models.predictive_world.action_expert import (
    committed_action_prefix_lengths, make_committed_action_prefix, sample_action_prefix_lengths,
)


COMPONENTS = {"motion_token": [0, 64], "left_hand": [64, 70], "right_hand": [70, 76]}
SNAPSHOT_FIELDS = (
    "state_indices", "state_valid", "state_actual_times", "history_visual_indices_by_camera",
    "history_visual_valid_by_camera", "history_visual_actual_times_by_camera",
    "action_indices", "action_valid", "action_actual_times", "future_measurement_indices",
    "future_measurement_valid", "future_measurement_actual_times", "world_action_group_valid", "eligibility",
)


def windows(root, *, independent=False):
    options = {"future_component_alignment": "independent_components_v1"} if independent else {}
    info, instruction, episodes, audit, stats = inspect_dataset(root, task="doll", **options)
    regs = extend_registries(registries(), info, stats, task="doll", audit_identity="fixture")
    schema = MeasurementSchema.from_mapping(MEASUREMENT, regs["measurement_registry"][MEASUREMENT])
    capability = SourceCapability.from_mapping("doll", regs["capability_registry"]["doll"])
    temporal = predictive_world_temporal_abi(
        history_visual_offsets_s=[-.4, -.2, 0], state_history_count=1, state_query_hz=50,
        future_group_ends_s=[.2, .4, .6], action_horizon=30, action_hz=50,
        visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04,
    )
    rows = [row for ep in episodes for row in episode_windows(
        ep, task="doll", instruction=instruction, temporal=temporal,
        capability=capability, measurement_schema=schema,
    )]
    return rows, episodes, audit, temporal


def asynchronous_recording(root):
    """Pause two independent body streams while hands continue to update."""
    recording(root)
    for ep in range(2):
        path = root / f"meta/chenyz_timing_{ep:06d}.jsonl"
        timing = [json.loads(line) for line in path.read_text().splitlines()]
        for index, row in enumerate(timing):
            decision = row["decision_monotonic"]
            motion = decision - .02
            if 10 <= index <= 14:
                motion = timing[9]["decision_monotonic"] - .02
            row["action"].update(
                body_received_monotonic=motion,
                hand_received_monotonic=decision - .005,
                hand_receive_monotonic=[decision - .005, decision - .045],
            )
            body = decision - .075
            if 6 <= index <= 14:
                body = timing[6]["decision_monotonic"] - .075
            row["body"]["received_monotonic"] = body
            row["hand_state"]["received_monotonic"] = decision - .005
        path.write_text("".join(json.dumps(row) + "\n" for row in timing))


def read_actions(path):
    table = pq.read_table(path, columns=list(ACTION_COLUMNS))
    return np.concatenate([np.asarray(table[name].to_pylist(), dtype=np.float32)
                           for name in ACTION_COLUMNS], axis=1)


def native(values, valid, *, times=None):
    horizon = len(values)
    return NativeActionBatch(
        values=torch.from_numpy(values.copy())[None], valid=torch.from_numpy(valid.copy())[None],
        times=(torch.arange(1, horizon + 1, dtype=torch.float32)[None] / 50 if times is None
               else torch.as_tensor(times, dtype=torch.float32)[None]),
        protocol_ids=(PROTOCOL,), native_dims=torch.tensor([76]), normalization_spaces=(SPACE,),
    )


def test_default_sampling_preserves_pre_migration_snapshot(tmp_path):
    recording(tmp_path)
    rows, _, audit, _ = windows(tmp_path)
    # Captured from the old implementation before adding the opt-in route.
    assert canonical_sha256([{key: row[key] for key in SNAPSHOT_FIELDS} for row in rows]) == (
        "2e157b0498510c920996525d02cc1930b0cedbb8954f87882b46bc97a9798857"
    )
    assert len(rows) == 180
    assert audit["time_alignment"]["action_clock"] == (
        "max(body_received_monotonic, hand_received_monotonic)"
    )
    assert all("action_indices_by_component" not in row for row in rows)


def test_independent_labels_use_actual_component_rows_and_clocks(tmp_path):
    asynchronous_recording(tmp_path)
    old_rows, _, _, _ = windows(tmp_path)
    rows, episodes, _, temporal = windows(tmp_path, independent=True)
    before = {path: path.read_bytes() for path in (tmp_path / "data").glob("*.parquet")}
    episodes_by_id = {
        f"doll/episode_{ep['episode']:06d}/segment_{ep['segment']:04d}": ep for ep in episodes
    }
    observed_fields = (
        "state_indices", "state_valid", "state_actual_times", "history_visual_indices_by_camera",
        "history_visual_valid_by_camera", "history_visual_actual_times_by_camera",
        "future_visual_indices", "future_visual_valid", "future_visual_actual_times",
    )
    timing_by_episode = {
        index: [json.loads(line) for line in (tmp_path / f"meta/chenyz_timing_{index:06d}.jsonl")
                .read_text().splitlines()] for index in range(2)
    }
    found_distinct_rows = False
    for old, row in zip(old_rows, rows, strict=True):
        assert canonical_sha256({key: row[key] for key in observed_fields}) == canonical_sha256(
            {key: old[key] for key in observed_fields}
        )
        assert row["materialization"]["future_component_alignment"] == "independent_components_v1"
        assert row["materialization"]["action_component_slices"] == COMPONENTS
        raw = read_actions(row["materialization"]["data_path"])
        values, valid = materialize_action_labels(row, lambda indices: raw[indices])
        assert values.shape == valid.shape == (30, 76)
        assert valid.dtype == np.bool_
        episode = episodes_by_id[row["episode_id"]]
        allowed = set(episode["row_indices"].tolist())
        timing = timing_by_episode[episode["episode"]]
        origin = timing[0]["decision_monotonic"]
        components_valid = []
        for component, (start, stop) in COMPONENTS.items():
            indices = np.asarray(row["action_indices_by_component"][component])
            mask = np.asarray(row["action_valid_by_component"][component], dtype=bool)
            actual = np.asarray(row["action_actual_times_by_component"][component])
            queries = np.asarray(temporal["action_offsets_s"])
            assert set(indices[mask]).issubset(allowed)
            assert (actual[mask] > 0).all()
            assert (actual[mask] <= queries[mask] + 1e-12).all()
            assert (queries[mask] - actual[mask] <= .04 + 1e-12).all()
            source_clock = np.asarray([
                item["action"]["body_received_monotonic"] if component == "motion_token"
                else item["action"]["hand_receive_monotonic"][0 if component == "left_hand" else 1]
                for item in timing
            ])
            np.testing.assert_allclose(actual[mask], source_clock[indices[mask]] - origin - row["anchor_time"],
                                       rtol=0, atol=1e-12)
            np.testing.assert_array_equal(valid[:, start:stop], np.broadcast_to(mask[:, None], (30, stop - start)))
            np.testing.assert_array_equal(values[mask, start:stop], raw[indices[mask], start:stop])
            components_valid.append(mask)
        np.testing.assert_array_equal(row["action_valid"], np.logical_and.reduce(components_valid))
        all_valid = np.asarray(row["action_valid"], dtype=bool)
        for slot in np.flatnonzero(all_valid):
            source_rows = {row["action_indices_by_component"][key][slot] for key in COMPONENTS}
            found_distinct_rows |= len(source_rows) == 3
        expected_world = [bool(all_valid[np.asarray(temporal["action_offsets_s"]) <= end + 1e-12].all())
                          for end in temporal["future_group_ends_s"]]
        assert row["world_action_group_valid"] == expected_world
        measurement_mask = np.asarray(row["future_measurement_valid"], dtype=bool)
        measurement_indices = np.asarray(row["future_measurement_indices"])
        assert set(measurement_indices[measurement_mask]).issubset(allowed)
        body_clock = np.asarray([item["body"]["received_monotonic"] for item in timing])
        np.testing.assert_allclose(
            np.asarray(row["future_measurement_actual_times"])[measurement_mask],
            body_clock[measurement_indices[measurement_mask]] - origin - row["anchor_time"], rtol=0, atol=1e-12,
        )
        if episode["segment"] == 0 and row["anchor_ordinal"] == len(episode["times"]) - 1:
            assert not np.asarray(row["action_valid_by_component"]["motion_token"]).any()
            assert not np.asarray(row["action_valid_by_component"]["left_hand"]).any()
            assert not np.asarray(row["action_valid_by_component"]["right_hand"]).any()
            assert not measurement_mask.any()
    assert found_distinct_rows
    assert all(path.read_bytes() == content for path, content in before.items())


def test_stale_body_labels_mask_independently_of_fresh_hands(tmp_path):
    asynchronous_recording(tmp_path)
    old_rows, _, _, _ = windows(tmp_path)
    rows, _, _, temporal = windows(tmp_path, independent=True)
    old = next(row for row in old_rows if row["episode_id"].endswith("episode_000000/segment_0000")
               and row["anchor_ordinal"] == 8)
    row = next(row for row in rows if row["sample_id"] == old["sample_id"])
    # At t0+60ms the motion token has stopped, but both hand streams are fresh.
    assert old["action_valid"][2]
    assert not row["action_valid_by_component"]["motion_token"][2]
    assert row["action_valid_by_component"]["left_hand"][2]
    assert row["action_valid_by_component"]["right_hand"][2]
    raw = read_actions(row["materialization"]["data_path"])
    values, valid = materialize_action_labels(row, lambda indices: raw[indices])
    assert not valid[2, :64].any() and valid[2, 64:].all()
    assert np.isfinite(values[valid]).all()
    assert not row["world_action_group_valid"][0]
    # An aggregate state timestamp previously hid this stopped body stream.
    assert old["future_measurement_valid"][2]
    assert not row["future_measurement_valid"][2]
    for candidate in rows:
        mask = np.asarray(candidate["future_measurement_valid"], dtype=bool)
        actual = np.asarray(candidate["future_measurement_actual_times"])
        queries = np.asarray(temporal["future_measurement_offsets_s"])
        assert (actual[mask] > 0).all()
        assert (actual[mask] <= queries[mask] + 1e-12).all()
        assert (queries[mask] - actual[mask] <= .04 + 1e-12).all()


def test_rtc_prefix_requires_contiguous_fully_valid_native_actions(tmp_path):
    asynchronous_recording(tmp_path)
    rows, _, _, _ = windows(tmp_path, independent=True)
    row = next(row for row in rows if row["episode_id"].endswith("episode_000000/segment_0000")
               and row["anchor_ordinal"] == 8)
    raw = read_actions(row["materialization"]["data_path"])
    values, valid = materialize_action_labels(row, lambda indices: raw[indices])
    actions = native(values, valid)
    actions.validate(batch_size=1, horizon=30)
    assert not valid[0].all()
    assert sample_action_prefix_lengths(actions, max_delay=15).item() == 0
    with pytest.raises(ValueError, match="fully valid"):
        make_committed_action_prefix(actions, torch.tensor([1]))
    # Also cover a later hole following two complete native steps.
    prefix_mask = np.ones_like(valid)
    prefix_mask[2, 0] = False
    actions = native(values, prefix_mask)
    for _ in range(100):
        assert sample_action_prefix_lengths(actions, max_delay=15).item() <= 1
    prefix = make_committed_action_prefix(actions, torch.tensor([1]))
    assert committed_action_prefix_lengths(prefix).item() == 1
    with pytest.raises(ValueError, match="fully valid"):
        make_committed_action_prefix(actions, torch.tensor([3]))
    partial = copy.deepcopy(prefix)
    partial.valid[0, 0, 70] = False
    with pytest.raises(ValueError, match="every native dimension"):
        committed_action_prefix_lengths(partial)
    hole = copy.deepcopy(prefix)
    hole.valid[0, 2, :] = True
    with pytest.raises(ValueError, match="left-contiguous"):
        committed_action_prefix_lengths(hole)


def test_partial_labels_retain_finite_real_aggregate_action_times(tmp_path):
    asynchronous_recording(tmp_path)
    rows, _, _, _ = windows(tmp_path, independent=True)
    row = next(row for row in rows if row["episode_id"].endswith("episode_000000/segment_0000")
               and row["anchor_ordinal"] == 8)
    raw = read_actions(row["materialization"]["data_path"])
    values, valid = materialize_action_labels(row, lambda indices: raw[indices])
    actions = native(values, valid, times=row["action_actual_times"])
    actions.validate(batch_size=1, horizon=30)
    partial = valid.any(axis=1) & ~valid.all(axis=1)
    assert partial.any()
    actual = np.asarray(row["action_actual_times"])
    assert np.isfinite(actual[valid.any(axis=1)]).all()
    for slot in np.flatnonzero(valid.any(axis=1)):
        expected = max(row["action_actual_times_by_component"][component][slot]
                       for component in COMPONENTS if row["action_valid_by_component"][component][slot])
        assert actual[slot] == expected


@pytest.mark.parametrize("change", [
    "missing_indices_component", "missing_mask_component", "missing_times_component",
    "short_indices", "short_mask", "short_times", "overlapping_slices", "bad_vector_shape",
    "negative_index", "nonboolean_mask", "invalid_valid_time", "inconsistent_complete_mask",
    "marker_without_component_fields", "unknown_alignment_marker",
])
def test_component_materialization_rejects_incomplete_or_malformed_contract(tmp_path, change):
    recording(tmp_path)
    rows, _, _, _ = windows(tmp_path, independent=True)
    row = copy.deepcopy(rows[8])
    if change == "missing_indices_component":
        del row["action_indices_by_component"]["left_hand"]
    elif change == "missing_mask_component":
        del row["action_valid_by_component"]["right_hand"]
    elif change == "missing_times_component":
        del row["action_actual_times_by_component"]["motion_token"]
    elif change == "short_indices":
        row["action_indices_by_component"]["motion_token"].pop()
    elif change == "short_mask":
        row["action_valid_by_component"]["left_hand"].pop()
    elif change == "short_times":
        row["action_actual_times_by_component"]["right_hand"].pop()
    elif change == "overlapping_slices":
        row["materialization"]["action_component_slices"]["left_hand"] = [63, 70]
    elif change == "negative_index":
        row["action_indices_by_component"]["motion_token"][0] = -1
    elif change == "nonboolean_mask":
        row["action_valid_by_component"]["left_hand"][0] = "false"
    elif change == "invalid_valid_time":
        slot = row["action_valid_by_component"]["right_hand"].index(True)
        row["action_actual_times_by_component"]["right_hand"][slot] = float("nan")
    elif change == "inconsistent_complete_mask":
        row["action_valid"][0] = not row["action_valid"][0]
    elif change == "marker_without_component_fields":
        for field in ("action_indices_by_component", "action_valid_by_component", "action_actual_times_by_component"):
            del row[field]
        del row["materialization"]["action_component_slices"]
    elif change == "unknown_alignment_marker":
        row["materialization"]["future_component_alignment"] = "unknown_components_v2"
    raw = read_actions(row["materialization"]["data_path"])
    reader = (lambda indices: raw[indices, :75]) if change == "bad_vector_shape" else (lambda indices: raw[indices])
    with pytest.raises(ValueError):
        materialize_action_labels(row, reader)


def test_legacy_action_materialization_retains_dense_row_semantics(tmp_path):
    recording(tmp_path)
    rows, _, _, _ = windows(tmp_path)
    row = rows[8]
    raw = read_actions(row["materialization"]["data_path"])
    values, valid = materialize_action_labels(row, lambda indices: raw[indices])
    expected_mask = np.broadcast_to(np.asarray(row["action_valid"])[:, None], (30, 76))
    np.testing.assert_array_equal(valid, expected_mask)
    np.testing.assert_array_equal(values[valid], raw[row["action_indices"]][valid])


def test_unknown_component_alignment_is_rejected_before_sampling(tmp_path):
    recording(tmp_path)
    with pytest.raises(ValueError):
        inspect_dataset(tmp_path, task="doll", future_component_alignment="misspelled_version")
