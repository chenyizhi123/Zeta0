import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world.real_sonic_selection import RealSonicSampleSelector, selection_contract


def source(tmp_path):
    (tmp_path / "meta").mkdir()
    timing = [dict(decision_monotonic=1., body=dict(received_monotonic=.99),
                   hand_state=dict(received_monotonic=.995, channel_received_monotonic=[.99, .995]))]
    (tmp_path / "meta/timing.jsonl").write_text(json.dumps(timing[0]) + "\n")
    quality = dict(timing_file="meta/timing.jsonl", task_success=None)
    (tmp_path / "meta/chenyz_episode_000000.json").write_text(json.dumps(quality))
    values = dict(**{"observation.state": [0.] * 29, "observation.hand_q": [.2] * 12,
                    "action.motion_token": [.25] * 64, "action.hand_q": [.3] * 12})
    parquet = tmp_path / "episode.parquet"
    pq.write_table(pa.Table.from_pylist([values]), parquet)
    cameras = ["observation.images.ego_view", "observation.images.left_wrist", "observation.images.right_wrist"]
    row = dict(sample_id="sample", source_id="real/place_doll", episode_id="real/place_doll/episode_000000/segment_0000",
        anchor_time=0., anchor_ordinal=0, state_indices=[0], state_valid=[True], state_actual_times=[-.005],
        action_indices=[0] * 30, action_valid=[True] * 30,
        history_visual_valid_by_camera={name: [False, False, True] for name in cameras},
        history_visual_actual_times_by_camera={name: [float("nan"), float("nan"), -.02] for name in cameras},
        materialization=dict(data_path=str(parquet)), eligibility=dict(prior=True, world=True, policy=True))
    data = SimpleNamespace(state_history_count=1, observed_visual_mode="current_multiview", state_max_age_s=.04, visual_max_age_s=.08)
    contract = selection_contract(data, {"source": "fixture"})
    return row, values, timing, quality, contract


def selector_with_change(tmp_path, change):
    row, values, timing, quality, contract = source(tmp_path)
    if change == "missing_state": row["state_valid"][0] = False
    elif change == "old_body": timing[0]["body"]["received_monotonic"] = .95
    elif change == "old_left_hand": timing[0]["hand_state"]["channel_received_monotonic"][0] = .95
    elif change == "future_right_hand": timing[0]["hand_state"]["channel_received_monotonic"][1] = 1.001
    elif change == "missing_camera": row["history_visual_valid_by_camera"]["observation.images.left_wrist"][-1] = False
    elif change == "old_camera": row["history_visual_actual_times_by_camera"]["observation.images.right_wrist"][-1] = -.081
    elif change == "future_camera": row["history_visual_actual_times_by_camera"]["observation.images.ego_view"][-1] = .001
    elif change == "large_token": values["action.motion_token"][0] = .6875
    elif change == "off_grid_token": values["action.motion_token"][0] = .2501
    elif change == "bad_action_hand": values["action.hand_q"][0] = 1.1
    elif change == "bad_state_hand": values["observation.hand_q"][0] = -.1
    elif change == "known_failure": quality["task_success"] = False
    elif change == "repeated_frame": timing[0]["cameras"] = {"ego_view": {"repeated": True}}
    pq.write_table(pa.Table.from_pylist([values]), tmp_path / "episode.parquet")
    (tmp_path / "meta/timing.jsonl").write_text(json.dumps(timing[0]) + "\n")
    (tmp_path / "meta/chenyz_episode_000000.json").write_text(json.dumps(quality))
    return RealSonicSampleSelector(tmp_path, contract), row


@pytest.mark.parametrize("change,reason", [
    ("missing_state", "current_state_unavailable"), ("old_body", "state_component_stale"),
    ("old_left_hand", "state_component_stale"), ("future_right_hand", "state_component_future"),
    ("missing_camera", "current_camera_unavailable"), ("old_camera", "current_camera_unavailable"),
    ("future_camera", "current_camera_future"), ("large_token", "motion_token_changed_by_deployment"),
    ("off_grid_token", "motion_token_changed_by_deployment"), ("bad_action_hand", "action_hand_out_of_range"),
    ("bad_state_hand", "current_hand_out_of_range"), ("known_failure", "known_failed_episode"),
])
def test_excludes_actual_input_or_label_violations(tmp_path, change, reason):
    selector, row = selector_with_change(tmp_path, change)
    assert [r["code"] for r in selector.reasons(row)] == [reason]


def test_fresh_repeated_frames_and_unknown_success_are_kept(tmp_path):
    selector, row = selector_with_change(tmp_path, "repeated_frame")
    assert selector.reasons(row) == []
    assert selector.contract["unknown_success"] == "retain_without_claiming_success"


def test_ignored_invalid_action_slots_do_not_exclude_masked_labels(tmp_path):
    selector, row = selector_with_change(tmp_path, "large_token")
    row["action_valid"] = [False] * 30
    assert selector.reasons(row) == []


def test_selection_keeps_data_unchanged_and_records_all_overlapping_reasons(tmp_path):
    selector, good = selector_with_change(tmp_path, None)
    bad = copy.deepcopy(good)
    bad.update(sample_id="bad", anchor_ordinal=1)
    bad["state_valid"] = [False]
    bad["history_visual_valid_by_camera"]["observation.images.left_wrist"][-1] = False
    before = (tmp_path / "episode.parquet").read_bytes()
    retained = list(selector.select([good, bad], tmp_path))
    assert retained == [good]
    assert (tmp_path / "episode.parquet").read_bytes() == before
    summary = json.loads((tmp_path / "sample_selection.json").read_text())
    assert summary["counts"]["excluded_world_rows"] == 1
    assert summary["world_reason_counts"] == {"current_state_unavailable": 1, "current_camera_unavailable": 1}
    assert sum(summary["world_primary_reason_counts"].values()) == 1
    ledger = [json.loads(line) for line in (tmp_path / "sample_exclusions.jsonl").read_text().splitlines()]
    assert ledger[0]["sample_id"] == "bad" and len(ledger[0]["reasons"]) == 2
    selector.verify_existing(tmp_path, {"rows": 1})
    with pytest.raises(ValueError, match="does not match"):
        selector.verify_existing(tmp_path, {"rows": 2})


def test_empty_joint_release_fails_closed(tmp_path):
    selector, bad = selector_with_change(tmp_path, "missing_state")
    with pytest.raises(ValueError, match="every joint"):
        list(selector.select([bad], tmp_path))
    assert not (tmp_path / "sample_selection.json").exists()
