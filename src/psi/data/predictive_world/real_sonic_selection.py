"""Auditable real-SONIC sample exclusions matching the live input contract."""
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.action_labels import materialize_action_labels
from psi.data.predictive_world.real_sonic_post_training import (
    MOTION_TOKEN_LIMIT, MOTION_TOKEN_STEP, STATE_COLUMNS, ACTION_COLUMNS,
)

REASON_TEXT = {
    "current_state_unavailable": "当前状态无效或超过组合状态40ms时效，不能作为真机当前输入",
    "state_component_stale": "身体或任一手的实际反馈超过40ms，组合时间max掩盖了较旧分量",
    "state_component_future": "身体或手反馈晚于决策时刻，会引入未来信息",
    "current_state_nonfinite": "当前实测状态含非有限值",
    "current_hand_out_of_range": "当前BrainCo实测位置不在[0,1]",
    "current_camera_unavailable": "至少一路当前相机无效、缺失或超过80ms时效",
    "current_camera_future": "当前图像晚于决策时刻，会引入未来信息",
    "action_nonfinite": "有效动作标签含非有限值",
    "motion_token_changed_by_deployment": "有效motion token超出±0.625或不在1/16格点，部署量化/限幅会改变标签",
    "action_hand_out_of_range": "有效BrainCo手目标不在[0,1]，部署限幅会改变标签",
    "known_failed_episode": "该段录制已有明确失败标签",
    "snapshot_nonfinite_time": "采集快照的实际接收时间不是有限值",
    "snapshot_future_input": "采集快照含决策时刻尚未接收到的输入",
    "snapshot_stale": "采集快照超过原采集器200ms可用性范围",
    "snapshot_anchor_mismatch": "采集快照的相机anchor与ego接收时间不一致",
    "snapshot_camera_skew": "原始三路相机配对超过采集器25ms偏差范围",
    "snapshot_body_offset": "身体反馈相对相机anchor偏差超过采集器40ms范围",
    "snapshot_hand_offset": "实际左右手反馈相对相机anchor偏差超过采集器50ms范围",
}


def selection_contract(data, audit, *, policy=None):
    if data.state_history_count != 1 or data.observed_visual_mode != "current_multiview":
        raise ValueError("deploy_input_v1 selection requires one current state and current cameras")
    mode = getattr(data, "real_observation_mode", None)
    policy = ("collector_snapshot_v1" if mode is not None else "deploy_input_v1") if policy is None else policy
    if policy not in ("deploy_input_v1", "collector_snapshot_v1") or (policy == "collector_snapshot_v1") != (mode is not None):
        raise ValueError("sample selection policy must agree with its observation mode")
    result = dict(format="zeta0.real_sonic_sample_selection.v1", policy=policy,
        dataset_audit_identity=canonical_sha256(audit), state_history_count=1,
        state_component_max_age_s=data.state_max_age_s, camera_max_age_s=data.visual_max_age_s,
        motion_token_limit=MOTION_TOKEN_LIMIT, motion_token_step=MOTION_TOKEN_STEP,
        time_tolerance_s=1e-9, value_tolerance=1e-6,
        required_cameras=["observation.images.ego_view", "observation.images.left_wrist", "observation.images.right_wrist"],
        repeated_frames="keep_when_causal_and_within_age", gaps="keep_existing_contiguous_segments",
        unknown_success="retain_without_claiming_success", known_failure="exclude",
        invalid_future_slots="keep_existing_false_masks", normalization="preserve_preselection_train_frame_statistics",
        selection_code_sha256=file_sha256(Path(__file__)))
    if mode is not None:
        from psi.real_observation_contract import validate_observation_contract
        contract = validate_observation_contract(data.real_observation_contract)
        if audit.get("real_observation_contract") != contract:
            raise ValueError("collector selection must bind the actual source observation contract")
        result.update(collector_observation_contract=contract, observation_delay="native_measured_jitter")
        if getattr(data, "real_snapshot_max_age_s", None) is not None:
            result["real_snapshot_max_age_s"] = data.real_snapshot_max_age_s
    if getattr(data, "real_action_projection", None) is not None:
        from psi.real_observation_contract import validate_action_projection
        projection = validate_action_projection(data.real_action_projection)
        if audit.get("real_action_projection") != projection:
            raise ValueError("action selection must bind the actual recorded projection")
        result.update(real_action_projection=projection, motion_token_min=projection["motion_token_min"],
                      motion_token_max=projection["motion_token_max"], motion_token_step=projection["motion_token_step"])
    return result


def _immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable selection artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


class RealSonicSampleSelector:
    def __init__(self, dataset, contract):
        self.dataset = Path(dataset).resolve()
        self.contract = contract
        self.cache = OrderedDict()

    def _episode(self, row):
        import pyarrow.parquet as pq
        match = re.search(r"/episode_(\d+)/segment_", row["episode_id"])
        if match is None:
            raise ValueError("selection requires an audited real-SONIC episode/segment identity")
        episode = int(match.group(1))
        key = (episode, row["materialization"]["data_path"])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        quality = json.loads((self.dataset / f"meta/chenyz_episode_{episode:06d}.json").read_text())
        timing = [json.loads(line) for line in (self.dataset / quality["timing_file"]).read_text().splitlines() if line]
        table = pq.read_table(key[1], columns=[*STATE_COLUMNS, *ACTION_COLUMNS])
        states = np.concatenate([np.asarray(table[name].to_pylist(), dtype=np.float32) for name in STATE_COLUMNS], axis=1)
        actions = np.concatenate([np.asarray(table[name].to_pylist(), dtype=np.float32) for name in ACTION_COLUMNS], axis=1)
        if states.shape != (len(timing), 41) or actions.shape != (len(timing), 76):
            raise ValueError("selection source numeric rows and timing sidecar disagree")
        value = (quality, timing, states, actions)
        self.cache[key] = value
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return value

    def reasons(self, row):
        quality, timing, states, actions = self._episode(row)
        contract = self.contract
        if (contract.get("real_action_projection") is not None
                and row["materialization"].get("real_action_projection") != contract["real_action_projection"]):
            raise ValueError("indexed action projection disagrees with its source-bound selection contract")
        epsilon = contract["time_tolerance_s"]
        tolerance = contract["value_tolerance"]
        result = []
        def add(code, **evidence):
            reason = REASON_TEXT[code]
            if code == "motion_token_changed_by_deployment" and contract.get("real_action_projection") is not None:
                reason = "有效motion token超出录制来源绑定的执行范围或FSQ格点，部署投影会改变标签"
            result.append(dict(code=code, reason=reason, evidence=evidence))
        if contract["policy"] == "collector_snapshot_v1":
            from psi.real_observation_contract import snapshot_timing_violations
            index = row.get("collector_snapshot_row_index")
            if type(index) is not int or row["state_indices"] != [index]:
                raise ValueError("collector snapshot must preserve its recorded state row")
            offsets = row["materialization"].get("collector_video_offsets", {})
            for camera in contract["required_cameras"]:
                if row["history_visual_indices_by_camera"][camera][-1] != index + offsets.get(camera, 0):
                    raise ValueError("collector snapshot images must preserve the same recorded row")
            sample = timing[index]
            if quality.get("observation_contract") != contract["collector_observation_contract"] or sample.get("contract") != contract["collector_observation_contract"]:
                raise ValueError("collector snapshot source limits disagree with selection provenance")
            hands = sample["hand_state"].get("channel_received_monotonic", [sample["hand_state"]["received_monotonic"]] * 2)
            violations = snapshot_timing_violations(
                decision_time=sample["decision_monotonic"], camera_anchor_time=sample["anchor_monotonic"],
                camera_times={camera: sample["cameras"][camera.removeprefix("observation.images.")]["received_monotonic"]
                              for camera in contract["required_cameras"]},
                body_time=sample["body"]["received_monotonic"], hand_times=hands,
                contract=contract["collector_observation_contract"],
                snapshot_max_age_s=contract.get("real_snapshot_max_age_s"))
            for violation in violations:
                add(violation["code"], **violation["evidence"])
            if not np.isfinite(states[index]).all():
                add("current_state_nonfinite", selected_state_row=index)
            if np.any((states[index, 29:] < -tolerance) | (states[index, 29:] > 1 + tolerance)):
                add("current_hand_out_of_range", selected_state_row=index)
        else:
            if len(row["state_indices"]) != 1 or len(row["state_valid"]) != 1:
                raise ValueError("selection requires exactly one observed state")
            if not row["state_valid"][0]:
                add("current_state_unavailable", valid=False, max_age_ms=contract["state_component_max_age_s"] * 1000)
            else:
                index = row["state_indices"][0]
                sample = timing[index]
                anchor = timing[0]["decision_monotonic"] + row["anchor_time"]
                hands = sample["hand_state"].get("channel_received_monotonic")
                hands = [sample["hand_state"]["received_monotonic"]] * 2 if hands is None else hands
                if len(hands) != 2:
                    raise ValueError("selection requires two BrainCo feedback times")
                times = [sample["body"]["received_monotonic"], *hands]
                ages = dict(zip(["body", "left_hand", "right_hand"], [anchor - t for t in times]))
                if any(age < -epsilon for age in ages.values()):
                    add("state_component_future", selected_state_row=index, ages_ms={k: v * 1000 for k, v in ages.items()})
                if any(age > contract["state_component_max_age_s"] + epsilon for age in ages.values()):
                    add("state_component_stale", selected_state_row=index, ages_ms={k: v * 1000 for k, v in ages.items()},
                        max_age_ms=contract["state_component_max_age_s"] * 1000)
                if not np.isfinite(states[index]).all():
                    add("current_state_nonfinite", selected_state_row=index)
                if np.any((states[index, 29:] < -tolerance) | (states[index, 29:] > 1 + tolerance)):
                    add("current_hand_out_of_range", selected_state_row=index)
            unavailable, future = [], []
            for camera in contract["required_cameras"]:
                valid = row.get("history_visual_valid_by_camera", {}).get(camera, [False])[-1]
                actual = row.get("history_visual_actual_times_by_camera", {}).get(camera, [float("nan")])[-1]
                if not valid or not np.isfinite(actual) or actual < -contract["camera_max_age_s"] - epsilon:
                    unavailable.append(camera)
                elif actual > epsilon:
                    future.append(camera)
            if unavailable:
                add("current_camera_unavailable", cameras=unavailable, max_age_ms=contract["camera_max_age_s"] * 1000)
            if future:
                add("current_camera_future", cameras=future)
        selected, selected_valid = materialize_action_labels(row, lambda indices: actions[np.asarray(indices)])
        slots = np.arange(len(selected))
        if not np.isfinite(selected[selected_valid]).all():
            add("action_nonfinite", valid_action_slots=slots[selected_valid.any(axis=-1)].tolist())
        token = selected[:, :64]
        lower = contract.get("motion_token_min", -contract["motion_token_limit"])
        upper = contract.get("motion_token_max", contract["motion_token_limit"])
        bad = np.any(selected_valid[:, :64] & ((token < lower - tolerance) | (token > upper + tolerance)
            | (np.abs(token / contract["motion_token_step"] - np.rint(token / contract["motion_token_step"])) > tolerance)), axis=1)
        if bad.any():
            motion_indices = row.get("action_indices_by_component", {}).get("motion_token", row["action_indices"])
            add("motion_token_changed_by_deployment", action_slots=slots[bad].tolist(),
                source_action_rows=np.asarray(motion_indices)[slots[bad]].tolist(),
                limit=contract["motion_token_limit"], step=contract["motion_token_step"])
        hands = selected[:, 64:]
        bad = np.any(selected_valid[:, 64:] & ((hands < -tolerance) | (hands > 1 + tolerance)), axis=1)
        if bad.any():
            add("action_hand_out_of_range", action_slots=slots[bad].tolist())
        if quality.get("task_success") is False:
            add("known_failed_episode")
        return result

    def select(self, rows, artifacts):
        artifacts = Path(artifacts)
        ledger = artifacts / "sample_exclusions.jsonl"
        temporary = ledger.with_name(f".{ledger.name}.{uuid.uuid4().hex}.tmp")
        counts = Counter()
        reason_counts = Counter()
        world_reason_counts = Counter()
        primary_world = Counter()
        episodes = {}
        identities = hashlib.sha256()
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    counts["input_rows"] += 1
                    eligible = row["eligibility"]
                    for view, valid in eligible.items():
                        counts[f"input_{view}_rows"] += bool(valid)
                    reasons = self.reasons(row)
                    episode = episodes.setdefault(row["episode_id"], dict(retained=0, excluded=0))
                    if reasons:
                        codes = [entry["code"] for entry in reasons]
                        counts["excluded_rows"] += 1
                        episode["excluded"] += 1
                        reason_counts.update(codes)
                        for view, valid in eligible.items():
                            counts[f"excluded_{view}_rows"] += bool(valid)
                        if eligible["world"]:
                            world_reason_counts.update(codes)
                            primary_world[codes[0]] += 1
                        handle.write(json.dumps(dict(sample_id=row["sample_id"], episode_id=row["episode_id"],
                            anchor_time=row["anchor_time"], anchor_ordinal=row["anchor_ordinal"],
                            original_eligibility=eligible, reasons=reasons), ensure_ascii=False, allow_nan=False) + "\n")
                    else:
                        counts["retained_rows"] += 1
                        episode["retained"] += 1
                        identities.update((row["sample_id"] + "\n").encode())
                        for view, valid in eligible.items():
                            counts[f"retained_{view}_rows"] += bool(valid)
                        yield row
            if not counts["retained_world_rows"]:
                raise ValueError("selection removed every joint world sample; no training release published")
            if ledger.exists():
                if file_sha256(ledger) != file_sha256(temporary):
                    raise ValueError("immutable sample exclusions differ")
                temporary.unlink()
            else:
                temporary.replace(ledger)
            summary = dict(format="zeta0.real_sonic_sample_exclusions.v1", contract=self.contract,
                counts=dict(counts), reason_counts=dict(reason_counts), world_reason_counts=dict(world_reason_counts),
                world_primary_reason_counts=dict(primary_world), reason_counts_overlap=True,
                episode_segments=episodes, retained_sample_ids_sha256=identities.hexdigest(),
                exclusions_file=ledger.name, exclusions_sha256=file_sha256(ledger), reason_text=REASON_TEXT)
            _immutable(artifacts / "sample_selection.json", (json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())
        finally:
            temporary.unlink(missing_ok=True)

    def verify_existing(self, artifacts, index_metadata):
        artifacts = Path(artifacts)
        summary = json.loads((artifacts / "sample_selection.json").read_text())
        if (summary["contract"] != self.contract
                or summary["counts"]["retained_rows"] != index_metadata["rows"]
                or file_sha256(artifacts / summary["exclusions_file"]) != summary["exclusions_sha256"]):
            raise ValueError("existing selection evidence does not match this filtered index")
