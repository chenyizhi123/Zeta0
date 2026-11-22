"""Calibration startup must accept masked futures from native 30 Hz data."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoProcessor

from psi.config.predictive_world import (
    PredictiveWorldConfig,
    PredictiveWorldDataConfig,
    PredictiveWorldRuntimeConfig,
    PredictiveWorldTrainingConfig,
)
from psi.data.predictive_world.dataset import PredictiveWorldOccurrenceDataset
from psi.data.predictive_world.index import build_episode_windows, publish_predictive_world_index
import psi.models.predictive_world.posttrain_transition as transition


@pytest.mark.parametrize("phase", ["joint", "world_model"])
def test_calibration_startup_uses_world_rows_with_partial_body_queries(tmp_path, monkeypatch, phase):
    from test_index import _capability, _episode, _provenance, _schema, _temporal

    temporal = _temporal(
        history_visual_offsets_s=(0.0,), state_history_count=1,
        action_horizon=50, action_hz=30.0,
        visual_max_age_s=0.08, state_max_age_s=0.04, action_max_age_s=0.04,
    )
    episode = _episode()
    for key in ("visual_timestamps", "state_timestamps", "action_timestamps"):
        episode[key] = [index / 30 for index in range(91)]
    rows = build_episode_windows(
        episode, history_visual_offsets_s=(0.0,), state_offsets_s=(0.0,),
        future_group_ends_s=(0.2, 0.4, 0.6),
        action_offsets_s=temporal["action_offsets_s"],
        future_measurement_offsets_s=temporal["future_measurement_offsets_s"],
        visual_max_age_s=0.08, state_max_age_s=0.04, action_max_age_s=0.04,
        capability=_capability(), measurement_schema=_schema(),
    )
    assert rows[0]["future_measurement_actual_times"][0] == 0
    assert not rows[0]["future_measurement_valid"][0]
    assert sum(rows[0]["future_measurement_valid"]) == 29
    path, _ = publish_predictive_world_index(
        tmp_path / "index.jsonl", rows=rows, provenance=_provenance(), temporal_abi=temporal,
    )
    # Keep the strict policy contract intact: this fixture reproduces the empty view.
    with pytest.raises(ValueError, match="complete-future"):
        PredictiveWorldOccurrenceDataset(
            path, None, split="train", seed=101, rank=0, world_size=1,
            start_occurrence=0, rank_batch_size=2, require_world_action=True,
            require_complete_future=True,
        )

    config = PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(
            index_path=str(path), history_visual_offsets_s=(0.0,), state_history_count=1,
            action_horizon=50, action_hz=30.0,
        ),
        training=PredictiveWorldTrainingConfig(
            phase=phase, micro_batch=2, max_updates=1,
            init_posttrain_migration={"calibration_batches": 1},
        ),
        runtime=PredictiveWorldRuntimeConfig(nodes=1, gpus_per_node=1),
    )
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    built = SimpleNamespace(
        model=model, checkpoint_modules={"world": model[0], "student": model[1]},
    )
    script = Path(__file__).resolve().parents[2] / "scripts/train/predictive_world/train.py"
    main = runpy.run_path(str(script))["main"]
    globals_ = main.__globals__
    # Replace asset/model loading only; run the real entrypoint, dataset and DataLoader.
    monkeypatch.setitem(globals_, "load_predictive_world_config", lambda *a, **k: (config, {}))
    monkeypatch.setitem(globals_, "_init_dist", lambda **k: (0, 1, torch.device("cpu")))
    monkeypatch.setitem(globals_, "assert_predictive_world_training_data_contract", lambda *a, **k:
        SimpleNamespace(capabilities={"source": _capability()}, feature_manifest={}, measurement_schemas={}))
    monkeypatch.setitem(globals_, "build_phase", lambda *a, **k: built)
    monkeypatch.setattr(AutoProcessor, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(chat_template="calibration fixture"))
    monkeypatch.setitem(globals_, "PredictiveWorldRowReader", lambda *a, **k:
        SimpleNamespace(read=lambda row, **k: row))
    monkeypatch.setitem(globals_, "PredictiveWorldTrainingCollator", lambda **k: list)
    monkeypatch.setattr(globals_["signal"], "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", [str(script), "--config", "unused.json", "--num-workers", "0"])

    class CalibrationReached(Exception):
        pass

    def check_calibration_input(built_arg, loader, **kwargs):
        assert built_arg is built and kwargs["batch_count"] == 1
        assert loader.dataset.view == "world" and loader.dataset.split == "train"
        assert loader.dataset.row_count == 1 and loader.dataset.start_occurrence == 0
        batch = next(iter(loader))
        assert len(batch) == 2
        for row in batch:
            assert row["future_measurement_valid"] == rows[0]["future_measurement_valid"]
            assert row["world_action_group_valid"] == [True, True, True]
        raise CalibrationReached

    # Stop at the calibration boundary: numerical calibration has its own model tests.
    monkeypatch.setattr(transition, "calibrate_posttrain", check_calibration_input)
    with pytest.raises(CalibrationReached):
        main()
