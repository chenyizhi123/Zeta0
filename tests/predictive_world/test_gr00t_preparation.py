import copy
import json
from pathlib import Path

import pytest

from psi.config.predictive_world import predictive_world_config_from_payload
from psi.data.predictive_world.action_labels import COMPONENT_FIELDS
from psi.data.predictive_world.gr00t_preparation import shorten_action_window


def test_horizon_conversion_preserves_future_causal_and_component_semantics():
    row = {"action_indices": list(range(51, 101)), "action_valid": [True] * 43 + [False] * 7,
           "action_actual_times": [i / 50 for i in range(1, 51)],
           "future_visual_indices": [60, 70, 80], "future_measurement_indices": list(range(51, 81)),
           "world_action_group_valid": [True, True, True], "state_indices": [50],
           "split": "validation", "episode_id": "task/episode_000002", "eligibility": {"policy": False}}
    for field in COMPONENT_FIELDS:
        row[field] = {"base": copy.deepcopy(row[field.removesuffix("_by_component")]),
                      "arms": copy.deepcopy(row[field.removesuffix("_by_component")])}
    original = copy.deepcopy(row)
    result = shorten_action_window(row, source_horizon=50)
    assert row == original
    for key in row:
        if key in COMPONENT_FIELDS:
            assert all(result[key][name] == value[:40] for name, value in row[key].items())
        elif key in ("action_indices", "action_valid", "action_actual_times"):
            assert result[key] == row[key][:40]
        elif key != "eligibility":
            assert result[key] == row[key]
    assert "eligibility" not in result
    with pytest.raises(ValueError, match="shorter"):
        shorten_action_window(row, source_horizon=30)
    with pytest.raises(ValueError, match="temporal ABI"):
        shorten_action_window(row, source_horizon=49)


def test_template_resolves_native_joint_training_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/gr00t_n1d7/joint.template.json").read_text())
    for key, value in payload["data"].items():
        if isinstance(value, str) and value.startswith("REQUIRED_"):
            payload["data"][key] = str(tmp_path / key)
    payload["model"].update(qwen_model=str(tmp_path / "backbone"), gr00t_model=str(tmp_path / "base"),
                            gr00t_config_sha256="a" * 64)
    payload["runtime"]["output_dir"] = str(tmp_path / "run")
    config, _ = predictive_world_config_from_payload(payload)
    assert config.training.joint_from_gr00t and not config.training.joint_from_scratch
    assert config.training.forecast_control_reader == "both"
    assert config.training.latent_prediction_space == "raw"
    assert config.data.action_horizon == 40 and config.model.action_solver_steps == 4
    assert config.runtime.world_size == 2 and config.training.micro_batch == 256
    assert config.training.effective_global_batch == 512 and config.training.max_updates == 40000


def test_prepare_release_rebinds_index_and_never_reuses_old_features(tmp_path, monkeypatch):
    from test_sonic_joint_training import _two_tasks
    from psi.data.predictive_world.sonic_joint_training import prepare_joint_release
    from psi.data.predictive_world.cache import canonical_sha256
    from psi.data.predictive_world.index import verify_predictive_world_index
    import psi.data.predictive_world.gr00t_preparation as preparation
    import psi.data.lerobot.action_aligned_cache as aligned
    import psi.data.predictor_pretrain.runtime as runtime
    import transformers

    manifest, entries = _two_tasks(tmp_path)
    source_dir = tmp_path / "source"
    source = prepare_joint_release(manifest, {e["task"]: e["config"] for e in entries}, source_dir,
        expected_tasks=("task_a", "task_b"), expected_episodes=2, validation_fraction=.5)
    source["data"]["current_visual_pooling"] = "content_area_v1"
    source["identity"] = canonical_sha256({k: v for k, v in source.items() if k != "identity"})
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{}')
    # Backbone byte extraction is tested with real safetensors separately. This
    # boundary exercises the actual publisher, registry hashes and config parser.
    monkeypatch.setattr(preparation, "verify_backbone_export", lambda *a, **k: {"fixture": "new"})
    from types import SimpleNamespace
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(chat_template="fixture template"))
    monkeypatch.setattr(aligned, "build_action_aligned_vlm_processor_fingerprint", lambda p: {"fixture": "processor"})
    monkeypatch.setattr(runtime, "fingerprint_model_directory", lambda path: "new-backbone")
    template = json.loads((Path(__file__).resolve().parents[2] / "configs/gr00t_n1d7/joint.template.json").read_text())
    output = tmp_path / "target"
    preparation.prepare_training(source_path, template, base=base, backbone=tmp_path / "backbone",
                                 output=output, run_output=tmp_path / "train")
    meta = verify_predictive_world_index(output / "predictive_world_index.jsonl")
    assert meta["rows"] == 160
    assert len(meta["temporal_abi"]["action_offsets_s"]) == 40
    assert not (output / "feature_cache").exists()
    assert json.loads((output / "feature_provenance.json").read_text())["encoder_id"] == "qwen3vl:new-backbone"
    assert json.loads((output / "feature_provenance.json").read_text())["video_decoder"] == "torchcodec"
    config = json.loads((output / "train.json").read_text())
    assert config["data"]["source_sampling_exponent"] == 0
    assert list(json.loads((output / "measurement_registry.json").read_text())) == ["sonic_measured_state64_v31"]
    old_rows = [json.loads(line) for line in (source_dir / "predictive_world_index.jsonl").read_text().splitlines()]
    new_rows = [json.loads(line) for line in (output / "predictive_world_index.jsonl").read_text().splitlines()]
    for old, new in zip(old_rows, new_rows):
        assert old["split"] == new["split"] and old["episode_id"] == new["episode_id"]
        assert old["future_visual_indices"] == new["future_visual_indices"]
        assert old["action_indices"][:40] == new["action_indices"]
