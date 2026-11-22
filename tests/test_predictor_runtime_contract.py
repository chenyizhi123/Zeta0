import json

import pytest

from psi.data.predictor_pretrain.runtime_contract import migrate_config, publish_config, resolve_timeouts


def test_equal_timeouts_for_old_and_new_configs():
    assert resolve_timeouts({}) == (600, 600)
    assert resolve_timeouts({"runtime_contract_version": "runtime_v2"}) == (600, 600)
    assert resolve_timeouts({"data_loader_timeout_seconds": 600, "process_group_timeout_seconds": 600}) == (600, 600)
    assert resolve_timeouts({"data_loader_timeout_seconds": 1200, "process_group_timeout_seconds": 1500}) == (1200, 1500)


@pytest.mark.parametrize("values", [
    {"data_loader_timeout_seconds": 600, "process_group_timeout_seconds": 599},
    {"data_loader_timeout_seconds": 600, "process_group_timeout_seconds": 1},
    {"data_loader_timeout_seconds": 1000},
    {"data_loader_timeout_seconds": 0},
    {"data_loader_timeout_seconds": float("nan")},
    {"process_group_timeout_seconds": float("inf")},
    {"process_group_timeout_seconds": -1},
    {"runtime_contract_version": "unknown"},
])
def test_invalid_timeouts_refused(values):
    with pytest.raises(ValueError):
        resolve_timeouts(values)


def test_explicit_migration_preserves_old_file_and_model_semantics(tmp_path):
    old = tmp_path / "stage_a_seed101_qk_norm_v1.json"
    new = tmp_path / "stage_a_seed101_qk_norm_v1_runtime_v2.json"
    old.write_text('{"predictor_stability_version":"qk_norm_v1","updates":30000,"teacher_sha256":"unchanged"}\n')
    before = old.read_bytes()
    migrate_config(old, new)
    result = json.loads(new.read_text())
    assert result == {**json.loads(before), "runtime_contract_version": "runtime_v2",
                      "data_loader_timeout_seconds": 600, "process_group_timeout_seconds": 600}
    assert old.read_bytes() == before
    published = new.read_bytes()
    migrate_config(old, new)
    assert new.read_bytes() == published
    with pytest.raises(ValueError, match="different output"):
        migrate_config(old, old)
    with pytest.raises(ValueError, match="immutable"):
        publish_config(new, {**result, "updates": 123})
    assert old.read_bytes() == before and new.read_bytes() == published
    assert not list(tmp_path.glob(".*"))


def test_failed_publication_leaves_no_partial_config(tmp_path, monkeypatch):
    import psi.data.predictor_pretrain.runtime_contract as runtime
    def fail(*args):
        raise OSError("simulated link failure")
    monkeypatch.setattr(runtime.os, "link", fail)
    with pytest.raises(OSError):
        publish_config(tmp_path / "config.json", {})
    assert list(tmp_path.iterdir()) == []
