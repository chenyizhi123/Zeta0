"""The public CLI and preparer use the same native v3 numerical adapter."""
import json

import pytest

from psi.framework.data import inspect_data
from test_sonic_lerobot_post_training import recording
from test_real_sonic_research_export import source as export_source_fixture  # noqa: F401 - shared fixture


def test_full_cli_data_audit_preserves_splits_outcomes_and_source(tmp_path):
    recording(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = inspect_data(str(tmp_path), full=True, task="custom", validation_fraction=.5, seed=0)
    assert report["numerical_contract_verified"]
    assert not report["training_admission_decided"] and not report["video_decoding_verified"]
    assert report["outcomes"] == {"unknown": 2}
    assert len(report["audit"]["train_episodes"]) == len(report["audit"]["validation_episodes"]) == 1
    assert not report["audit"]["time_alignment"]["physical_receive_clocks_available"]
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_full_cli_data_audit_uses_existing_schema_rejection(tmp_path):
    recording(tmp_path)
    path = tmp_path / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["names"][0] = "measured_joint_not_command"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="requires audited"):
        inspect_data(str(tmp_path), full=True)


def test_public_package_inspector_uses_immutable_source_verification(request):
    from psi.deploy.research_inference_weights import export_real_sonic_research_policy
    from psi.framework.bundle import inspect_bundle
    source = request.getfixturevalue("export_source_fixture")
    export_real_sonic_research_policy(source["checkpoint"], source["config"], source["output"])
    report = inspect_bundle(str(source["output"]), full=True)
    assert report["source_contract_verified"] and report["hashes_verified"]
    assert not report["legacy_package"] and not report["hardware_acceptance_verified"]
    path = source["output"] / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["encoder_id"] = "different-encoder"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="encoder id mismatch"):
        inspect_bundle(str(source["output"]), full=True)
