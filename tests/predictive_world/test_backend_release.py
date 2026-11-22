"""Public templates and native-source resolution across installation layouts."""

import json
from pathlib import Path
import subprocess

import pytest

from psi.config.predictive_world import predictive_world_config_from_payload
from psi.models.predictive_world.gr00t import native_source


@pytest.mark.parametrize("backend", ["psi0", "gr00t_n1d7"])
def test_public_joint_recipe_validates_and_binds_backend_identity(tmp_path, backend):
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / f"configs/{backend}/joint.template.json").read_text())
    for section in ("data", "model", "runtime"):
        for key, value in payload[section].items():
            if isinstance(value, str) and value.startswith("REQUIRED_"):
                payload[section][key] = ("a" * 64 if key.endswith("sha256") else str(tmp_path / key))
    config, _ = predictive_world_config_from_payload(payload)
    assert config.training.latent_prediction_space == "raw"
    assert config.training.objective_contract_version == 6
    assert config.training.joint_from_gr00t == (backend == "gr00t_n1d7")
    # The parser must reject merely relabeling a prepared host as the other one.
    payload["model"]["host_architecture"] = "zeta_native" if backend == "gr00t_n1d7" else "gr00t_n1d7"
    with pytest.raises(ValueError):
        predictive_world_config_from_payload(payload)


def test_missing_installed_native_source_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ZETA0_GR00T_SOURCE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="ZETA0_GR00T_SOURCE"):
        native_source()


def test_native_source_refuses_parent_repository_and_wrong_revision(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "child"
    (source / "gr00t").mkdir(parents=True)
    monkeypatch.setenv("ZETA0_GR00T_SOURCE", str(source))
    with pytest.raises(ValueError, match="own Git checkout"):
        native_source()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "-c", "core.hooksPath=/dev/null",
                    "commit", "--allow-empty", "-qm", "fixture"], check=True)
    with pytest.raises(ValueError, match="pinned revision"):
        native_source()
