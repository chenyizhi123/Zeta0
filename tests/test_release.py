import json
import subprocess

from scripts.check_release import check_release


def test_release_scans_vendor_credentials_and_rejects_unignored_artifacts(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    vendor = tmp_path / "third_party/vendor"
    vendor.mkdir(parents=True)
    secret = "ghp_" + "A" * 36
    (vendor / "config.py").write_text(f"TOKEN = '{secret}'\n")
    (tmp_path / "private.ckpt").write_bytes(b"checkpoint")
    (tmp_path / ".env.production").write_text("LOCAL=value\n")
    (tmp_path / ".env.sample").write_text("TOKEN=\n")
    report = check_release(tmp_path)
    assert {(f["path"], f["kind"]) for f in report["findings"]} == {
        ("third_party/vendor/config.py", "github_token"),
        ("private.ckpt", "local_artifact"),
        (".env.production", "local_artifact"),
    }
    assert secret not in json.dumps(report)


def test_release_respects_export_selection_and_checks_new_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.ckpt\n")
    (tmp_path / "private.ckpt").write_bytes(b"local checkpoint")
    (tmp_path / "core.py").write_text("# New source is included without staging.\n")
    report = check_release(tmp_path)
    assert report == {"source_files": 2, "findings": []}


def test_release_rejects_nested_g1_private_material_even_if_force_added(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    names = ('real/g1/config/local.json', 'real/g1/config/hand-hardware.json',
             'real/g1/.local/source.py', 'real/g1/external_dependencies/sdk/header.h',
             'real/g1/GR00T-WholeBodyControl/gear_sonic/data/mesh.stl')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    report = check_release(tmp_path)
    assert {f['path'] for f in report['findings']} == set(names)
