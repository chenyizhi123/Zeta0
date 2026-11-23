import runpy
from pathlib import Path
from zipfile import ZipFile
from zipfile import ZipInfo
import stat
import sys

import pytest

check_wheel = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_distribution.py")
)["check_wheel"]


@pytest.mark.parametrize(
    "extra", [None, "gr00t/model.py", "psi/../../secret", "/psi/file.py"]
)
def test_wheel_boundaries(tmp_path, extra):
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "zeta0_humanoid-0.0.0.dist-info/METADATA", "Name: zeta0-humanoid\nVersion: 0.0.0\n"
        )
        for name in (
            "psi/__init__.py",
            "psi/data/predictor_pretrain/runtime_contract.py",
            "zeta0/__init__.py",
        ):
            archive.writestr(name, "")
        for name in ("LICENSE", "NOTICE", "Apache-2.0.txt", "THIRD_PARTY_NOTICES.md"):
            archive.writestr("zeta0_humanoid-0.0.0.dist-info/licenses/" + name, "")
        if extra:
            archive.writestr(extra, "")
    if extra:
        with pytest.raises(ValueError, match="Unexpected wheel member"):
            check_wheel(wheel)
    else:
        check_wheel(wheel)


def test_missing_notices_refused(tmp_path):
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("zeta0_humanoid-0.0.0.dist-info/METADATA", "Name: zeta0-humanoid\n")
        archive.writestr("psi/__init__.py", "")
        archive.writestr("psi/data/predictor_pretrain/runtime_contract.py", "")
        archive.writestr("zeta0/__init__.py", "")
    with pytest.raises(ValueError, match="Missing attribution"):
        check_wheel(wheel)


def valid_members():
    return {
        'zeta0_humanoid-0.0.0.dist-info/METADATA': 'Name: zeta0-humanoid\nVersion: 0.0.0\n',
        'psi/__init__.py': '',
        'psi/data/predictor_pretrain/runtime_contract.py': '',
        'zeta0/__init__.py': '',
        **{f'zeta0_humanoid-0.0.0.dist-info/licenses/{name}': '' for name in
           ('LICENSE', 'NOTICE', 'Apache-2.0.txt', 'THIRD_PARTY_NOTICES.md')},
    }


@pytest.mark.parametrize('extra', [
    'psi/__init__.py', 'psi//extra.py', 'psi/./extra.py',
    'psi/..\\outside.py', 'psi/file:stream', 'psi/data',
])
def test_ambiguous_member_refused(tmp_path, extra):
    wheel = tmp_path / 'ambiguous.whl'
    with ZipFile(wheel, 'w') as archive:
        for name, content in valid_members().items():
            archive.writestr(name, content)
        if extra == 'psi/__init__.py':
            with pytest.warns(UserWarning, match='Duplicate name'):
                archive.writestr(extra, 'different bytes')
        else:
            archive.writestr(extra, '')
    with pytest.raises(ValueError, match='wheel member'):
        check_wheel(wheel)


def test_nested_metadata_refused(tmp_path):
    wheel = tmp_path / 'nested.whl'
    with ZipFile(wheel, 'w') as archive:
        for name, content in valid_members().items():
            archive.writestr('psi/' + name if '.dist-info/' in name else name, content)
    with pytest.raises(ValueError, match='metadata'):
        check_wheel(wheel)


def test_symlink_member_refused(tmp_path):
    wheel = tmp_path / 'symlink.whl'
    with ZipFile(wheel, 'w') as archive:
        for name, content in valid_members().items():
            archive.writestr(name, content)
        link = ZipInfo('psi/link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, '../outside')
    with pytest.raises(ValueError, match='wheel member'):
        check_wheel(wheel)


def test_explicit_directory_entries_accepted(tmp_path):
    wheel = tmp_path / 'directories.whl'
    with ZipFile(wheel, 'w') as archive:
        archive.writestr('psi/', '')
        archive.writestr('psi/data/', '')
        for name, content in valid_members().items():
            archive.writestr(name, content)
    check_wheel(wheel)


def test_cli_returns_failure_without_extracting(tmp_path, run_isolated_process):
    wheel = tmp_path / 'bad-cli.whl'
    with ZipFile(wheel, 'w') as archive:
        for name, content in valid_members().items():
            archive.writestr(name, content)
        archive.writestr('psi/./extra.py', '')
    checker = Path(__file__).resolve().parents[1] / 'scripts/check_distribution.py'
    result = run_isolated_process(
        [sys.executable, str(checker), str(wheel)], cwd=tmp_path, timeout=10,
    )
    assert result.returncode != 0
    assert 'Unexpected wheel member' in result.stderr
    assert set(tmp_path.iterdir()) == {wheel}
