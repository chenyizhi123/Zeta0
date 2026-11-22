import pytest

from psi.data.predictive_world.assets import verify_asset_boundary


def test_shared_links_are_followed_and_directory_cycles_are_bounded(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    target = shared / 'data'
    target.mkdir()
    (target / 'sample').write_bytes(b'data')
    (shared / 'public').symlink_to(target, target_is_directory=True)
    (target / 'loop').symlink_to(shared, target_is_directory=True)
    assert verify_asset_boundary([shared / 'public'], shared) == 3


def test_nested_external_link_is_rejected_even_when_currently_accessible(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    external = tmp_path / 'project'
    external.mkdir()
    (external / 'model').write_bytes(b'weights')
    data = shared / 'data'
    data.mkdir()
    (shared / 'public').symlink_to(data, target_is_directory=True)
    (data / 'nested').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes configured shared boundary'):
        verify_asset_boundary([shared / 'public'], shared)


def test_dangling_input_is_rejected(tmp_path):
    (tmp_path / 'broken').symlink_to(tmp_path / 'absent')
    with pytest.raises(FileNotFoundError):
        verify_asset_boundary([tmp_path / 'broken'], tmp_path)


def test_historical_path_text_is_not_rewritten_or_treated_as_active_input(tmp_path):
    metadata = tmp_path / 'source.json'
    contents = '{"original_path":"/unmounted/project/training/run"}'
    metadata.write_text(contents)
    assert verify_asset_boundary([metadata], tmp_path) == 1
    assert metadata.read_text() == contents
