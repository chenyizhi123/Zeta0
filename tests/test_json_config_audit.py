import pytest

from scripts.audit_repository import json_config_loads, scan


@pytest.mark.parametrize("text", [
    '{"url":"https://example.invalid/a//b", /* comment */ "items":[1,2,],}',
    '// header\n{"url":"https://example.invalid/a//b", "items":[1,2]} // end',
])
def test_jsonc_preserves_strings_and_values(text):
    assert json_config_loads(text, comments=True) == {
        "url": "https://example.invalid/a//b", "items": [1, 2]
    }
    with pytest.raises(ValueError):
        json_config_loads(text)


def test_escaped_strings_are_not_rewritten():
    text = r'{"value":"a\"/* not comment */ ,]\\",}'
    assert json_config_loads(text, comments=True)["value"] == 'a"/* not comment */ ,]\\'


@pytest.mark.parametrize("text", [
    '[,]', '{,}', '[1,,]', '{"a":,}', '[1/* gap */2]', '[tru/* gap */e]',
    '/* unterminated', '{"a":1} garbage', '{a:1}', "{'a':1}",
    '[NaN]', '[Infinity]', '[-Infinity]', '["unterminated]',
])
def test_jsonc_does_not_repair_invalid_json(text):
    with pytest.raises(ValueError):
        json_config_loads(text, comments=True)


def test_dialect_selection_does_not_relax_data_json(tmp_path):
    editor = tmp_path / '.vscode'
    editor.mkdir()
    valid_jsonc = '{"value": [1,], // comment\n}'
    for path in (editor / 'settings.json', tmp_path / 'pyrightconfig.json', tmp_path / 'other.jsonc'):
        path.write_text(valid_jsonc)
    (tmp_path / 'data.json').write_text(valid_jsonc)
    (editor / 'launch.json').write_text('[,]')
    result = scan(tmp_path)
    assert result['counts']['jsonc'] == 4
    assert result['counts']['json'] == 1
    assert {(item['path'], item['kind']) for item in result['findings']} == {
        ('data.json', 'json_syntax'), ('.vscode/launch.json', 'jsonc_syntax')
    }
