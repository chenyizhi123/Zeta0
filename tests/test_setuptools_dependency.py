"""Exercise actual manifest exclusions across Unicode normalization forms."""

import unicodedata

import pytest
from setuptools.command.egg_info import FileList


@pytest.mark.parametrize('directive,path', [
    ('exclude café.txt', 'café.txt'),
    ('global-exclude café.txt', 'nested/café.txt'),
    ('recursive-exclude nested café.txt', 'nested/café.txt'),
    ('prune privé', 'privé/fixture.txt'),
])
@pytest.mark.parametrize('normalization', ['NFC', 'NFD'])
def test_manifest_exclusion_matches_unicode_forms(directive, path, normalization):
    filename = unicodedata.normalize(normalization, path)
    listing = FileList()
    listing.files = [filename, 'public.txt']
    listing.process_template_line(directive)
    assert listing.files == ['public.txt']
