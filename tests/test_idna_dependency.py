"""Check bounded IDNA validation and real HTTP URL preparation without network."""

import idna
import pytest
import requests
from yarl import URL


@pytest.mark.parametrize("entry", [idna.encode, idna.alabel, idna.check_label])
def test_oversized_context_label_rejected_before_context_scan(monkeypatch, entry):
    original = idna.core.valid_contexto
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(idna.core, "valid_contexto", observe)
    # The early cap is domain-sized for UTS 46 compatibility, not 63 characters.
    # A bounded 256-character fixture exceeds it without a DoS-sized payload.
    with pytest.raises(idna.IDNAError):
        entry("\u0660" * 256)
    assert calls == []


@pytest.mark.parametrize("host,ascii_host", [
    ("example.test", "example.test"),
    ("bücher.example", "xn--bcher-kva.example"),
    ("BÜCHER.example", "xn--bcher-kva.example"),
])
def test_http_libraries_preserve_hostname_encoding(host, ascii_host):
    url = f"https://{host}/path?value=1"
    expected = f"https://{ascii_host}/path?value=1"
    assert requests.Request("GET", url).prepare().url == expected
    assert str(URL(url)) == expected


def test_unicode_hostname_roundtrip():
    encoded = idna.encode("bücher.example")
    assert encoded == b"xn--bcher-kva.example"
    assert idna.decode(encoded) == "bücher.example"
