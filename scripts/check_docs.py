"""Offline check of inline Markdown links in Zeta's maintained documentation.

Checks local files and ATX heading anchors, not external URLs, reference-style
links or HTML anchors. Upstream/vendor docs are not recursively checked.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DOC_ROOTS = (
    'docs/upstream',
    'real/g1/external_dependencies',
    'real/g1/GR00T-WholeBodyControl/external_dependencies',
    'real/g1/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty',
    'real/g1/GR00T-WholeBodyControl/gear_sonic/data',
)


def maintained_document(path, root=ROOT):
    return not any(path.is_relative_to(root / name) for name in EXTERNAL_DOC_ROOTS)

ENTRY_DOCS = (
    "README.md",
    "docs/README.md",
    "docs/reproducibility.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md",
    "configs/predictive_world/README.md",
    "real/README.md",
    "configs/psi0/README.md",
    "configs/gr00t_n1d7/README.md",
)
LINK = re.compile(r"\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)\)")


def prose_lines(text: str):
    """Yield numbered lines outside fenced blocks (backticks or tildes)."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = marker
            elif (
                marker[0] == fence[0]
                and len(marker) >= len(fence)
                and not suffix.strip()
            ):
                fence = None
            continue
        if fence is None:
            yield number, line


def heading_anchors(text: str) -> set[str]:
    anchors = set()
    for _, line in prose_lines(text):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            slug = re.sub(r"[^\w\- ]", "", match[1].lower()).replace(" ", "-")
            candidate = slug
            suffix = 0
            while candidate in anchors:
                suffix += 1
                candidate = f"{slug}-{suffix}"
            anchors.add(candidate)
    return anchors


def check_document(path: Path, root: Path) -> list[str]:
    errors = []
    root = root.resolve()
    for number, line in prose_lines(path.read_text(encoding="utf-8")):
        for match in LINK.finditer(line):
            link = match[1].removeprefix("<").removesuffix(">")
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            relative = unquote(parsed.path)
            target = (
                (root / relative.lstrip("/"))
                if relative.startswith("/")
                else (path.parent / relative if relative else path)
            ).resolve()
            reason = None
            if not target.is_relative_to(root):
                reason = "target escapes repository"
            elif not target.exists():
                reason = "missing target"
            elif (
                parsed.fragment and target.suffix.lower() == ".md" and target.is_file()
            ):
                if unquote(parsed.fragment) not in heading_anchors(
                    target.read_text(encoding="utf-8")
                ):
                    reason = "missing heading anchor"
            if reason:
                errors.append(f"{path.relative_to(root)}:{number}: {reason}: {link}")
    return errors


def main() -> int:
    paths = [ROOT / name for name in ENTRY_DOCS]
    paths.extend(sorted((ROOT / "docs/maintenance").glob("*.md")))
    paths.extend(sorted(path for path in (ROOT / "docs").glob("*.md") if path not in paths))
    paths.append(ROOT / ".github/pull_request_template.md")
    discovered = list(ROOT.glob("*.md"))
    for directory in ("docs", "configs", "scripts", "real", "examples"):
        discovered.extend((ROOT / directory).rglob("*.md"))
    paths.extend(sorted(path for path in discovered if path not in paths
                        and maintained_document(path)))
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: missing maintained document")
        else:
            errors.extend(check_document(path, ROOT))
    for error in errors:
        print(error)
    print(f"Checked {len(paths)} maintained documents: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
