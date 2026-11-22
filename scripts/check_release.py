"""Check the exact source-export selection, including vendored text, offline.

No matching credential values are printed. This is a boundary check, not proof
of redistribution rights or a complete secret detector.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

try:
    from scripts.audit_repository import CREDENTIAL_PATTERNS
    from scripts.export_source import source_files
except ModuleNotFoundError:  # Direct script invocation from any working directory.
    from audit_repository import CREDENTIAL_PATTERNS
    from export_source import source_files


FORBIDDEN_ROOTS = {
    "data", "models", "checkpoints", "posttrain_assets", "posttrain_runs",
    "eval_runs", "runs", "diagnostics", "deploy_packages", "paper", "INSPIRE.md",
}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".parquet", ".arrow",
    ".hdf5", ".h5", ".mcap", ".bag", ".pem", ".key", ".mp4", ".zip",
}


def check_release(root: Path) -> dict:
    root = root.resolve()
    paths = source_files(root)
    findings = []
    for relative in paths:
        name = relative.as_posix()
        g1_local = (
            name.startswith('real/g1/config/') and relative.suffix == '.json'
            and not relative.name.endswith('.example.json')
        ) or any(name.startswith('real/g1/' + prefix) for prefix in (
            'external_dependencies/', 'GR00T-WholeBodyControl/external_dependencies/',
            'GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/',
            'GR00T-WholeBodyControl/gear_sonic/data/',
        ))
        if (g1_local or '.local' in relative.parts or relative.parts[0] in FORBIDDEN_ROOTS
                or relative.suffix.lower() in FORBIDDEN_SUFFIXES
                or (relative.name.startswith(".env") and relative.name not in {".env.sample", ".env.example"})
                or relative.name.endswith((".local.json", ".local.yaml"))):
            findings.append({"path": name, "kind": "local_artifact"})
        try:
            contents = (root / relative).read_text(encoding="utf-8")
        except UnicodeError:
            continue  # Images and other reviewed source assets remain allowed.
        for kind, pattern in CREDENTIAL_PATTERNS.items():
            for number, line in enumerate(contents.splitlines(), 1):
                if re.search(pattern, line):
                    findings.append({"path": name, "line": number, "kind": kind})
    return {"source_files": len(paths), "findings": findings}


def main() -> int:
    report = check_release(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    return int(bool(report["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
