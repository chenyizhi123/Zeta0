"""Versioned Stage-A runtime settings, independent of model architecture."""

import json
import math
import os
import tempfile
from pathlib import Path

RUNTIME_CONTRACT_VERSION = "runtime_v2"
DEFAULT_LOADER_TIMEOUT = 600.0
DEFAULT_PROCESS_GROUP_TIMEOUT = 600.0


def resolve_timeouts(config):
    version = config.get("runtime_contract_version")
    if version not in (None, RUNTIME_CONTRACT_VERSION):
        raise ValueError(f"unsupported runtime contract: {version}")
    loader = float(config.get("data_loader_timeout_seconds", DEFAULT_LOADER_TIMEOUT))
    communication = float(config.get("process_group_timeout_seconds", DEFAULT_PROCESS_GROUP_TIMEOUT))
    if not all(math.isfinite(value) and value > 0 for value in (loader, communication)):
        raise ValueError("loader and process-group timeouts must be finite and positive")
    if communication < loader:
        raise ValueError("process-group timeout must not be shorter than loader timeout")
    return loader, communication


def publish_config(path, config):
    """Publish a new immutable config; never overwrite a different contract."""
    path = Path(path)
    resolve_timeouts(config)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("immutable Stage-A config already exists with different content; use a new output path")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic no-replace publication on the same filesystem. Readers never
        # see partial JSON, and concurrent publishers cannot overwrite a file.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != config:
                raise ValueError("immutable Stage-A config already exists with different content; use a new output path") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def migrate_config(source, output):
    """Copy an old config into a new runtime contract without touching inputs."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("migration requires a different output path")
    config = json.loads(source.read_text())
    loader, communication = resolve_timeouts(config)
    migrated = {**config, "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
                "data_loader_timeout_seconds": loader,
                "process_group_timeout_seconds": communication}
    return publish_config(output, migrated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publish a new Stage-A runtime config; never overwrite the source")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(migrate_config(args.source, args.output))
