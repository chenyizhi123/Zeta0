"""Describe and optionally verify an existing real-SONIC inference package."""
from __future__ import annotations

from pathlib import Path

from .common import read_object


def inspect_bundle(directory: str, *, full: bool = False) -> dict:
    root = Path(directory).expanduser().resolve()
    manifest = read_object(root / "inference_manifest.json")
    if manifest.get("format") != "zeta0.research_inference_weights.v1":
        raise ValueError("Expected a real-SONIC research inference package; use its owning loader for other checkpoint types.")
    if full:
        from psi.deploy.research_inference_weights import verify_inference_package
        manifest = verify_inference_package(root)
    return {
        "format": "zeta0.bundle_inspection.v1", "ok": True, "directory": str(root),
        "scope": "source contract and file hashes" if full else "unverified manifest metadata",
        "hashes_verified": full,
        "source_contract_verified": full and "package_files_sha256" in manifest,
        "legacy_package": "package_files_sha256" not in manifest,
        "task": manifest.get("task"), "architecture": manifest.get("architecture"),
        "temporal_abi": manifest.get("temporal_abi"),
        "action_protocol_id": manifest.get("action_protocol_id"),
        "state_schema_id": manifest.get("state_schema_id"),
        "encoder_id": manifest.get("encoder_id"), "processor_id": manifest.get("processor_id"),
        "rtc_max_prefix_steps": manifest.get("rtc_max_prefix_steps"),
        "hardware_acceptance_verified": False,
        "next_step": "Use the matching runtime, encoder and first-frame reference. Hash verification does not certify hardware compatibility.",
    }
