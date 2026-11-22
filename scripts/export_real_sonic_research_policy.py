#!/usr/bin/env python3
"""Package a completed joint checkpoint for real-SONIC research inference."""

import argparse
import json

from psi.deploy.research_inference_weights import export_real_sonic_research_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Complete phase=joint checkpoint directory")
    parser.add_argument("--config", required=True, help="Exact original training config bound to this checkpoint")
    parser.add_argument("--output", required=True, help="New research inference directory; cannot overwrite")
    args = parser.parse_args()
    manifest = export_real_sonic_research_policy(args.checkpoint, args.config, args.output)
    print(json.dumps({
        "event": "real_sonic_research_policy_packaged", "output": args.output,
        "source_completed_updates": manifest["source_completed_updates"],
        "state_history_count": manifest["temporal_abi"]["state_history_count"],
        "rtc_max_prefix_steps": manifest["rtc_max_prefix_steps"],
        "validated_production_deploy_bundle": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
