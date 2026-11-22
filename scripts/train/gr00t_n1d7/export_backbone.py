"""Prepare N1.7's frozen backbone locally; no downloads or training."""
import argparse
import json

from psi.models.predictive_world.gr00t_assets import export_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cosmos-processor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = export_backbone(args.base, args.cosmos_processor, args.output)
    print(json.dumps({"output": args.output, "selected_layers": result["selected_layers"],
                      "source_config_sha256": result["source_config_sha256"]}))


if __name__ == "__main__":
    main()
