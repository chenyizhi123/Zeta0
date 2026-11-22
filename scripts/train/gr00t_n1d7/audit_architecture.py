#!/usr/bin/env python3
"""Check full model shapes against real weight headers without allocating its tensors."""
import argparse
import json
from pathlib import Path


def audit(base, backbone):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from psi.models.predictive_world.gr00t import native_types, PROJECTOR_PREFIXES
    from psi.models.predictive_world.gr00t_assets import verify_backbone_export
    from psi.data.predictive_world.cache import file_sha256

    base, backbone = Path(base).resolve(), Path(backbone).resolve()
    source = verify_backbone_export(backbone, base, expected_config_sha256=file_sha256(base / "config.json"))
    config_type, head_type = native_types()
    config = config_type.from_pretrained(str(base), local_files_only=True)
    config.max_num_embodiments = 1
    config.strict_action_padding_mask = config.strict_vl_padding_mask = True
    with torch.device("meta"):
        qwen = Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(str(backbone), local_files_only=True))
        head = head_type(config)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {"backbone.model." + k: tuple(v.shape) for k, v in qwen.state_dict().items()}
    expected.update({"action_head." + k: tuple(v.shape) for k, v in head.state_dict().items()
                     if not k.startswith(PROJECTOR_PREFIXES)})
    actual = {key for key in index if key.startswith("backbone.model.") or (
        key.startswith("action_head.") and not key.removeprefix("action_head.").startswith(PROJECTOR_PREFIXES))}
    if set(expected) != actual:
        raise ValueError(f"model weight inventory differs: {sorted(set(expected) ^ actual)[:20]}")
    for shard in sorted(set(index.values())):
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            for key in expected:
                if index[key] == shard and tuple(handle.get_slice(key).get_shape()) != expected[key]:
                    raise ValueError(f"model weight shape differs: {key}")
    return dict(passed=True, mode="meta_architecture_and_real_safetensors_headers",
        full_model_forward=False, source_config_sha256=source["source_config_sha256"],
        selected_layers=config.select_layer, action_horizon=config.action_horizon,
        backbone_parameters=sum(p.numel() for p in qwen.parameters()),
        head_parameters=sum(p.numel() for p in head.parameters()), checked_tensors=len(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.base, args.backbone)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
