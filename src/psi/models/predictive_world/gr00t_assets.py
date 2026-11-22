"""Offline, immutable extraction of N1.7's actual frozen Qwen backbone."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from psi.data.predictive_world.cache import file_sha256


EXPORT_FORMAT = "zeta0.gr00t_n1d7.backbone_export.v1"
PROCESSOR_FILES = ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                   "video_preprocessor_config.json", "merges.txt", "vocab.json", "chat_template.json",
                   "chat_template.jinja", "added_tokens.json", "special_tokens_map.json")


def initialization_contract(training, source):
    """Serializable initialization identity, also checked on checkpoint reload."""
    if not training.joint_from_gr00t or source.get("format") != EXPORT_FORMAT:
        raise ValueError("GR00T initialization requires an explicit source-bound backbone")
    for key in ("source_weights_sha256", "exported_weights_sha256"):
        values = source.get(key)
        if not isinstance(values, dict) or not values:
            raise ValueError("GR00T initialization needs source and exported weight identities")
        for name, digest in values.items():
            _safe_file(".", name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid GR00T weight digest")
    return {
        "format": "zeta0.gr00t_n1d7.joint_initialization.v1",
        "kind": "pretrained_gr00t_new_zeta_and_embodiment", "seed": training.seed,
        "source": source, "student_initialization": "independent_copy_of_initialized_teacher",
        "new_parameters": ["H", "F", "E", "D", "future_projection", "embodiment_projectors"],
        "pretrained_parameters": ["frozen_backbone", "DiT", "vlln", "vl_self_attention", "position_embedding"],
        "optimizer_restored": False, "optimizer_update_offset": 0,
    }


def _safe_file(root, name):
    if Path(name).name != name:
        raise ValueError("checkpoint index contains a nonlocal filename")
    return Path(root) / name


def export_backbone(base, processor_source, output, *, shard_bytes=512 * 1024 * 1024):
    """Export every backbone tensor, retaining only the checkpoint's text layers.

    Uses one small output shard at a time on CPU. The native checkpoint already
    contains backbone weights; neither Cosmos weights nor internet access are
    required. An atomic directory publication prevents partial exports from being
    mistaken for valid frozen encoders.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, processor_source, output = map(Path, (base, processor_source, output))
    if output.exists():
        raise FileExistsError(f"backbone output must be new: {output}")
    if shard_bytes <= 0:
        raise ValueError("shard size must be positive")
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "Gr00tN1d7":
        raise ValueError("backbone extraction requires an N1.7 checkpoint")
    qwen_config = copy.deepcopy(json.loads((processor_source / "config.json").read_text()))
    if qwen_config.get("model_type") != "qwen3_vl":
        raise ValueError("Cosmos processor source must include its Qwen3-VL configuration")
    layers = int(config["select_layer"])
    if not 0 < layers <= qwen_config["text_config"]["num_hidden_layers"]:
        raise ValueError("invalid selected N1.7 backbone depth")
    qwen_config["text_config"]["num_hidden_layers"] = layers
    if qwen_config["text_config"]["hidden_size"] != config["backbone_embedding_dim"]:
        raise ValueError("Cosmos config width differs from the native N1.7 backbone")
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "backbone.model."
    keys = sorted(key for key in index if key.startswith(prefix))
    present_layers = {int(key.split("language_model.layers.", 1)[1].split(".", 1)[0])
                      for key in keys if "language_model.layers." in key}
    if present_layers != set(range(layers)):
        raise ValueError("checkpoint language layers disagree with select_layer")
    if not any("visual." in key for key in keys):
        raise ValueError("checkpoint contains no visual backbone")
    source_weights = {name: file_sha256(_safe_file(base, name)) for name in sorted(set(index.values()))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gr00t-backbone-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for required in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
            if not (processor_source / required).is_file():
                raise FileNotFoundError(processor_source / required)
        for name in PROCESSOR_FILES:
            if (processor_source / name).is_file():
                shutil.copyfile(processor_source / name, stage / name)
        (stage / "config.json").write_text(json.dumps(qwen_config, indent=2) + "\n")
        weight_map, output_hashes, pending = {}, {}, {}
        pending_bytes, total_bytes = 0, 0

        def flush():
            nonlocal pending_bytes
            if not pending:
                return
            name = f"model-{len(output_hashes) + 1:05d}.safetensors"
            save_file(pending, stage / name, metadata={"format": "pt"})
            weight_map.update({key: name for key in pending})
            output_hashes[name] = file_sha256(stage / name)
            pending.clear()
            pending_bytes = 0

        for shard in sorted({index[key] for key in keys}):
            with safe_open(_safe_file(base, shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    if index[key] != shard:
                        continue
                    tensor = handle.get_tensor(key)
                    size = tensor.numel() * tensor.element_size()
                    if pending_bytes + size > shard_bytes:
                        flush()
                    pending[key.removeprefix(prefix)] = tensor
                    pending_bytes += size
                    total_bytes += size
                    if pending_bytes >= shard_bytes:
                        flush()
                flush()
        (stage / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n")
        manifest = {
            "format": EXPORT_FORMAT, "source_config_sha256": file_sha256(base / "config.json"),
            "source_index_sha256": file_sha256(base / "model.safetensors.index.json"),
            "source_weights_sha256": source_weights, "selected_layers": layers,
            "backbone_embedding_dim": config["backbone_embedding_dim"],
            "exported_weights_sha256": output_hashes,
            "exported_config_sha256": file_sha256(stage / "config.json"),
            "processor_config_source_sha256": file_sha256(processor_source / "config.json"),
            "preprocessing": "zeta_aspect_v1_post_merger_content_area_overlap_4x4_v1",
        }
        (stage / "gr00t_backbone_export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Refuse overwrites even if another preparation process won the race.
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return manifest


def verify_backbone_export(backbone, base, *, expected_config_sha256):
    """Bind the current exported backbone and source-head weights before loading."""
    backbone, base = Path(backbone), Path(base)
    manifest = json.loads((backbone / "gr00t_backbone_export.json").read_text())
    if manifest.get("format") != EXPORT_FORMAT:
        raise ValueError("N1.7 requires a source-bound backbone export, not a generic Qwen model")
    if (manifest["source_config_sha256"] != expected_config_sha256
            or file_sha256(base / "config.json") != expected_config_sha256
            or file_sha256(backbone / "config.json") != manifest["exported_config_sha256"]
            or file_sha256(base / "model.safetensors.index.json") != manifest["source_index_sha256"]):
        raise ValueError("GR00T backbone/config identity mismatch")
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index.values()) != set(manifest["source_weights_sha256"]):
        raise ValueError("GR00T source shard inventory mismatch")
    exported_index = json.loads((backbone / "model.safetensors.index.json").read_text())["weight_map"]
    expected_keys = {key.removeprefix("backbone.model.") for key in index if key.startswith("backbone.model.")}
    if (set(exported_index) != expected_keys
            or set(exported_index.values()) != set(manifest["exported_weights_sha256"])):
        raise ValueError("GR00T exported backbone inventory mismatch")
    for name, expected in manifest["source_weights_sha256"].items():
        if file_sha256(_safe_file(base, name)) != expected:
            raise ValueError(f"GR00T base weight identity mismatch: {name}")
    for name, expected in manifest["exported_weights_sha256"].items():
        if file_sha256(_safe_file(backbone, name)) != expected:
            raise ValueError(f"GR00T exported backbone weight identity mismatch: {name}")
    return manifest


def load_backbone(path, *, dtype, device):
    """Reject missing/unexpected weights instead of silently initializing them."""
    from transformers import Qwen3VLForConditionalGeneration
    from psi.models.predictive_world.gr00t import native_types

    native_types()  # Validate source and forbid upstream's test-only weight skipping.
    model, information = Qwen3VLForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True, dtype=dtype, attn_implementation="sdpa",
        output_loading_info=True)
    if any(information.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"GR00T frozen backbone did not load exactly: {information}")
    return model.to(device)
