"""Bind actual frozen-Qwen files and preprocessing runtime before execution."""
from __future__ import annotations

from psi.data.predictive_world.cache import canonical_sha256


def load_qwen_processor(model_path):
    """Load offline assets, including Cosmos' tokenizer-owned chat template.

    Some Cosmos snapshots store the template only in tokenizer_config.json.
    Transformers loads that into the tokenizer but does not propagate it to the
    multimodal processor. Reuse that exact source-bound string without changing
    assets, image processing, or an already configured processor template.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    if not processor.chat_template:
        template = getattr(processor.tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template.strip():
            raise ValueError("Qwen processor assets need an unambiguous source chat template")
        processor.chat_template = template
    return processor


def verify_qwen_provenance(model_path, provenance, *, processor=None):
    from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
    from psi.data.predictor_pretrain.runtime import fingerprint_model_directory

    if (provenance.get('image_preprocessing') != 'aspect_v1'
            or provenance.get('feature_layer') != 'post_merger_content_area_overlap_4x4_v1'
            or provenance.get('dtype') != 'float16'
            or provenance.get('feature_shape', [16, 2048]) != [16, 2048]):
        raise ValueError('Qwen feature provenance does not match the implemented feature ABI')
    if processor is None:
        processor = load_qwen_processor(model_path)
    fingerprint = build_action_aligned_vlm_processor_fingerprint(processor)
    processor_id = 'qwen3vl-processor:' + canonical_sha256(fingerprint)
    if processor_id != provenance.get('processor_id'):
        raise ValueError(
            f"Qwen processor identity mismatch: expected={provenance.get('processor_id')}, "
            f"actual={processor_id}, actual_fingerprint={fingerprint}. "
            'Restore the source runtime or perform an explicitly reviewed migration; do not relabel old artifacts.'
        )
    # Check cheap runtime identity before hashing multi-GB weights. This happens
    # once at process startup, never per frame or per optimizer update.
    encoder_id = 'qwen3vl:' + fingerprint_model_directory(model_path)
    if encoder_id != provenance.get('encoder_id'):
        raise ValueError(f"Qwen encoder identity mismatch: expected={provenance.get('encoder_id')}, actual={encoder_id}")
    return {'encoder_id': encoder_id, 'processor_id': processor_id, 'processor_fingerprint': fingerprint}
