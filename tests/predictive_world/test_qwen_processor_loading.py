import json

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from psi.data.predictive_world.provenance import load_qwen_processor


TEMPLATE = "{% for m in messages %}{% for c in m['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}"


def assets(path, *, processor_template=None, tokenizer_template=TEMPLATE):
    vocabulary = {s: i for i, s in enumerate([
        "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "a"])}
    (path / 'vocab.json').write_text(json.dumps(vocabulary))
    (path / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = Qwen2TokenizerFast(vocab_file=str(path / 'vocab.json'),
        merges_file=str(path / 'merges.txt'), additional_special_tokens=list(vocabulary)[:4])
    processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(
        min_pixels=28 * 28, max_pixels=28 * 28), tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(), chat_template=processor_template)
    processor.save_pretrained(path)
    config = json.loads((path / 'tokenizer_config.json').read_text())
    config['chat_template'] = tokenizer_template
    (path / 'tokenizer_config.json').write_text(json.dumps(config))


def test_tokenizer_owned_template_renders_real_multimodal_inputs(tmp_path):
    assets(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert original.chat_template is None
    processor = load_qwen_processor(tmp_path)
    conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'a'}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    reference = original.apply_chat_template(conversation, chat_template=TEMPLATE,
                                             tokenize=False, add_generation_prompt=True)
    assert text == reference
    image = Image.new('RGB', (28, 28), color=(90, 120, 160))
    actual = processor(text=[text], images=[image], return_tensors='pt')
    expected = original(text=[reference], images=[image], return_tensors='pt')
    assert (actual['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).any()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_existing_processor_template_takes_precedence(tmp_path):
    assets(tmp_path, processor_template=TEMPLATE, tokenizer_template='different')
    assert load_qwen_processor(tmp_path).chat_template == TEMPLATE


@pytest.mark.parametrize('template', [None, '', {'default': TEMPLATE, 'other': 'ambiguous'}])
def test_missing_or_ambiguous_tokenizer_template_fails(tmp_path, template):
    assets(tmp_path, tokenizer_template=template)
    with pytest.raises(ValueError, match='source chat template'):
        load_qwen_processor(tmp_path)
