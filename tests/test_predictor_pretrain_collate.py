import numpy as np
import torch
from PIL import Image

from psi.data.predictor_pretrain.collate import PredictorPretrainCollator


class _ImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert all(isinstance(image, Image.Image) and image.size == (320, 240) for image in images)
        values = torch.from_numpy(np.stack([np.asarray(image).copy() for image in images]))
        return {
            "pixel_values": values.permute(0, 3, 1, 2).float().div(127.5).sub(1.0),
            "image_grid_thw": torch.tensor([[1, 16, 20]] * len(images)),
        }


class _Processor:
    def __init__(self):
        self.image_processor = _ImageProcessor()

    def apply_chat_template(self, _conversation, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "prompt"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text and padding and return_tensors == "pt"
        visual = self.image_processor(images=images, return_tensors=return_tensors)
        return {
            "input_ids": torch.ones(len(text), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
            **visual,
        }


def test_collator_uses_the_same_policy_preprocessing_as_future_cache():
    image = torch.linspace(0, 1, 3 * 480 * 640).reshape(3, 480, 640)
    row = {
        "image": image, "instruction": "move", "raw_states": np.zeros((8, 3), np.float32),
        "state_history_mask": np.ones(8, bool), "state_present": True,
        "state_schema_id": "g1", "sample_id": "s", "source_id": "HE",
        "episode_id": "7", "occurrence_index": 0,
        "intent_target_vlm_state": np.zeros((16, 2048), np.float16),
    }
    batch = PredictorPretrainCollator(_Processor())([row])
    assert batch["pixel_values"].min() <= -0.99
    assert batch["pixel_values"].max() >= 0.99
    assert batch["image_grid_thw"].tolist() == [[1, 16, 20]]
    assert batch["episode_id"] == ["7"]


def test_multiview_history_collator_keeps_roles_and_primary_selection():
    class RecordingProcessor(_Processor):
        def __init__(self):
            super().__init__()
            self.conversations = []

        def apply_chat_template(self, conversation, **kwargs):
            self.conversations.append(conversation)
            return super().apply_chat_template(conversation, **kwargs)

    def row(count):
        images = [torch.full((3, 24, 32), float(index) / 5) for index in range(count)]
        return {
            "image": images[1], "images": images,
            "image_slots": [
                {"camera_role": role, "actual_offset_seconds": offset}
                for role, offset in [("primary", -0.2), ("primary", 0), ("left_wrist", 0)][:count]
            ],
            "primary_current_image_index": 1,
            "image_slot_mask": [True] * count + [False],
            "instruction": "grasp", "raw_states": np.zeros((8, 3), np.float32),
            "state_history_mask": np.ones(8, bool), "state_present": True,
            "state_schema_id": "g1", "sample_id": f"s{count}", "source_id": "HE",
            "occurrence_index": 0, "intent_target_vlm_state": np.zeros((16, 2048), np.float16),
        }

    processor = RecordingProcessor()
    batch = PredictorPretrainCollator(processor)([row(3), row(2)])
    assert batch["input_ids"].shape[0] == 2
    assert batch["image_grid_thw"].shape[0] == 5
    assert batch["image_counts"].tolist() == [3, 2]
    assert batch["primary_current_image_indices"].tolist() == [1, 4]
    content = processor.conversations[0][0]["content"]
    assert len([item for item in content if item["type"] == "image"]) == 3
    assert "-0.200" in content[0]["text"]
    assert "left_wrist" in content[4]["text"]
    assert batch["image_slot_mask"] == [[True, True, True, False], [True, True, False]]
