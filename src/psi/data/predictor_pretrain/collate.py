"""Canonical Qwen collation for heterogeneous Stage-A samples."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
from psi.models.action_aligned_image import content_boxes, image_preprocessing_spec, process_aspect_qwen


class PredictorPretrainCollator:
    def __init__(self, processor, *, image_preprocessing: str = "legacy_v1") -> None:
        self.processor = processor
        image_preprocessing_spec(image_preprocessing)
        self.image_preprocessing = image_preprocessing

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty Stage-A batch")
        # Current images must use the exact policy resize/quantization path used
        # by the offline future-target cache. Passing decoded [0,1] tensors to
        # the HF processor directly makes it rescale by 1/255 a second time.
        policy_images, conversations, image_counts, primary_indices = [], [], [], []
        for row in rows:
            raw_images = row["images"] if "images" in row else [row["image"]]
            if not raw_images:
                raise ValueError("each sample must contain its primary-current image")
            primary = int(row.get("primary_current_image_index", 0))
            if not 0 <= primary < len(raw_images):
                raise ValueError("primary-current image index is outside this sample")
            primary_indices.append(len(policy_images) + primary)
            image_counts.append(len(raw_images))
            images = [preprocess_endpoint_vlm_image(value, image_preprocessing=self.image_preprocessing)
                      for value in raw_images]
            policy_images.extend(images)
            content = []
            slots = row.get("image_slots")
            if slots is not None and len(slots) != len(images):
                raise ValueError("image_slots must describe every real image in order")
            if "images" in row and slots is None:
                raise ValueError("multi-image samples require explicit image_slots roles and times")
            for index, image in enumerate(images):
                if slots is not None:
                    slot = slots[index]
                    offset = float(slot["actual_offset_seconds"])
                    content.append({"type": "text", "text": (
                        f"Camera: {slot['camera_role']}; observation time: {offset:+.3f} seconds."
                    )})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": str(row["instruction"])})
            conversations.append([{"role": "user", "content": content}])
        texts = [
            self.processor.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
            for value in conversations
        ]
        # Rows already contain decoded tensors; qwen_vl_utils is only needed
        # for resolving URL/path/message image objects and is deliberately not
        # a runtime dependency here.
        image_inputs = policy_images
        qwen = (process_aspect_qwen(self.processor, images=image_inputs, texts=texts)
                if self.image_preprocessing == "aspect_v1" else self.processor(
                    text=texts, images=image_inputs, padding=True, return_tensors="pt"))
        # Match the deployed endpoint ABI: exact public image-only processing
        # supplies visual values/grid, while the chat call supplies placeholders.
        if self.image_preprocessing == "legacy_v1":
            exact = self.processor.image_processor(images=policy_images, return_tensors="pt")
            if not torch.equal(exact["image_grid_thw"], qwen["image_grid_thw"]):
                raise ValueError("Qwen image-only and chat image grids disagree")
            qwen["pixel_values"] = exact["pixel_values"]
            qwen["image_grid_thw"] = exact["image_grid_thw"]

        history = {int(np.asarray(row["raw_states"]).shape[0]) for row in rows}
        if len(history) != 1:
            raise ValueError("all Stage-A rows must use the same state-history length")
        max_dim = max(1, *(int(np.asarray(row["raw_states"]).shape[1]) for row in rows))
        batch = len(rows)
        history_len = history.pop()
        raw_states = torch.zeros(batch, history_len, max_dim, dtype=torch.float32)
        dimension_mask = torch.zeros_like(raw_states, dtype=torch.bool)
        state_history_mask = torch.zeros(batch, history_len, dtype=torch.bool)
        state_present = torch.tensor([bool(row["state_present"]) for row in rows])
        for index, row in enumerate(rows):
            values = torch.as_tensor(np.asarray(row["raw_states"]), dtype=torch.float32)
            valid_history = torch.as_tensor(row["state_history_mask"], dtype=torch.bool)
            if values.shape[0] != history_len or valid_history.shape != (history_len,):
                raise ValueError("invalid raw state/history mask shape")
            if values.numel():
                finite = torch.isfinite(values)
                values = torch.where(finite, values, torch.zeros_like(values))
                raw_states[index, :, : values.shape[1]] = values
                dimension_mask[index, :, : values.shape[1]] = finite & valid_history[:, None]
            state_history_mask[index] = valid_history & state_present[index]
        target = torch.as_tensor(np.stack([row["intent_target_vlm_state"] for row in rows]))
        result = {
            "sample_id": [str(row["sample_id"]) for row in rows],
            "source_id": [str(row["source_id"]) for row in rows],
            "episode_id": [str(row.get("episode_id", "")) for row in rows],
            "occurrence_index": torch.tensor([int(row["occurrence_index"]) for row in rows]),
            "input_ids": qwen["input_ids"],
            "attention_mask": qwen["attention_mask"].bool(),
            "pixel_values": qwen["pixel_values"],
            "image_grid_thw": qwen["image_grid_thw"],
            "image_counts": torch.tensor(image_counts, dtype=torch.long),
            "primary_current_image_indices": torch.tensor(primary_indices, dtype=torch.long),
            "image_slots": [row.get("image_slots", []) for row in rows],
            "image_slot_mask": [row.get("image_slot_mask", [True]) for row in rows],
            "raw_states": raw_states,
            "state_dimension_mask": dimension_mask,
            "state_schema_ids": [str(row["state_schema_id"]) for row in rows],
            "state_history_mask": state_history_mask,
            "state_present": state_present,
            "intent_target_vlm_state": target,
            "intent_target_valid": torch.ones(batch, dtype=torch.bool),
        }
        action_present = torch.tensor([bool(row.get("action_present", False)) for row in rows])
        if self.image_preprocessing == "aspect_v1":
            result["image_content_boxes"] = content_boxes(policy_images)
        if action_present.any():
            step_counts = {int(np.asarray(row["raw_actions"]).shape[0]) for row in rows}
            if len(step_counts) != 1:
                raise ValueError("all action-supervised rows must use one physical-time grid")
            action_steps = step_counts.pop()
            max_action_dim = max(int(np.asarray(row["raw_actions"]).shape[1]) for row in rows)
            raw_actions = torch.zeros(batch, action_steps, max_action_dim, dtype=torch.float32)
            action_mask = torch.zeros_like(raw_actions, dtype=torch.bool)
            for index, row in enumerate(rows):
                values = torch.as_tensor(np.asarray(row["raw_actions"]), dtype=torch.float32)
                time_mask = torch.as_tensor(row["action_time_mask"], dtype=torch.bool)
                if values.ndim != 2 or values.shape[0] != action_steps or time_mask.shape != (action_steps,):
                    raise ValueError("invalid raw action/time mask shape")
                if values.shape[1]:
                    finite = torch.isfinite(values)
                    raw_actions[index, :, : values.shape[1]] = torch.where(
                        finite, values, torch.zeros_like(values)
                    )
                    action_mask[index, :, : values.shape[1]] = finite & time_mask[:, None]
            result.update({
                "actions": raw_actions,
                "actions_mask": action_mask,
                "action_protocol_ids": [str(row["action_protocol_id"]) for row in rows],
                "action_present": action_present,
                "action_native_dims": torch.tensor(
                    [int(np.asarray(row["raw_actions"]).shape[1]) for row in rows],
                    dtype=torch.long,
                ),
            })
        return result


__all__ = ["PredictorPretrainCollator"]
