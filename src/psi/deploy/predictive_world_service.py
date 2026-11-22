"""Transport-neutral, causal predictive-world inference service."""

from __future__ import annotations

import contextlib
import io
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from PIL import Image

from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.deploy.predictive_world_history import (
    CausalHistoryWindow,
    PredictiveWorldHistoryBuffer,
    PredictiveWorldRequest,
    QueueRevisionGuard,
)
from psi.models.predictive_world.policy import PredictiveWorldPolicy


RTC_MODEL_TIME_MODES = ("legacy_command_time_v1", "training_action_grid_v1")


@dataclass(frozen=True)
class PredictiveWorldResponse:
    session_id: str
    request_seq: int
    queue_revision: int
    action_grid_start_time: float
    action_frequency: float
    actions: np.ndarray
    observation_age_s: float
    inference_latency_s: float


class ExecutionBackend(Protocol):
    def map_receive_time(
        self, *, clock_domain: str, clock_mapping_id: str, server_monotonic: float
    ) -> float: ...

    def validate_and_publish(
        self,
        actions: np.ndarray,
        *,
        action_grid_start_time: float,
        action_frequency: float,
        expected_queue_revision: int,
    ) -> Any: ...


class QwenRequestAdapter:
    """Convert a validated history window into the exact training observation ABI."""

    def __init__(
        self,
        processor: Any,
        *,
        image_preprocessing: str,
        action_horizon: int,
        normalization_spaces: Mapping[str, str],
        camera_labels: Mapping[str, str] | None = None,
        rtc_model_time_mode: str = "legacy_command_time_v1",
        current_visual_pooling: str = "canvas_v1",
    ) -> None:
        if image_preprocessing not in {"legacy_v1", "aspect_v1"} or action_horizon <= 0:
            raise ValueError("request adapter image/action configuration is invalid")
        self.processor = processor
        self.image_preprocessing = image_preprocessing
        self.action_horizon = int(action_horizon)
        self.normalization_spaces = dict(normalization_spaces)
        self.camera_labels = dict(camera_labels or {})
        if rtc_model_time_mode not in RTC_MODEL_TIME_MODES:
            raise ValueError("unsupported RTC model time mode")
        self.rtc_model_time_mode = rtc_model_time_mode
        if current_visual_pooling not in {"canvas_v1", "content_area_v1"}:
            raise ValueError("unsupported current visual pooling")
        self.current_visual_pooling = current_visual_pooling
        if current_visual_pooling == "content_area_v1" and image_preprocessing != "aspect_v1":
            raise ValueError("content-area observations require aspect_v1 preprocessing")

    def _model_prefix_times(
        self, request: PredictiveWorldRequest, window: CausalHistoryWindow
    ) -> torch.Tensor:
        prefix = window.committed_prefix
        if self.rtc_model_time_mode == "legacy_command_time_v1":
            # Preserve the historical adapter, including its float32 subtraction.
            return torch.tensor(prefix.command_times) - request.observation_anchor_time
        if (
            request.action_protocol_id != "sonic_semantic40_v31"
            or request.native_action_dim != 40
            or request.action_frequency != 50.0
            or self.action_horizon not in (30, 40, 50)
        ):
            raise ValueError("training_action_grid_v1 requires the SONIC semantic40 50 Hz / 30-, 40- or 50-step contract")
        prefix.validate(
            anchor_time=request.observation_anchor_time,
            action_hz=request.action_frequency,
            native_dim=request.native_action_dim,
        )
        if abs(request.action_grid_start_time - request.observation_anchor_time) > 1e-6:
            raise ValueError("training_action_grid_v1 requires the command grid to start at the observation anchor")
        # SONIC training selects future actions at +0.02-second intervals, whereas the
        # execution queue schedules these slots from t. Adapt only the model
        # slot coordinates; command_times retain their physical queue meaning.
        # Construct grid values in float64 before casting, matching index rows
        # and avoiding cancellation when the observation clock is large.
        return (torch.arange(1, len(prefix.values) + 1, dtype=torch.float64) / 50).float()

    def __call__(
        self, request: PredictiveWorldRequest, window: CausalHistoryWindow
    ) -> ObservedHistory:
        images, labels = [], []
        primary = None
        for item in window.images:
            if item is None:
                continue
            try:
                if item.raw_rgb is not None:
                    raw = np.asarray(item.raw_rgb)
                    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
                        raise ValueError("raw RGB image must be uint8 HWC with three channels")
                    decoded = Image.fromarray(raw).convert("RGB")
                else:
                    with Image.open(io.BytesIO(item.encoded_image)) as image:
                        decoded = image.convert("RGB")
                images.append(preprocess_endpoint_vlm_image(
                    decoded, image_preprocessing=self.image_preprocessing
                ))
            except Exception as error:
                raise ValueError(f"invalid encoded observation image {item.frame_id!r}") from error
            labels.append((self.camera_labels.get(item.camera_id, item.camera_id), item.capture_time - request.observation_anchor_time))
            if item.camera_id == "primary":
                primary = len(images) - 1
        if not images or primary is None:
            raise ValueError("validated history has no current image")
        content = []
        for image, (camera, time) in zip(images, labels):
            content.extend((
                {"type": "text", "text": f"Camera: {camera}; observation time: {time:+.3f} seconds."},
                {"type": "image", "image": image},
            ))
        content.append({"type": "text", "text": request.task_text})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        exact = self.processor.image_processor(images=images, return_tensors="pt")
        if not torch.equal(encoded["image_grid_thw"], exact["image_grid_thw"]):
            raise ValueError("Qwen request grids disagree")
        encoded["pixel_values"] = exact["pixel_values"]
        encoded["image_grid_thw"] = exact["image_grid_thw"]
        encoded["image_counts"] = torch.tensor([len(images)])
        encoded["primary_current_image_indices"] = torch.tensor([primary])
        if self.current_visual_pooling == "content_area_v1":
            from psi.models.action_aligned_image import content_boxes
            encoded["image_content_boxes"] = content_boxes(images)
        values = torch.from_numpy(window.state_values).unsqueeze(0)
        valid = torch.from_numpy(window.state_dimension_valid).unsqueeze(0)
        times = torch.from_numpy(
            window.state_actual_times - request.observation_anchor_time
        ).float().unsqueeze(0)
        native_dim = request.native_action_dim
        prefix_values = torch.zeros(1, self.action_horizon, native_dim)
        prefix_valid = torch.zeros_like(prefix_values, dtype=torch.bool)
        prefix_times = torch.zeros(1, self.action_horizon)
        count = len(window.committed_prefix.values)
        if count > self.action_horizon:
            raise ValueError("committed prefix exceeds trained action horizon")
        model_prefix_times = self._model_prefix_times(request, window)
        if count:
            prefix_values[0, :count] = torch.tensor(window.committed_prefix.values)
            prefix_valid[0, :count] = torch.tensor(window.committed_prefix.valid)
            prefix_times[0, :count] = model_prefix_times
        space = self.normalization_spaces.get(request.action_protocol_id)
        if space is None:
            raise ValueError("request action protocol is absent from normalization registry")
        committed = NativeActionBatch(
            prefix_values, prefix_valid, prefix_times,
            (request.action_protocol_id,), torch.tensor([native_dim]), (space,),
        )
        return ObservedHistory(
            qwen_inputs=dict(encoded), raw_states=values, state_valid=valid,
            state_times=times, state_schema_ids=(window.state_schema_id,),
            action_protocol_ids=(request.action_protocol_id,),
            action_native_dims=torch.tensor([native_dim]),
            action_normalization_spaces=(space,), committed_actions=committed,
        )


class PredictiveWorldService:
    """Serialize per-process inference and expose explicit queue commit semantics."""

    def __init__(
        self,
        policy: PredictiveWorldPolicy,
        adapter: QwenRequestAdapter,
        history: PredictiveWorldHistoryBuffer,
        *,
        device: torch.device,
        backend: ExecutionBackend | None = None,
    ) -> None:
        self.policy, self.adapter, self.history, self.device = policy, adapter, history, device
        self.backend = backend
        self._guards: dict[str, QueueRevisionGuard] = {}
        self._lock = threading.Lock()

    def infer(
        self,
        request: PredictiveWorldRequest,
        *,
        receive_time: float,
        seed: int,
    ) -> PredictiveWorldResponse:
        with self._lock:
            started = time.perf_counter()
            window = self.history.build(request, receive_time=receive_time)
            revision = request.committed_prefix.queue_revision
            guard = self._guards.get(request.session_id)
            if request.reset or guard is None:
                guard = QueueRevisionGuard(revision)
                self._guards[request.session_id] = guard
            guard.authorize_publish(revision)
            observed = self.adapter(request, window)
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            action_noise = torch.randn(
                (1, self.policy.student.action_steps, request.native_action_dim),
                generator=generator, device=self.device,
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast:
                actions = self.policy.sample(
                    _to_device(observed, self.device),
                    action_noise=action_noise,
                )[0, :, : request.native_action_dim].float().cpu().numpy()
            guard.authorize_publish(revision)
            return PredictiveWorldResponse(
                request.session_id, request.request_seq, revision,
                request.action_grid_start_time, request.action_frequency, actions,
                receive_time - request.observation_anchor_time,
                time.perf_counter() - started,
            )

    def commit_response(self, response: PredictiveWorldResponse) -> int:
        return self.commit_revision(response.session_id, response.queue_revision)

    def commit_revision(self, session_id: str, queue_revision: int) -> int:
        with self._lock:
            guard = self._guards[session_id]
            guard.authorize_publish(queue_revision)
            return guard.advance()

    def infer_and_publish(
        self, request: PredictiveWorldRequest, *, receive_time: float, seed: int
    ) -> tuple[PredictiveWorldResponse, Any]:
        if self.backend is None:
            raise RuntimeError("no validated execution backend is configured")
        response = self.infer(request, receive_time=receive_time, seed=seed)
        with self._lock:
            guard = self._guards[response.session_id]
            guard.authorize_publish(response.queue_revision)
            receipt = self.backend.validate_and_publish(
                response.actions,
                action_grid_start_time=response.action_grid_start_time,
                action_frequency=response.action_frequency,
                expected_queue_revision=response.queue_revision,
            )
            guard.advance()
        return response, receipt


def _to_device(observed: ObservedHistory, device: torch.device) -> ObservedHistory:
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        return value

    committed = observed.committed_actions
    if committed is not None:
        committed = NativeActionBatch(
            move(committed.values), move(committed.valid), move(committed.times),
            committed.protocol_ids, move(committed.native_dims), committed.normalization_spaces,
        )
    return ObservedHistory(
        qwen_inputs=move(observed.qwen_inputs), raw_states=move(observed.raw_states),
        state_valid=move(observed.state_valid), state_times=move(observed.state_times),
        state_schema_ids=observed.state_schema_ids,
        action_protocol_ids=observed.action_protocol_ids,
        action_native_dims=move(observed.action_native_dims),
        action_normalization_spaces=observed.action_normalization_spaces,
        committed_actions=committed,
    )


__all__ = [
    "ExecutionBackend", "PredictiveWorldResponse", "PredictiveWorldService",
    "QwenRequestAdapter",
]
