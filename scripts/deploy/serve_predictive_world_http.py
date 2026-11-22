#!/usr/bin/env python3
"""HTTP transport for the strict causal predictive-world service."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import time
from pathlib import Path

import torch

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.assets import resolve_asset_path
from psi.deploy.predictive_world_history import (
    CommittedActionPrefix,
    PredictiveWorldHistoryBuffer,
    PredictiveWorldRequest,
    TimedImage,
    TimedState,
)
from psi.deploy.predictive_world_service import PredictiveWorldService, QwenRequestAdapter
from psi.models.predictive_world.factory import build_deploy_policy


def _request(value: dict) -> PredictiveWorldRequest:
    prefix = value["committed_prefix"]
    return PredictiveWorldRequest(
        session_id=str(value["session_id"]), episode_id=str(value["episode_id"]),
        request_seq=int(value["request_seq"]),
        observation_anchor_time=float(value["observation_anchor_time"]),
        clock_domain=str(value["clock_domain"]),
        clock_mapping_id=str(value["clock_mapping_id"]),
        task_text=str(value["task_text"]), task_revision=str(value["task_revision"]),
        images=tuple(TimedImage(
            str(row["camera_id"]), float(row["capture_time"]), str(row["frame_id"]),
            base64.b64decode(row["encoded_image_base64"], validate=True),
        ) for row in value["images"]),
        states=tuple(TimedState(
            float(row["capture_time"]), int(row["source_seq"]),
            str(row["state_schema_id"]), tuple(float(item) for item in row["values"]),
            tuple(bool(item) for item in row["valid"]),
        ) for row in value["states"]),
        action_protocol_id=str(value["action_protocol_id"]),
        action_frequency=float(value["action_frequency"]),
        action_grid_start_time=float(value["action_grid_start_time"]),
        native_action_dim=int(value["native_action_dim"]),
        committed_prefix=CommittedActionPrefix(
            queue_revision=int(prefix["queue_revision"]),
            command_times=tuple(float(item) for item in prefix["command_times"]),
            values=tuple(tuple(float(item) for item in row) for row in prefix["values"]),
            valid=tuple(tuple(bool(item) for item in row) for row in prefix["valid"]),
            acknowledged_until=float(prefix["acknowledged_until"]),
        ), reset=bool(value.get("reset", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deploy-bundle", required=True)
    parser.add_argument(
        "--backend-factory", required=True,
        help="module:function returning a safety-validating execution backend",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config, _ = load_predictive_world_config(
        args.config, allow_existing_output_override=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_deploy_policy(config, args.deploy_bundle, device=device)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map)), local_files_only=True)
    protocols = json.loads(Path(config.data.action_protocol_registry).read_text())
    spaces = {
        name: str(spec["normalization_space"])
        for name, spec in protocols.items()
    }
    feature_manifest = Path(config.data.feature_cache)
    if feature_manifest.is_dir():
        feature_manifest = feature_manifest / "feature_manifest.json"
    preprocessing = json.loads(feature_manifest.read_text())["provenance"]["image_preprocessing"]
    module_name, separator, function_name = args.backend_factory.partition(":")
    if not separator:
        raise ValueError("--backend-factory must use module:function syntax")
    backend = getattr(importlib.import_module(module_name), function_name)(config)
    service = PredictiveWorldService(
        policy,
        QwenRequestAdapter(
            processor, image_preprocessing=preprocessing,
            action_horizon=config.data.action_horizon, normalization_spaces=spaces,
            current_visual_pooling=config.data.current_visual_pooling,
        ),
        PredictiveWorldHistoryBuffer(
            action_hz=config.data.action_hz,
            visual_offsets_s=config.data.observed_visual_offsets_s,
            observed_visual_mode=config.data.observed_visual_mode,
            visual_max_age_s=config.data.visual_max_age_s,
            state_max_age_s=config.data.state_max_age_s,
            state_count=config.data.state_history_count,
            state_hz=config.data.state_query_hz,
        ), device=device, backend=backend,
    )
    from fastapi import FastAPI, HTTPException
    import uvicorn

    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "route": "predictive_world"}

    @app.post("/infer")
    def infer(payload: dict):
        try:
            request = _request(payload)
            receive_time = backend.map_receive_time(
                clock_domain=request.clock_domain,
                clock_mapping_id=request.clock_mapping_id,
                server_monotonic=time.monotonic(),
            )
            response, receipt = service.infer_and_publish(
                request, receive_time=receive_time, seed=int(payload["seed"])
            )
            return {
                "session_id": response.session_id, "request_seq": response.request_seq,
                "queue_revision": response.queue_revision,
                "action_grid_start_time": response.action_grid_start_time,
                "action_frequency": response.action_frequency,
                "actions": response.actions.tolist(),
                "backend_receipt": receipt,
                "observation_age_s": response.observation_age_s,
                "inference_latency_s": response.inference_latency_s,
            }
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
