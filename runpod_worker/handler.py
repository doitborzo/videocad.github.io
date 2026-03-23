from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import gdown
import runpod

from videocad_onshape.action_codec import StepContext
from videocad_onshape.inference import VideoCADInferenceEngine, build_inference_settings_from_env, decode_image


def ensure_checkpoint() -> Path:
    settings = build_inference_settings_from_env()
    checkpoint_path = settings.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return checkpoint_path

    gdrive_id = os.getenv("VIDEOCAD_RUNPOD_CHECKPOINT_GDRIVE_ID")
    if not gdrive_id:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} and VIDEOCAD_RUNPOD_CHECKPOINT_GDRIVE_ID is not set."
        )
    gdown.download(id=gdrive_id, output=str(checkpoint_path), quiet=False)
    return checkpoint_path


@lru_cache(maxsize=1)
def get_engine() -> VideoCADInferenceEngine:
    ensure_checkpoint()
    engine = VideoCADInferenceEngine(build_inference_settings_from_env())
    engine.load()
    return engine


def handler(event: dict) -> dict:
    input_data = event.get("input", {})
    if input_data.get("healthcheck"):
        get_engine()
        return {"ok": True}

    frames = [decode_image(item) for item in input_data["frames"]]
    target_image = decode_image(input_data["target_image"])
    action_history = input_data.get("action_history", [])
    step_context_data = input_data.get("step_context", {})
    step_context = StepContext(
        step_index=int(step_context_data.get("step_index", 0)),
        op=str(step_context_data.get("op", "")),
        pending_numeric_text=step_context_data.get("pending_numeric_text"),
    )
    raw_action = get_engine().predict(frames, action_history, target_image, step_context)
    return {"command": raw_action.command, "params": raw_action.params}


runpod.serverless.start({"handler": handler})

