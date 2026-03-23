from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import json
import os

import numpy as np
from PIL import Image

from .action_codec import RawModelAction, StepContext
from .compat import dataclass
from .errors import ConfigurationError


def decode_image(encoded: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("L").resize((224, 224))


@dataclass
class InferenceSettings:
    checkpoint_path: Path
    model_config_path: Path
    model_name: str = "cad_past_10_actions_and_states_timestep_embedding"
    device: str = "cpu"


class VideoCADInferenceEngine:
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._model = None
        self._torch = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.settings.checkpoint_path.exists():
            raise ConfigurationError(f"Checkpoint does not exist: {self.settings.checkpoint_path}")
        if not self.settings.model_config_path.exists():
            raise ConfigurationError(f"Model config does not exist: {self.settings.model_config_path}")

        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"PyTorch is not available: {exc}") from exc

        from videocad_onshape.vendor.model.autoregressive_transformer import AutoRegressiveTransformer

        with self.settings.model_config_path.open("r", encoding="utf-8") as fh:
            config_map = json.load(fh)
        model_config = dict(config_map[self.settings.model_name])
        model_config["device"] = self.settings.device
        model = AutoRegressiveTransformer(**model_config).to(self.settings.device)

        checkpoint = torch.load(self.settings.checkpoint_path, map_location=self.settings.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module._orig_mod."):
                clean_state_dict[key.replace("module._orig_mod.", "")] = value
            elif key.startswith("module."):
                clean_state_dict[key.replace("module.", "")] = value
            else:
                clean_state_dict[key] = value
        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()
        self._model = model
        self._torch = torch

    def predict(
        self,
        frame_history: list[Image.Image],
        action_history: list[list[int]],
        target_image: Image.Image,
        step_context: StepContext | None = None,
    ) -> RawModelAction:
        self.load()
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        seq_len = len(frame_history)
        frames = np.stack([self._to_array(image) for image in frame_history], axis=0)
        target = self._to_array(target_image)[0]
        actions = np.zeros((seq_len, 7), dtype=np.float32)
        truncated_history = action_history[-(seq_len - 1) :] if seq_len > 1 else []
        for index, action in enumerate(truncated_history, start=1):
            actions[index] = np.asarray(action, dtype=np.float32)
        actions[:, 0] = actions[:, 0] / 4.0
        actions[:, 1:] = np.where(actions[:, 1:] >= 0, actions[:, 1:] / 1000.0, actions[:, 1:])

        frame_tensor = torch.tensor(frames, dtype=torch.float32, device=self.settings.device).unsqueeze(0)
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=self.settings.device).unsqueeze(0)
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.settings.device).unsqueeze(0)

        with torch.no_grad():
            cmds, params = self._model.forward(
                {
                    "frames": frame_tensor,
                    "actions": action_tensor,
                    "cad_image": target_tensor,
                }
            )

        command = int(torch.argmax(cmds[:, -1], dim=-1).item())
        param_values = torch.argmax(params[:, -1], dim=-1).squeeze(0).tolist()
        return RawModelAction(command=command, params=[int(value) for value in param_values])

    def _to_array(self, image: Image.Image) -> np.ndarray:
        grayscale = image.convert("L").resize((224, 224))
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        normalized = (array - 0.5) / 0.5
        return normalized[np.newaxis, :, :]


def build_inference_settings_from_env() -> InferenceSettings:
    checkpoint_path = Path(os.getenv("VIDEOCAD_RUNPOD_CHECKPOINT_PATH", "/models/videocadformer.pt"))
    model_config_path = Path(
        os.getenv(
            "VIDEOCAD_RUNPOD_MODEL_CONFIG_PATH",
            "/app/src/videocad_onshape/vendor/model_configs/transformer_experiments.json",
        )
    )
    model_name = os.getenv("VIDEOCAD_RUNPOD_MODEL_NAME", "cad_past_10_actions_and_states_timestep_embedding")
    default_device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in {None, ""} else "cpu"
    device = os.getenv("VIDEOCAD_RUNPOD_DEVICE", default_device)
    return InferenceSettings(
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        model_name=model_name,
        device=device,
    )
