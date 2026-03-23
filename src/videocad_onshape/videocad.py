from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol
import base64
import json
import urllib.error
import urllib.request

from PIL import Image

from .action_codec import RawModelAction, StepContext
from .config import ModelSettings
from .errors import ConfigurationError


class Predictor(Protocol):
    def predict_next_action(
        self,
        frame_history: list[Image.Image],
        action_history: list[list[int]],
        target_image: Image.Image,
        step_context: StepContext,
    ) -> RawModelAction:
        ...

    def healthcheck(self) -> tuple[bool, str]:
        ...


def _encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("L").resize((224, 224)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(slots=True)
class RunpodPredictor:
    settings: ModelSettings

    def predict_next_action(
        self,
        frame_history: list[Image.Image],
        action_history: list[list[int]],
        target_image: Image.Image,
        step_context: StepContext,
    ) -> RawModelAction:
        payload = {
            "frames": [_encode_image(image) for image in frame_history],
            "action_history": action_history,
            "target_image": _encode_image(target_image),
            "step_context": {
                "step_index": step_context.step_index,
                "op": step_context.op,
                "pending_numeric_text": step_context.pending_numeric_text,
            },
        }
        result = self._runsync(payload)
        command = int(result["command"])
        params = [int(value) for value in result["params"]]
        return RawModelAction(command=command, params=params)

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self._validate()
            self._runsync({"healthcheck": True})
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, "Runpod endpoint reachable"

    def _validate(self) -> None:
        if self.settings.backend != "runpod":
            raise ConfigurationError(f"Unsupported model backend: {self.settings.backend}")
        if not self.settings.runpod_endpoint_id:
            raise ConfigurationError("model.runpod_endpoint_id is required.")
        if not self.settings.runpod_api_key:
            raise ConfigurationError("model.runpod_api_key is required.")

    def _runsync(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        self._validate()
        wait_ms = max(1000, min(300000, int(self.settings.runpod_timeout_seconds * 1000)))
        url = (
            f"{self.settings.runpod_base_url.rstrip('/')}/"
            f"{self.settings.runpod_endpoint_id}/runsync?wait={wait_ms}"
        )
        body = json.dumps({"input": input_payload}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.runpod_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.runpod_timeout_seconds + 10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise ConfigurationError(f"Runpod request failed: {exc.code} {message}") from exc
        except urllib.error.URLError as exc:
            raise ConfigurationError(f"Runpod request failed: {exc}") from exc

        if data.get("status") not in {None, "COMPLETED"}:
            raise ConfigurationError(f"Runpod returned status {data.get('status')}: {data}")
        output = data.get("output", data)
        if not isinstance(output, dict):
            raise ConfigurationError(f"Unexpected Runpod output: {data}")
        return output


def build_predictor(settings: ModelSettings) -> Predictor:
    if settings.backend != "runpod":
        raise ConfigurationError(f"Unsupported model backend: {settings.backend}")
    return RunpodPredictor(settings=settings)
