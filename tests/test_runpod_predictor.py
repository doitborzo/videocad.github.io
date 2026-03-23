from io import BytesIO
import json

from PIL import Image

from videocad_onshape.action_codec import StepContext
from videocad_onshape.config import ModelSettings
from videocad_onshape.videocad import RunpodPredictor


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_runpod_predictor_decodes_runsync_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        body = json.loads(request.data.decode("utf-8"))
        captured["body"] = body
        return FakeResponse({"status": "COMPLETED", "output": {"command": 4, "params": [1, 2, 3, 4, 5, 6]}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    predictor = RunpodPredictor(
        ModelSettings(
            backend="runpod",
            runpod_endpoint_id="endpoint123",
            runpod_api_key="rpa_test",
            runpod_base_url="https://api.runpod.ai/v2",
            runpod_timeout_seconds=30,
        )
    )
    image = Image.new("L", (224, 224), color=200)
    action = predictor.predict_next_action(
        [image],
        [],
        image,
        StepContext(step_index=1, op="draw_rectangle"),
    )

    assert action.command == 4
    assert action.params == [1, 2, 3, 4, 5, 6]
    assert captured["url"].startswith("https://api.runpod.ai/v2/endpoint123/runsync")
    assert "input" in captured["body"]


def test_runpod_predictor_healthcheck(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse({"status": "COMPLETED", "output": {"ok": True}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    predictor = RunpodPredictor(
        ModelSettings(
            backend="runpod",
            runpod_endpoint_id="endpoint123",
            runpod_api_key="rpa_test",
        )
    )
    ok, detail = predictor.healthcheck()

    assert ok is True
    assert "Runpod" in detail
