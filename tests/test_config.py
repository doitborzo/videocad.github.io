from pathlib import Path

from videocad_onshape.config import load_config


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEOCAD_ONSHAPE_OUTPUT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("VIDEOCAD_ONSHAPE_WINDOW_WIDTH", "1440")
    monkeypatch.setenv("VIDEOCAD_ONSHAPE_RUNPOD_ENDPOINT_ID", "endpoint-123")

    config = load_config()

    assert config.paths.output_dir == Path(tmp_path / "artifacts")
    assert config.window.width == 1440
    assert config.model.runpod_endpoint_id == "endpoint-123"


def test_dotenv_loads_planner_key_from_config_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("VIDEOCAD_ONSHAPE_PLANNER_API_KEY", raising=False)
    config_path = tmp_path / "videocad_onshape.toml"
    config_path.write_text("[planner]\nprovider = \"mock\"\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "VIDEOCAD_ONSHAPE_PLANNER_API_KEY=sk-test-from-dotenv\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.planner.api_key == "sk-test-from-dotenv"
