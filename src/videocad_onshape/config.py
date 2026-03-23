from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import os
import tomllib


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_CANVAS_SELECTORS = (
    "canvas",
    "[data-testid='cad-canvas'] canvas",
    ".cad-partstudio-canvas canvas",
    ".GLCanvas canvas",
)


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if value and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1]
    return key, value


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _coerce_number(value: str, cast: type[int] | type[float]) -> int | float:
    return cast(value.strip())


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class WindowGeometry:
    x: int = 40
    y: int = 60
    width: int = 1600
    height: int = 1000


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class PlannerSettings:
    provider: str = "openai-compatible"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1/chat/completions"
    timeout_seconds: float = 30.0
    temperature: float = 0.0


@dataclass(slots=True)
class BrowserSettings:
    backend: str = "playwright"
    profile_path: Path | None = None
    profile_name: str | None = None
    blank_part_studio_url: str | None = None
    headless: bool = False
    canvas_selectors: tuple[str, ...] = DEFAULT_CANVAS_SELECTORS
    canvas_override: Rect | None = None


@dataclass(slots=True)
class ModelSettings:
    backend: str = "runpod"
    runpod_endpoint_id: str | None = None
    runpod_api_key: str | None = None
    runpod_base_url: str = "https://api.runpod.ai/v2"
    runpod_timeout_seconds: float = 120.0


@dataclass(slots=True)
class SafetySettings:
    emergency_stop_hotkey: str = "ctrl+c"
    max_actions_per_step: int = 40
    max_retries_per_action_class: int = 5
    stall_delta_threshold: float = 1.0
    stall_patience: int = 4
    geometry_tolerance_px: float = 6.0
    min_inter_action_delay_sec: float = 0.2
    target_similarity_threshold: float = 0.82
    post_action_change_threshold: float = 0.75


@dataclass(slots=True)
class PathsSettings:
    output_dir: Path = field(default_factory=lambda: _project_root() / "runs")


@dataclass(slots=True)
class AppConfig:
    config_path: Path | None = None
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    window: WindowGeometry = field(default_factory=WindowGeometry)
    safety: SafetySettings = field(default_factory=SafetySettings)
    paths: PathsSettings = field(default_factory=PathsSettings)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for section in ("config_path",):
            if data.get(section) is not None:
                data[section] = str(data[section])
        for group in ("browser", "model", "paths"):
            for key, value in list(data[group].items()):
                if isinstance(value, Path):
                    data[group][key] = str(value)
        return data


def _load_rect(data: dict[str, Any] | None) -> Rect | None:
    if not data:
        return None
    return Rect(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def _load_from_mapping(data: dict[str, Any], config_path: Path | None) -> AppConfig:
    base_dir = config_path.parent if config_path else _project_root()

    planner_data = data.get("planner", {})
    browser_data = data.get("browser", {})
    model_data = data.get("model", {})
    window_data = data.get("window", {})
    safety_data = data.get("safety", {})
    paths_data = data.get("paths", {})

    return AppConfig(
        config_path=config_path,
        planner=PlannerSettings(
            provider=str(planner_data.get("provider", "openai-compatible")),
            model=str(planner_data.get("model", "gpt-4.1-mini")),
            api_key=planner_data.get("api_key"),
            base_url=str(planner_data.get("base_url", "https://api.openai.com/v1/chat/completions")),
            timeout_seconds=float(planner_data.get("timeout_seconds", 30.0)),
            temperature=float(planner_data.get("temperature", 0.0)),
        ),
        browser=BrowserSettings(
            backend=str(browser_data.get("backend", "playwright")),
            profile_path=_resolve_path(browser_data.get("profile_path"), base_dir),
            profile_name=browser_data.get("profile_name"),
            blank_part_studio_url=browser_data.get("blank_part_studio_url"),
            headless=bool(browser_data.get("headless", False)),
            canvas_selectors=tuple(browser_data.get("canvas_selectors", DEFAULT_CANVAS_SELECTORS)),
            canvas_override=_load_rect(browser_data.get("canvas_override")),
        ),
        model=ModelSettings(
            backend=str(model_data.get("backend", "runpod")),
            runpod_endpoint_id=model_data.get("runpod_endpoint_id"),
            runpod_api_key=model_data.get("runpod_api_key"),
            runpod_base_url=str(model_data.get("runpod_base_url", "https://api.runpod.ai/v2")),
            runpod_timeout_seconds=float(model_data.get("runpod_timeout_seconds", 120.0)),
        ),
        window=WindowGeometry(
            x=int(window_data.get("x", 40)),
            y=int(window_data.get("y", 60)),
            width=int(window_data.get("width", 1600)),
            height=int(window_data.get("height", 1000)),
        ),
        safety=SafetySettings(
            emergency_stop_hotkey=str(safety_data.get("emergency_stop_hotkey", "ctrl+c")),
            max_actions_per_step=int(safety_data.get("max_actions_per_step", 40)),
            max_retries_per_action_class=int(safety_data.get("max_retries_per_action_class", 5)),
            stall_delta_threshold=float(safety_data.get("stall_delta_threshold", 1.0)),
            stall_patience=int(safety_data.get("stall_patience", 4)),
            geometry_tolerance_px=float(safety_data.get("geometry_tolerance_px", 6.0)),
            min_inter_action_delay_sec=float(safety_data.get("min_inter_action_delay_sec", 0.2)),
            target_similarity_threshold=float(safety_data.get("target_similarity_threshold", 0.82)),
            post_action_change_threshold=float(safety_data.get("post_action_change_threshold", 0.75)),
        ),
        paths=PathsSettings(
            output_dir=_resolve_path(paths_data.get("output_dir", "runs"), base_dir) or (_project_root() / "runs"),
        ),
    )


def _apply_env_overrides(config: AppConfig) -> None:
    env_map: dict[str, tuple[str, str, Any]] = {
        "VIDEOCAD_ONSHAPE_PLANNER_PROVIDER": ("planner", "provider", str),
        "VIDEOCAD_ONSHAPE_PLANNER_MODEL": ("planner", "model", str),
        "VIDEOCAD_ONSHAPE_PLANNER_API_KEY": ("planner", "api_key", str),
        "VIDEOCAD_ONSHAPE_PLANNER_BASE_URL": ("planner", "base_url", str),
        "VIDEOCAD_ONSHAPE_PLANNER_TIMEOUT_SECONDS": ("planner", "timeout_seconds", float),
        "VIDEOCAD_ONSHAPE_BROWSER_BACKEND": ("browser", "backend", str),
        "VIDEOCAD_ONSHAPE_BROWSER_PROFILE_PATH": ("browser", "profile_path", Path),
        "VIDEOCAD_ONSHAPE_BROWSER_PROFILE_NAME": ("browser", "profile_name", str),
        "VIDEOCAD_ONSHAPE_BROWSER_BLANK_PART_STUDIO_URL": ("browser", "blank_part_studio_url", str),
        "VIDEOCAD_ONSHAPE_MODEL_BACKEND": ("model", "backend", str),
        "VIDEOCAD_ONSHAPE_RUNPOD_ENDPOINT_ID": ("model", "runpod_endpoint_id", str),
        "VIDEOCAD_ONSHAPE_RUNPOD_API_KEY": ("model", "runpod_api_key", str),
        "VIDEOCAD_ONSHAPE_RUNPOD_BASE_URL": ("model", "runpod_base_url", str),
        "VIDEOCAD_ONSHAPE_RUNPOD_TIMEOUT_SECONDS": ("model", "runpod_timeout_seconds", float),
        "VIDEOCAD_ONSHAPE_WINDOW_X": ("window", "x", int),
        "VIDEOCAD_ONSHAPE_WINDOW_Y": ("window", "y", int),
        "VIDEOCAD_ONSHAPE_WINDOW_WIDTH": ("window", "width", int),
        "VIDEOCAD_ONSHAPE_WINDOW_HEIGHT": ("window", "height", int),
        "VIDEOCAD_ONSHAPE_OUTPUT_DIR": ("paths", "output_dir", Path),
        "VIDEOCAD_ONSHAPE_HEADLESS": ("browser", "headless", _coerce_bool),
        "VIDEOCAD_ONSHAPE_MAX_ACTIONS_PER_STEP": ("safety", "max_actions_per_step", int),
        "VIDEOCAD_ONSHAPE_MAX_RETRIES_PER_ACTION_CLASS": ("safety", "max_retries_per_action_class", int),
    }
    for env_name, (section_name, attr_name, parser) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        value = parser(raw)
        if parser is Path:
            value = Path(raw).expanduser()
        setattr(getattr(config, section_name), attr_name, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).expanduser().resolve() if path else None
    dotenv_candidates: list[Path] = []
    if config_path:
        dotenv_candidates.append(config_path.parent / ".env")
    dotenv_candidates.append(_project_root() / ".env")
    seen: set[Path] = set()
    for dotenv_path in dotenv_candidates:
        resolved = dotenv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_dotenv_file(resolved)
    if config_path and config_path.exists():
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
        config = _load_from_mapping(data, config_path)
    else:
        config = _load_from_mapping({}, config_path)
    _apply_env_overrides(config)
    return config
