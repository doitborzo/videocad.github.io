from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import Rect, WindowGeometry
from .errors import LiveRuntimeError


@dataclass(slots=True)
class ScreenRect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(slots=True)
class CanvasCalibration:
    window: ScreenRect
    canvas: ScreenRect
    source: str

    def map_normalized_point(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        return (
            self.canvas.x + norm_x * self.canvas.width,
            self.canvas.y + norm_y * self.canvas.height,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": asdict(self.window),
            "canvas": asdict(self.canvas),
            "source": self.source,
        }


def _within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def build_calibration(
    window_snapshot: dict[str, float],
    canvas_box: dict[str, float] | Rect,
    expected_window: WindowGeometry,
    tolerance_px: float,
    source: str,
) -> CanvasCalibration:
    window = ScreenRect(
        x=float(window_snapshot["x"]),
        y=float(window_snapshot["y"]),
        width=float(window_snapshot["width"]),
        height=float(window_snapshot["height"]),
    )

    canvas = ScreenRect(
        x=float(canvas_box.x if isinstance(canvas_box, Rect) else canvas_box["x"]),
        y=float(canvas_box.y if isinstance(canvas_box, Rect) else canvas_box["y"]),
        width=float(canvas_box.width if isinstance(canvas_box, Rect) else canvas_box["width"]),
        height=float(canvas_box.height if isinstance(canvas_box, Rect) else canvas_box["height"]),
    )

    if not _within_tolerance(window.width, expected_window.width, tolerance_px) or not _within_tolerance(
        window.height, expected_window.height, tolerance_px
    ):
        raise LiveRuntimeError(
            f"Window geometry {window.width}x{window.height} does not match expected "
            f"{expected_window.width}x{expected_window.height}."
        )

    return CanvasCalibration(window=window, canvas=canvas, source=source)
