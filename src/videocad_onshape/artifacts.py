from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from PIL import Image, ImageDraw

from .action_codec import DecodedAction


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value)!r}")


@dataclass(slots=True)
class ArtifactManager:
    run_dir: Path
    targets_dir: Path
    frames_dir: Path
    visuals_dir: Path
    actions_log_path: Path
    summary_path: Path

    @classmethod
    def create(cls, root: Path) -> "ArtifactManager":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / timestamp
        targets_dir = run_dir / "targets"
        frames_dir = run_dir / "frames"
        visuals_dir = run_dir / "visuals"
        for directory in (run_dir, targets_dir, frames_dir, visuals_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            targets_dir=targets_dir,
            frames_dir=frames_dir,
            visuals_dir=visuals_dir,
            actions_log_path=run_dir / "actions.jsonl",
            summary_path=run_dir / "summary.json",
        )

    def save_target(self, step_index: int, image: Image.Image) -> Path:
        path = self.targets_dir / f"step_{step_index:03d}.png"
        image.save(path)
        return path

    def save_frame(self, step_index: int, cycle_index: int, image: Image.Image, kind: str) -> Path:
        path = self.frames_dir / f"step_{step_index:03d}_cycle_{cycle_index:03d}_{kind}.png"
        image.save(path)
        return path

    def save_action_visualization(self, step_index: int, cycle_index: int, frame: Image.Image, action: DecodedAction) -> Path:
        image = frame.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        if action.point:
            x, y = action.point
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(240, 78, 58), width=2)
            draw.line((x - 10, y, x + 10, y), fill=(240, 78, 58), width=2)
            draw.line((x, y - 10, x, y + 10), fill=(240, 78, 58), width=2)
        label = action.kind
        if action.text:
            label += f": {action.text}"
        draw.rounded_rectangle((8, 8, min(200, 10 + 7 * len(label)), 28), radius=6, fill=(255, 255, 255))
        draw.text((12, 13), label, fill=(30, 30, 30))
        path = self.visuals_dir / f"step_{step_index:03d}_cycle_{cycle_index:03d}.png"
        image.save(path)
        return path

    def append_action_log(self, payload: dict[str, Any]) -> None:
        with self.actions_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=_json_default) + "\n")

    def write_summary(self, payload: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)

