from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import UnsupportedPromptError

DRAWING_OPS = {"draw_rectangle", "draw_circle", "draw_line", "add_dimension"}
EXTRUDE_OPS = {"extrude_add", "extrude_cut"}
SUPPORTED_PLANES = {"front", "top", "right"}
SUPPORTED_OPS = {
    "new_part_studio",
    "select_plane",
    "start_sketch",
    "draw_rectangle",
    "draw_circle",
    "draw_line",
    "add_dimension",
    "finish_sketch",
    "extrude_add",
    "extrude_cut",
}


def _require_positive(step: str, name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedPromptError(f"{step}.{name} must be numeric.") from exc
    if number <= 0:
        raise UnsupportedPromptError(f"{step}.{name} must be greater than 0.")
    return number


def _require_number(step: str, name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedPromptError(f"{step}.{name} must be numeric.") from exc


@dataclass(slots=True)
class IRStep:
    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.op not in SUPPORTED_OPS:
            raise UnsupportedPromptError(f"Unsupported IR op: {self.op}")

        if self.op == "select_plane":
            plane = str(self.params.get("plane", "")).lower()
            if plane not in SUPPORTED_PLANES:
                raise UnsupportedPromptError("select_plane.plane must be one of front, top, right.")
            self.params["plane"] = plane
            return

        if self.op == "draw_rectangle":
            mode = str(self.params.get("mode", "")).lower()
            if mode not in {"centered", "corner"}:
                raise UnsupportedPromptError("draw_rectangle.mode must be centered or corner.")
            self.params["mode"] = mode
            self.params["width"] = _require_positive(self.op, "width", self.params.get("width"))
            self.params["height"] = _require_positive(self.op, "height", self.params.get("height"))
            return

        if self.op == "draw_circle":
            self.params["center_x"] = _require_number(self.op, "center_x", self.params.get("center_x", 0))
            self.params["center_y"] = _require_number(self.op, "center_y", self.params.get("center_y", 0))
            self.params["radius"] = _require_positive(self.op, "radius", self.params.get("radius"))
            return

        if self.op == "draw_line":
            x1 = _require_number(self.op, "x1", self.params.get("x1"))
            y1 = _require_number(self.op, "y1", self.params.get("y1"))
            x2 = _require_number(self.op, "x2", self.params.get("x2"))
            y2 = _require_number(self.op, "y2", self.params.get("y2"))
            if x1 == x2 and y1 == y2:
                raise UnsupportedPromptError("draw_line endpoints must differ.")
            self.params.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            return

        if self.op == "add_dimension":
            target = str(self.params.get("target", "")).strip()
            if not target:
                raise UnsupportedPromptError("add_dimension.target is required.")
            self.params["target"] = target
            self.params["value"] = _require_positive(self.op, "value", self.params.get("value"))
            return

        if self.op in EXTRUDE_OPS:
            self.params["distance"] = _require_positive(self.op, "distance", self.params.get("distance"))

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "params": self.params}


@dataclass(slots=True)
class CADProgram:
    prompt: str
    steps: list[IRStep]

    @classmethod
    def from_dict(cls, prompt: str, data: dict[str, Any]) -> "CADProgram":
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise UnsupportedPromptError("Planner returned no executable steps.")
        steps = []
        for raw in raw_steps:
            if not isinstance(raw, dict) or "op" not in raw:
                raise UnsupportedPromptError("Each planner step must be an object with op and params.")
            params = raw.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise UnsupportedPromptError(f"Step params for {raw.get('op')} must be an object.")
            steps.append(IRStep(op=str(raw["op"]).strip(), params=params))
        program = cls(prompt=prompt, steps=steps)
        program.validate()
        return program

    def validate(self) -> None:
        if not self.steps:
            raise UnsupportedPromptError("Program must contain at least one step.")
        if self.steps[0].op != "new_part_studio":
            self.steps.insert(0, IRStep("new_part_studio", {}))

        in_sketch = False
        sketch_has_geometry = False
        sketch_ready_for_extrude = False
        has_part_studio = False

        for index, step in enumerate(self.steps):
            step.validate()

            if step.op == "new_part_studio":
                if index != 0:
                    raise UnsupportedPromptError("new_part_studio may only appear as the first step.")
                has_part_studio = True
                in_sketch = False
                sketch_has_geometry = False
                sketch_ready_for_extrude = False
                continue

            if not has_part_studio:
                raise UnsupportedPromptError("Program must begin with new_part_studio.")

            if step.op == "select_plane":
                if in_sketch:
                    raise UnsupportedPromptError("Cannot change plane while a sketch is active.")
                continue

            if step.op == "start_sketch":
                if in_sketch:
                    raise UnsupportedPromptError("A sketch is already active.")
                in_sketch = True
                sketch_has_geometry = False
                continue

            if step.op in {"draw_rectangle", "draw_circle", "draw_line"}:
                if not in_sketch:
                    raise UnsupportedPromptError(f"{step.op} requires an active sketch.")
                sketch_has_geometry = True
                continue

            if step.op == "add_dimension":
                if not in_sketch:
                    raise UnsupportedPromptError("add_dimension requires an active sketch.")
                continue

            if step.op == "finish_sketch":
                if not in_sketch:
                    raise UnsupportedPromptError("finish_sketch requires an active sketch.")
                if not sketch_has_geometry:
                    raise UnsupportedPromptError("finish_sketch requires at least one sketch primitive.")
                in_sketch = False
                sketch_ready_for_extrude = True
                continue

            if step.op in EXTRUDE_OPS:
                if in_sketch:
                    raise UnsupportedPromptError(f"{step.op} cannot run while a sketch is active.")
                if not sketch_ready_for_extrude:
                    raise UnsupportedPromptError(f"{step.op} requires a completed sketch.")
                sketch_ready_for_extrude = False

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "steps": [step.to_dict() for step in self.steps]}

