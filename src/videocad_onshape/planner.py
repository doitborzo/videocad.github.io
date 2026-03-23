from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import json
import re
import urllib.error
import urllib.request

from .cad_ir import CADProgram
from .config import PlannerSettings
from .errors import ConfigurationError, UnsupportedPromptError

UNSUPPORTED_KEYWORDS = {
    "assembly": "assemblies are out of scope in v1",
    "assemblies": "assemblies are out of scope in v1",
    "mate": "mates are out of scope in v1",
    "mates": "mates are out of scope in v1",
    "fillet": "fillets are out of scope in v1",
    "fillets": "fillets are out of scope in v1",
    "chamfer": "chamfers are out of scope in v1",
    "chamfers": "chamfers are out of scope in v1",
    "drawing": "drawings are out of scope in v1",
    "drawings": "drawings are out of scope in v1",
    "spline": "splines are out of scope in v1",
    "splines": "splines are out of scope in v1",
    "import": "imported or arbitrary existing geometry is out of scope in v1",
    "mesh": "mesh workflows are out of scope in v1",
}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "half",
}

SYSTEM_PROMPT = """
You are a CAD planner for a constrained Onshape agent.
Return JSON with one top-level key: "steps".
Each step must be an object with:
- "op": one of new_part_studio, select_plane, start_sketch, draw_rectangle, draw_circle, draw_line, add_dimension, finish_sketch, extrude_add, extrude_cut
- "params": an object

Rules:
- Only produce steps supported by the allowed ops list.
- Reject unsupported tasks by returning {"error": "..."}.
- Always include new_part_studio as the first step.
- Use select_plane with exactly one of front, top, right.
- For draw_rectangle use params {mode, width, height}.
- For draw_circle use params {center_x, center_y, radius}.
- For draw_line use params {x1, y1, x2, y2}.
- For add_dimension use params {target, value}.
- For extrude_add / extrude_cut use params {distance}.
- Assume millimeter-style scalar values but do not include units.
- Do not invent ambiguous dimensions; return an error instead.
""".strip()


class Planner(Protocol):
    def plan(self, prompt: str) -> CADProgram:
        ...

    def healthcheck(self) -> tuple[bool, str]:
        ...


def guard_prompt(prompt: str) -> None:
    lowered = prompt.lower()
    for keyword, reason in UNSUPPORTED_KEYWORDS.items():
        if keyword in lowered:
            raise UnsupportedPromptError(f"Unsupported request: {reason}.")
    if not re.search(r"\d", lowered) and not any(word in lowered for word in NUMBER_WORDS):
        raise UnsupportedPromptError("Prompt is ambiguous: at least one explicit dimension is required.")


@dataclass(slots=True)
class OpenAICompatiblePlanner:
    settings: PlannerSettings

    def _request(self, prompt: str) -> dict[str, Any]:
        if not self.settings.api_key:
            raise ConfigurationError("planner.api_key is required for the configured planner provider.")

        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.settings.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise ConfigurationError(f"Planner API request failed: {exc.code} {message}") from exc
        except urllib.error.URLError as exc:
            raise ConfigurationError(f"Planner API request failed: {exc}") from exc

        message = body["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        parsed = json.loads(message)
        if "error" in parsed:
            raise UnsupportedPromptError(str(parsed["error"]))
        return parsed

    def plan(self, prompt: str) -> CADProgram:
        guard_prompt(prompt)
        return CADProgram.from_dict(prompt, self._request(prompt))

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self._request("Create a centered rectangle 10 by 20 and extrude it by 5.")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, "planner API reachable"


@dataclass(slots=True)
class RuleBasedPlanner:
    settings: PlannerSettings | None = None

    def plan(self, prompt: str) -> CADProgram:
        guard_prompt(prompt)
        lowered = prompt.lower()
        plane = "front"
        for candidate in ("front", "top", "right"):
            if candidate in lowered:
                plane = candidate
                break

        rectangle_match = re.search(r"rectangle.*?(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)", lowered)
        circle_match = re.search(r"circle.*?radius\s*(\d+(?:\.\d+)?)", lowered)
        line_match = re.search(
            r"line.*?\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?.*?\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?",
            lowered,
        )
        extrude_match = re.search(r"(extrude|cut).*?(\d+(?:\.\d+)?)", lowered)

        steps: list[dict[str, Any]] = [
            {"op": "new_part_studio", "params": {}},
            {"op": "select_plane", "params": {"plane": plane}},
            {"op": "start_sketch", "params": {}},
        ]

        if rectangle_match:
            width = float(rectangle_match.group(1))
            height = float(rectangle_match.group(2))
            mode = "centered" if "center" in lowered else "corner"
            steps.extend(
                [
                    {"op": "draw_rectangle", "params": {"mode": mode, "width": width, "height": height}},
                    {"op": "add_dimension", "params": {"target": "rectangle.width", "value": width}},
                    {"op": "add_dimension", "params": {"target": "rectangle.height", "value": height}},
                ]
            )
        elif circle_match:
            radius = float(circle_match.group(1))
            steps.extend(
                [
                    {"op": "draw_circle", "params": {"center_x": 0, "center_y": 0, "radius": radius}},
                    {"op": "add_dimension", "params": {"target": "circle.radius", "value": radius}},
                ]
            )
        elif line_match:
            steps.append(
                {
                    "op": "draw_line",
                    "params": {
                        "x1": float(line_match.group(1)),
                        "y1": float(line_match.group(2)),
                        "x2": float(line_match.group(3)),
                        "y2": float(line_match.group(4)),
                    },
                }
            )
        else:
            raise UnsupportedPromptError("Rule-based planner could not normalize this prompt into the v1 CAD IR.")

        steps.append({"op": "finish_sketch", "params": {}})

        if extrude_match:
            distance = float(extrude_match.group(2))
            op = "extrude_cut" if "cut" in extrude_match.group(1) or "hole" in lowered or "pocket" in lowered else "extrude_add"
            steps.append({"op": op, "params": {"distance": distance}})

        return CADProgram.from_dict(prompt, {"steps": steps})

    def healthcheck(self) -> tuple[bool, str]:
        return True, "rule-based planner ready"


def build_planner(settings: PlannerSettings) -> Planner:
    if settings.provider == "mock":
        return RuleBasedPlanner(settings=settings)
    return OpenAICompatiblePlanner(settings=settings)

