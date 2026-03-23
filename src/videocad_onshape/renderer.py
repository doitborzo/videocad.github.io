from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cad_ir import CADProgram, IRStep

IMAGE_SIZE = 224
BACKGROUND = (247, 246, 240)
GRID = (228, 226, 218)
SKETCH = (37, 83, 140)
SOLID_FILL = (143, 171, 182)
SOLID_EDGE = (54, 78, 88)
CUT_EDGE = (178, 80, 73)
ANNOTATION = (70, 70, 70)


@dataclass(slots=True)
class SketchPrimitive:
    kind: str
    plane: str
    params: dict[str, float | str]


@dataclass(slots=True)
class SolidFeature:
    kind: str
    plane: str
    params: dict[str, float | str]
    distance: float
    operation: str


@dataclass(slots=True)
class SceneState:
    active_plane: str = "front"
    active_sketch: list[SketchPrimitive] = field(default_factory=list)
    closed_sketch: list[SketchPrimitive] = field(default_factory=list)
    solids: list[SolidFeature] = field(default_factory=list)
    dimensions: list[tuple[str, float]] = field(default_factory=list)
    extent_hint: float = 40.0


@dataclass(slots=True)
class RenderedStep:
    index: int
    step: IRStep
    image: Image.Image


def _plane_point(plane: str, x: float, y: float, depth: float = 0.0) -> tuple[float, float, float]:
    if plane == "front":
        return (x, y, depth)
    if plane == "top":
        return (x, depth, y)
    return (depth, y, x)


def _plane_normal(plane: str) -> tuple[float, float, float]:
    if plane == "front":
        return (0.0, 0.0, 1.0)
    if plane == "top":
        return (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0)


def _project(point: tuple[float, float, float], scale: float) -> tuple[float, float]:
    x, y, z = point
    return (
        IMAGE_SIZE / 2 + scale * (x - 0.65 * z),
        IMAGE_SIZE * 0.64 - scale * (y + 0.35 * z),
    )


def _polygon_for_primitive(primitive: SketchPrimitive, samples: int = 40) -> list[tuple[float, float]]:
    params = primitive.params
    if primitive.kind == "rectangle":
        width = float(params["width"])
        height = float(params["height"])
        if params["mode"] == "centered":
            x0, y0 = -width / 2, -height / 2
        else:
            x0, y0 = 0.0, 0.0
        points = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
        return points

    if primitive.kind == "circle":
        cx = float(params["center_x"])
        cy = float(params["center_y"])
        radius = float(params["radius"])
        return [(cx + radius * cos(theta), cy + radius * sin(theta)) for theta in np.linspace(0, 2 * pi, samples)]

    if primitive.kind == "line":
        return [
            (float(params["x1"]), float(params["y1"])),
            (float(params["x2"]), float(params["y2"])),
        ]

    return []


class CanonicalRenderer:
    def __init__(self, image_size: int = IMAGE_SIZE) -> None:
        self.image_size = image_size
        self._font = ImageFont.load_default()

    def render_program_steps(self, program: CADProgram) -> list[RenderedStep]:
        scene = SceneState()
        outputs: list[RenderedStep] = []
        for index, step in enumerate(program.steps):
            self._apply_step(scene, step)
            outputs.append(RenderedStep(index=index, step=step, image=self.render_scene(scene, step)))
        return outputs

    def _apply_step(self, scene: SceneState, step: IRStep) -> None:
        if step.op == "select_plane":
            scene.active_plane = str(step.params["plane"])
            return
        if step.op == "start_sketch":
            scene.active_sketch = []
            scene.dimensions = []
            return
        if step.op == "draw_rectangle":
            scene.extent_hint = max(scene.extent_hint, float(step.params["width"]), float(step.params["height"]))
            scene.active_sketch.append(SketchPrimitive("rectangle", scene.active_plane, dict(step.params)))
            return
        if step.op == "draw_circle":
            scene.extent_hint = max(scene.extent_hint, float(step.params["radius"]) * 2)
            scene.active_sketch.append(SketchPrimitive("circle", scene.active_plane, dict(step.params)))
            return
        if step.op == "draw_line":
            for key in ("x1", "y1", "x2", "y2"):
                scene.extent_hint = max(scene.extent_hint, abs(float(step.params[key])))
            scene.active_sketch.append(SketchPrimitive("line", scene.active_plane, dict(step.params)))
            return
        if step.op == "add_dimension":
            scene.extent_hint = max(scene.extent_hint, float(step.params["value"]))
            scene.dimensions.append((str(step.params["target"]), float(step.params["value"])))
            return
        if step.op == "finish_sketch":
            scene.closed_sketch = list(scene.active_sketch)
            return
        if step.op in {"extrude_add", "extrude_cut"}:
            distance = float(step.params["distance"])
            scene.extent_hint = max(scene.extent_hint, distance)
            for primitive in scene.closed_sketch:
                if primitive.kind == "line":
                    continue
                scene.solids.append(
                    SolidFeature(
                        kind=primitive.kind,
                        plane=primitive.plane,
                        params=dict(primitive.params),
                        distance=distance,
                        operation=step.op,
                    )
                )

    def render_scene(self, scene: SceneState, step: IRStep) -> Image.Image:
        image = Image.new("RGB", (self.image_size, self.image_size), BACKGROUND)
        draw = ImageDraw.Draw(image)
        self._draw_grid(draw)

        scale = max(1.0, 68.0 / max(scene.extent_hint, 10.0))
        for solid in scene.solids:
            self._draw_solid(draw, solid, scale)
        for primitive in scene.closed_sketch:
            self._draw_sketch_primitive(draw, primitive, scale, emphasize=step.op != "finish_sketch")
        for primitive in scene.active_sketch:
            self._draw_sketch_primitive(draw, primitive, scale, emphasize=True)

        if scene.dimensions:
            label = ", ".join(f"{target}={value:g}" for target, value in scene.dimensions[-2:])
            draw.rounded_rectangle((8, 8, 152, 28), radius=6, fill=(255, 255, 255), outline=GRID)
            draw.text((12, 13), label, fill=ANNOTATION, font=self._font)

        draw.text((8, self.image_size - 18), step.op, fill=ANNOTATION, font=self._font)
        return image.convert("L")

    def _draw_grid(self, draw: ImageDraw.ImageDraw) -> None:
        for offset in range(16, self.image_size, 16):
            draw.line((0, offset, self.image_size, offset), fill=GRID, width=1)
            draw.line((offset, 0, offset, self.image_size), fill=GRID, width=1)

    def _draw_solid(self, draw: ImageDraw.ImageDraw, solid: SolidFeature, scale: float) -> None:
        base_2d = _polygon_for_primitive(SketchPrimitive(solid.kind, solid.plane, solid.params))
        if not base_2d:
            return
        normal = _plane_normal(solid.plane)
        front = [_project(_plane_point(solid.plane, x, y, 0.0), scale) for x, y in base_2d]
        back = [
            _project(
                (
                    _plane_point(solid.plane, x, y, 0.0)[0] + normal[0] * solid.distance,
                    _plane_point(solid.plane, x, y, 0.0)[1] + normal[1] * solid.distance,
                    _plane_point(solid.plane, x, y, 0.0)[2] + normal[2] * solid.distance,
                ),
                scale,
            )
            for x, y in base_2d
        ]
        for front_pt, back_pt in zip(front, back):
            draw.line((*front_pt, *back_pt), fill=SOLID_EDGE, width=2)
        fill = SOLID_FILL if solid.operation == "extrude_add" else BACKGROUND
        edge = SOLID_EDGE if solid.operation == "extrude_add" else CUT_EDGE
        draw.polygon(back, fill=fill, outline=edge)
        draw.polygon(front, fill=fill, outline=edge)

    def _draw_sketch_primitive(
        self,
        draw: ImageDraw.ImageDraw,
        primitive: SketchPrimitive,
        scale: float,
        emphasize: bool,
    ) -> None:
        projected = [_project(_plane_point(primitive.plane, x, y, 0.0), scale) for x, y in _polygon_for_primitive(primitive)]
        color = SKETCH if emphasize else ANNOTATION
        if primitive.kind == "line":
            draw.line((*projected[0], *projected[1]), fill=color, width=2)
        elif primitive.kind == "circle":
            draw.line(projected + [projected[0]], fill=color, width=2)
        else:
            draw.polygon(projected, outline=color)

