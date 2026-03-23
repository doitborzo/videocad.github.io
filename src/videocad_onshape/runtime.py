from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .action_codec import ActionCodec
from .artifacts import ArtifactManager
from .compat import dataclass
from .config import AppConfig
from .controller import AutonomousController, ExecutionSummary
from .native import DryRunNativeExecutor, MacOSNativeExecutor
from .onshape import build_session
from .planner import build_planner
from .renderer import CanonicalRenderer
from .videocad import build_predictor


@dataclass
class HealthcheckItem:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthcheckResult:
    items: list[HealthcheckItem]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "items": [item.to_dict() for item in self.items],
        }


class Application:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.planner = build_planner(config.planner)
        self.renderer = CanonicalRenderer()
        self.session = build_session(config.browser)
        self.predictor = build_predictor(config.model)

    def execute_prompt(self, prompt: str, mode: str) -> ExecutionSummary:
        program = self.planner.plan(prompt)
        rendered = [(item.step, item.image) for item in self.renderer.render_program_steps(program)]
        self.config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = ArtifactManager.create(self.config.paths.output_dir)
        native = DryRunNativeExecutor() if mode == "dry-run" else MacOSNativeExecutor(self.config.safety.min_inter_action_delay_sec)
        controller = AutonomousController(
            predictor=self.predictor,
            native=native,
            codec=ActionCodec(),
            safety=self.config.safety,
            artifacts=artifacts,
        )
        self.session.start()
        try:
            self.session.open_blank_part_studio()
            calibration = self.session.calibrate(self.config.window, self.config.safety.geometry_tolerance_px)
            summary = controller.execute(program, rendered, self.session, calibration, mode)
            artifacts.write_summary(
                {
                    "mode": mode,
                    "program": program.to_dict(),
                    "summary": summary.to_dict(),
                    "calibration": calibration.to_dict(),
                }
            )
            return summary
        finally:
            self.session.close()

    def calibrate(self) -> dict[str, Any]:
        self.session.start()
        try:
            self.session.open_blank_part_studio()
            calibration = self.session.calibrate(self.config.window, self.config.safety.geometry_tolerance_px)
            return calibration.to_dict()
        finally:
            self.session.close()

    def healthcheck(self) -> HealthcheckResult:
        items: list[HealthcheckItem] = []
        planner_ok, planner_detail = self.planner.healthcheck()
        items.append(HealthcheckItem(name="planner", ok=planner_ok, detail=planner_detail))

        predictor_ok, predictor_detail = self.predictor.healthcheck()
        items.append(HealthcheckItem(name="predictor", ok=predictor_ok, detail=predictor_detail))

        native = MacOSNativeExecutor(self.config.safety.min_inter_action_delay_sec)
        native_ok, native_detail = native.permissions_ok()
        items.append(HealthcheckItem(name="native-events", ok=native_ok, detail=native_detail))

        try:
            self.session.start()
            self.session.open_blank_part_studio()
            calibration = self.session.calibrate(self.config.window, self.config.safety.geometry_tolerance_px)
            items.append(
                HealthcheckItem(
                    name="onshape-canvas",
                    ok=True,
                    detail=f"canvas {calibration.canvas.width:.0f}x{calibration.canvas.height:.0f} detected",
                )
            )
            center_x, center_y = calibration.map_normalized_point(0.5, 0.5)
            items.append(
                HealthcheckItem(
                    name="coordinate-mapping",
                    ok=True,
                    detail=f"normalized center maps to ({center_x:.1f}, {center_y:.1f})",
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(HealthcheckItem(name="onshape-canvas", ok=False, detail=str(exc)))
            items.append(HealthcheckItem(name="coordinate-mapping", ok=False, detail="skipped because canvas calibration failed"))
        finally:
            self.session.close()

        return HealthcheckResult(items=items)
