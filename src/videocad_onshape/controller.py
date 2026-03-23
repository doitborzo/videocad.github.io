from __future__ import annotations

from dataclasses import asdict
from signal import SIGINT, SIGTERM, signal
from time import sleep
from typing import Any

import numpy as np
from PIL import Image

from .action_codec import ActionCodec, DecodedAction, StepContext
from .artifacts import ArtifactManager
from .cad_ir import CADProgram, IRStep
from .calibration import CanvasCalibration
from .compat import dataclass
from .config import SafetySettings
from .errors import LiveRuntimeError, SafetyStopError
from .native import NativeExecutor
from .videocad import Predictor


@dataclass
class ExecutionSummary:
    mode: str
    run_dir: str
    status: str
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmergencyStopMonitor:
    def __init__(self) -> None:
        self.triggered = False
        signal(SIGINT, self._handle_signal)
        signal(SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self.triggered = True

    def check(self) -> None:
        if self.triggered:
            raise SafetyStopError("Emergency stop requested.")


class AutonomousController:
    def __init__(
        self,
        predictor: Predictor,
        native: NativeExecutor,
        codec: ActionCodec,
        safety: SafetySettings,
        artifacts: ArtifactManager,
    ) -> None:
        self.predictor = predictor
        self.native = native
        self.codec = codec
        self.safety = safety
        self.artifacts = artifacts
        self.stop_monitor = EmergencyStopMonitor()

    def execute(
        self,
        program: CADProgram,
        rendered_steps: list[tuple[IRStep, Image.Image]],
        session: Any,
        calibration: CanvasCalibration,
        mode: str,
    ) -> ExecutionSummary:
        action_history: list[list[int]] = []
        summary_status = "completed"
        for step_index, (step, target_image) in enumerate(rendered_steps):
            self.artifacts.save_target(step_index, target_image)
            if step.op == "new_part_studio":
                continue
            if mode == "dry-run":
                self._dry_run_step(step_index, step, target_image, session, calibration, action_history)
                continue
            self._run_step(step_index, step, target_image, session, calibration, action_history)

        return ExecutionSummary(
            mode=mode,
            run_dir=str(self.artifacts.run_dir),
            status=summary_status,
            step_count=len(rendered_steps),
        )

    def _dry_run_step(
        self,
        step_index: int,
        step: IRStep,
        target_image: Image.Image,
        session: Any,
        calibration: CanvasCalibration,
        action_history: list[list[int]],
    ) -> None:
        frame = session.capture_frame()
        self.artifacts.save_frame(step_index, 0, frame, "input")
        raw_action = self.predictor.predict_next_action([frame], action_history, target_image, self._step_context(step_index, step))
        decoded = self.codec.decode(raw_action, calibration, self._step_context(step_index, step))
        self.artifacts.save_action_visualization(step_index, 0, frame, decoded)
        self.artifacts.append_action_log(
            {
                "step_index": step_index,
                "mode": "dry-run",
                "step": step.to_dict(),
                "raw_action": raw_action.to_vector(),
                "decoded_action": asdict(decoded),
            }
        )
        action_history.append(raw_action.to_vector())

    def _run_step(
        self,
        step_index: int,
        step: IRStep,
        target_image: Image.Image,
        session: Any,
        calibration: CanvasCalibration,
        action_history: list[list[int]],
    ) -> None:
        history_frames: list[Image.Image] = []
        repeated_action_counts: dict[str, int] = {}
        low_delta_cycles = 0

        for cycle in range(self.safety.max_actions_per_step):
            self.stop_monitor.check()
            frame = session.capture_frame()
            self.artifacts.save_frame(step_index, cycle, frame, "before")
            history_frames.append(frame)
            history_frames = history_frames[-10:]

            similarity = self._similarity(frame, target_image)
            if similarity >= self.safety.target_similarity_threshold and cycle > 0:
                return

            context = self._step_context(step_index, step)
            raw_action = self.predictor.predict_next_action(history_frames, action_history, target_image, context)
            decoded = self.codec.decode(raw_action, calibration, context)
            repeated_action_counts[decoded.kind] = repeated_action_counts.get(decoded.kind, 0) + 1
            if repeated_action_counts[decoded.kind] > self.safety.max_retries_per_action_class:
                raise LiveRuntimeError(f"Exceeded retry limit for action kind {decoded.kind} on step {step_index}.")

            self._execute_native(decoded)
            after_frame = session.capture_frame()
            self.artifacts.save_frame(step_index, cycle, after_frame, "after")
            delta = self._delta(frame, after_frame)
            if delta < self.safety.post_action_change_threshold:
                low_delta_cycles += 1
            else:
                low_delta_cycles = 0
            if low_delta_cycles >= self.safety.stall_patience:
                raise LiveRuntimeError(f"UI stalled on step {step_index}; screenshots stopped changing.")

            self.artifacts.append_action_log(
                {
                    "step_index": step_index,
                    "mode": "run",
                    "cycle_index": cycle,
                    "step": step.to_dict(),
                    "raw_action": raw_action.to_vector(),
                    "decoded_action": asdict(decoded),
                    "target_similarity": similarity,
                    "post_action_delta": delta,
                }
            )
            action_history.append(raw_action.to_vector())
            sleep(self.safety.min_inter_action_delay_sec)

        raise LiveRuntimeError(f"Exceeded max actions for step {step_index} ({step.op}).")

    def _execute_native(self, action: DecodedAction) -> None:
        if action.kind == "move":
            self.native.execute("move", point=action.point)
            return
        if action.kind == "click":
            self.native.execute("click")
            return
        if action.kind == "scroll":
            self.native.execute("scroll", direction=action.scroll_direction)
            return
        if action.kind == "keys":
            self.native.execute("keys", keys=action.keys or (), repeat=action.repeat)
            return
        if action.kind == "type":
            self.native.execute("type", text=action.text or "")

    def _step_context(self, step_index: int, step: IRStep) -> StepContext:
        numeric_text = None
        if step.op == "add_dimension":
            numeric_text = f"{float(step.params['value']):g}"
        if step.op in {"extrude_add", "extrude_cut"}:
            numeric_text = f"{float(step.params['distance']):g}"
        return StepContext(step_index=step_index, op=step.op, pending_numeric_text=numeric_text)

    def _delta(self, before: Image.Image, after: Image.Image) -> float:
        before_arr = np.asarray(before.convert("L"), dtype=np.float32)
        after_arr = np.asarray(after.convert("L"), dtype=np.float32)
        return float(np.mean(np.abs(before_arr - after_arr)))

    def _similarity(self, live: Image.Image, target: Image.Image) -> float:
        live_arr = np.asarray(live.convert("L").resize(target.size), dtype=np.float32)
        target_arr = np.asarray(target.convert("L"), dtype=np.float32)
        delta = np.mean(np.abs(live_arr - target_arr)) / 255.0
        return max(0.0, 1.0 - float(delta))
