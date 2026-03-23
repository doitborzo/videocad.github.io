from __future__ import annotations

from .compat import dataclass

from .calibration import CanvasCalibration

KEY_BY_BIN = {
    0: ("a",),
    1: ("l",),
    2: ("c",),
    3: ("y",),
    4: ("tab",),
    5: ("space",),
    6: ("enter",),
    7: ("up",),
    8: ("down",),
    9: ("left",),
    10: ("right",),
    11: ("esc",),
    12: ("shift", "s"),
    13: ("shift", "e"),
    14: ("shift", "y"),
    15: ("shift", "h"),
    16: ("shift", "p"),
    17: ("shift", "0"),
    18: ("shift", "1"),
    19: ("shift", "7"),
}


@dataclass
class RawModelAction:
    command: int
    params: list[int]

    def to_vector(self) -> list[int]:
        params = list(self.params[:6])
        while len(params) < 6:
            params.append(-1)
        return [self.command, *params]


@dataclass
class StepContext:
    step_index: int
    op: str
    pending_numeric_text: str | None = None


@dataclass
class DecodedAction:
    kind: str
    point: tuple[float, float] | None = None
    keys: tuple[str, ...] | None = None
    repeat: int = 1
    text: str | None = None
    scroll_direction: int = 0
    raw: RawModelAction | None = None


class ActionCodec:
    def decode(
        self,
        raw_action: RawModelAction,
        calibration: CanvasCalibration,
        step_context: StepContext,
    ) -> DecodedAction:
        params = list(raw_action.params[:6]) + [-1] * (6 - len(raw_action.params))

        if raw_action.command == 0:
            x_bin = max(0, min(999, params[0]))
            y_bin = max(0, min(999, params[1]))
            norm_x = max(0.0, min(1.0, (x_bin - 15) / 1000.0))
            norm_y = max(0.0, min(1.0, y_bin / 1000.0))
            point = calibration.map_normalized_point(norm_x, norm_y)
            return DecodedAction(kind="move", point=point, raw=raw_action)

        if raw_action.command == 1:
            key_index = max(0, params[2] // 50) if params[2] >= 0 else 0
            repeat = max(1, (params[3] // 200) + 2) if params[3] >= 0 else 1
            return DecodedAction(kind="keys", keys=KEY_BY_BIN.get(key_index, ("enter",)), repeat=repeat, raw=raw_action)

        if raw_action.command == 2:
            direction = 1 if params[4] >= 500 else -1
            return DecodedAction(kind="scroll", scroll_direction=direction, raw=raw_action)

        if raw_action.command == 3:
            text = step_context.pending_numeric_text
            if text is None:
                text = str(params[5] if params[5] >= 0 else "")
            return DecodedAction(kind="type", text=text, raw=raw_action)

        return DecodedAction(kind="click", raw=raw_action)
