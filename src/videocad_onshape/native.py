from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import time


class NativeExecutor(Protocol):
    def permissions_ok(self) -> tuple[bool, str]:
        ...

    def execute(self, action_kind: str, **kwargs: object) -> None:
        ...


@dataclass(slots=True)
class DryRunNativeExecutor:
    events: list[dict[str, object]] = field(default_factory=list)

    def permissions_ok(self) -> tuple[bool, str]:
        return True, "dry-run executor active"

    def execute(self, action_kind: str, **kwargs: object) -> None:
        payload = {"kind": action_kind, **kwargs}
        self.events.append(payload)


@dataclass(slots=True)
class MacOSNativeExecutor:
    inter_action_delay_sec: float = 0.2

    def permissions_ok(self) -> tuple[bool, str]:
        try:
            from ApplicationServices import AXIsProcessTrusted  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return False, f"Accessibility API unavailable: {exc}"
        trusted = bool(AXIsProcessTrusted())
        return trusted, ("Accessibility permission granted" if trusted else "Grant Accessibility permission to terminal/python")

    def execute(self, action_kind: str, **kwargs: object) -> None:
        import Quartz  # type: ignore

        if action_kind == "move":
            point = kwargs["point"]
            Quartz.CGWarpMouseCursorPosition(point)
        elif action_kind == "click":
            point = kwargs.get("point")
            if point:
                Quartz.CGWarpMouseCursorPosition(point)
            event_down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, point or (0, 0), Quartz.kCGMouseButtonLeft)
            event_up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, point or (0, 0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
        elif action_kind == "scroll":
            direction = int(kwargs.get("direction", 0))
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 1, direction * 8))
        elif action_kind == "keys":
            keys = tuple(kwargs["keys"])
            repeat = int(kwargs.get("repeat", 1))
            self._press_keys(keys, repeat)
        elif action_kind == "type":
            text = str(kwargs.get("text", ""))
            self._type_text(text)
        time.sleep(self.inter_action_delay_sec)

    def _press_keys(self, keys: tuple[str, ...], repeat: int) -> None:
        import Quartz  # type: ignore

        keycodes = {
            "a": 0,
            "c": 8,
            "l": 37,
            "y": 16,
            "tab": 48,
            "space": 49,
            "enter": 36,
            "esc": 53,
            "up": 126,
            "down": 125,
            "left": 123,
            "right": 124,
            "shift": 56,
            "s": 1,
            "e": 14,
            "h": 4,
            "p": 35,
            "0": 29,
            "1": 18,
            "7": 26,
        }
        for _ in range(repeat):
            for key in keys:
                code = keycodes[key]
                event = Quartz.CGEventCreateKeyboardEvent(None, code, True)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            for key in reversed(keys):
                code = keycodes[key]
                event = Quartz.CGEventCreateKeyboardEvent(None, code, False)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def _type_text(self, text: str) -> None:
        import Quartz  # type: ignore

        event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        event_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(event_up, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)

