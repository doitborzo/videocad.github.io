from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

from .calibration import CanvasCalibration, build_calibration
from .compat import dataclass
from .config import BrowserSettings, WindowGeometry
from .errors import ConfigurationError, LiveRuntimeError


class OnshapeSession(Protocol):
    def start(self) -> None:
        ...

    def open_blank_part_studio(self) -> None:
        ...

    def calibrate(self, window: WindowGeometry, tolerance_px: float) -> CanvasCalibration:
        ...

    def capture_frame(self) -> Image.Image:
        ...

    def close(self) -> None:
        ...


@dataclass
class PlaywrightOnshapeSession:
    settings: BrowserSettings
    _playwright: object | None = None
    _browser_context: object | None = None
    _page: object | None = None

    def _resolve_chrome_profile(self) -> tuple[Path, str | None]:
        assert self.settings.profile_path is not None
        profile_path = self.settings.profile_path.expanduser()
        profile_name = self.settings.profile_name
        if profile_path.name == "Default" or profile_path.name.startswith("Profile "):
            return profile_path.parent, profile_path.name
        return profile_path, profile_name

    def start(self) -> None:
        if self.settings.profile_path is None:
            raise ConfigurationError("browser.profile_path is required for Playwright mode.")
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Playwright is not installed: {exc}") from exc

        self._playwright = sync_playwright().start()
        chromium = self._playwright.chromium
        user_data_dir, profile_name = self._resolve_chrome_profile()
        args = []
        if profile_name:
            args.append(f"--profile-directory={profile_name}")
        self._browser_context = chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=self.settings.headless,
            viewport=None,
            channel="chrome",
            args=args,
        )
        pages = self._browser_context.pages
        self._page = pages[0] if pages else self._browser_context.new_page()

    def open_blank_part_studio(self) -> None:
        if self._page is None:
            raise LiveRuntimeError("Playwright page is not initialized.")
        if not self.settings.blank_part_studio_url:
            raise ConfigurationError("browser.blank_part_studio_url is required for live Onshape runs.")
        self._page.goto(self.settings.blank_part_studio_url, wait_until="domcontentloaded")

    def calibrate(self, window: WindowGeometry, tolerance_px: float) -> CanvasCalibration:
        if self._page is None:
            raise LiveRuntimeError("Playwright page is not initialized.")

        window_snapshot = self._page.evaluate(
            "() => ({x: window.screenX || 0, y: window.screenY || 0, width: window.outerWidth || window.innerWidth, height: window.outerHeight || window.innerHeight})"
        )
        canvas_box = None
        for selector in self.settings.canvas_selectors:
            locator = self._page.locator(selector).first
            if locator.count() == 0:
                continue
            box = locator.bounding_box()
            if box:
                canvas_box = box
                break
        if canvas_box is None and self.settings.canvas_override is not None:
            canvas_box = self.settings.canvas_override
        if canvas_box is None:
            raise LiveRuntimeError("Could not detect the Onshape canvas.")
        return build_calibration(window_snapshot, canvas_box, window, tolerance_px, source="playwright")

    def capture_frame(self) -> Image.Image:
        if self._page is None:
            raise LiveRuntimeError("Playwright page is not initialized.")
        image_bytes = self._page.screenshot(type="png")
        return Image.open(BytesIO(image_bytes)).convert("L").resize((224, 224))

    def close(self) -> None:
        if self._browser_context is not None:
            self._browser_context.close()
        if self._playwright is not None:
            self._playwright.stop()


def build_session(settings: BrowserSettings) -> OnshapeSession:
    if settings.backend != "playwright":
        raise ConfigurationError(f"Unsupported browser backend: {settings.backend}")
    return PlaywrightOnshapeSession(settings)
