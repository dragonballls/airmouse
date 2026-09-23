"""Windows UI Automation helpers for contextual typing mode."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

_TEXT_CONTROL_TYPES = {"Edit", "Document", "ComboBox"}


class TextInputDetector:
    """Low-frequency UIA hit-testing with a short cursor dwell trigger."""

    def __init__(
        self,
        dwell_seconds: float = 0.65,
        poll_seconds: float = 0.20,
        move_tolerance_px: float = 14.0,
    ) -> None:
        self.dwell_seconds = dwell_seconds
        self.poll_seconds = poll_seconds
        self.move_tolerance_px = move_tolerance_px
        self._desktop: Any | None = None
        self._next_poll = 0.0
        self._candidate_key: str | None = None
        self._candidate_started = 0.0
        self._last_point: tuple[int, int] | None = None
        self._last_result = False
        self._last_descriptor = ""
        self._available = True

        try:
            from pywinauto import Desktop
            self._desktop = Desktop(backend="uia")
        except Exception as exc:
            self._available = False
            logger.warning("UI Automation text-input detection unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_descriptor(self) -> str:
        return self._last_descriptor

    def _element_key(self, element: Any) -> str:
        try:
            info = element.element_info
            return "|".join(
                [
                    str(getattr(info, "handle", "")),
                    str(getattr(info, "automation_id", "")),
                    str(getattr(info, "runtime_id", "")),
                    str(getattr(info, "name", "")),
                ]
            )
        except Exception:
            return repr(element)

    def _is_text_control(self, element: Any) -> bool:
        try:
            info = element.element_info
            control_type = str(getattr(info, "control_type", "") or "")
            if control_type in _TEXT_CONTROL_TYPES:
                is_editable = getattr(element, "is_editable", None)
                if callable(is_editable):
                    try:
                        return bool(is_editable())
                    except Exception:
                        pass
                return True
        except Exception:
            return False
        return False

    def _hit_test(self, x: int, y: int) -> tuple[bool, str, str | None]:
        if self._desktop is None:
            return False, "", None
        try:
            element = self._desktop.from_point(x, y)
            if not self._is_text_control(element):
                return False, "", None
            info = element.element_info
            control_type = str(getattr(info, "control_type", "") or "text input")
            name = str(getattr(info, "name", "") or "").strip()
            descriptor = f"{control_type}: {name}" if name else control_type
            return True, descriptor, self._element_key(element)
        except Exception as exc:
            logger.debug("UIA hit-test failed: %s", exc)
            return False, "", None

    def update(self, x: int, y: int, now: float | None = None) -> bool:
        """Return True once the pointer dwells over a text input."""
        if not self._available:
            return False

        now = time.perf_counter() if now is None else now
        if now < self._next_poll:
            return self._last_result

        self._next_poll = now + self.poll_seconds
        hit, descriptor, key = self._hit_test(x, y)

        if not hit or key is None:
            self._last_point = None

            self._candidate_key = None
            self._candidate_started = 0.0
            self._last_result = False
            self._last_descriptor = ""
            return False

        moved = False
        if self._last_point is not None:
            dx = x - self._last_point[0]
            dy = y - self._last_point[1]
            moved = math.hypot(dx, dy) > self.move_tolerance_px
        self._last_point = (x, y)

        if moved or key != self._candidate_key:
            self._candidate_key = key
            self._candidate_started = now
            self._last_result = False
        else:
            self._last_result = (now - self._candidate_started) >= self.dwell_seconds

        self._last_descriptor = descriptor
        return self._last_result

    def reset(self) -> None:
        self._candidate_key = None
        self._candidate_started = 0.0
        self._last_result = False
        self._last_descriptor = ""
