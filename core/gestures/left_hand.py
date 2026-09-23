# core/gestures/left_hand.py
"""Left-hand safety controller.

The left hand no longer emits Windows shortcuts from ambiguous poses. It is
reserved for explicit mode controls handled by main.py, especially keyboard
activation.
"""

from __future__ import annotations

import logging

from core.actuator import MouseActuator
from core.tracker import Landmark

logger = logging.getLogger(__name__)


class LeftHandProcessor:
    """Deliberately inert in the default safe interaction profile."""

    def __init__(self, actuator: MouseActuator) -> None:
        self._actuator = actuator

    def process(self, landmarks: list[Landmark] | None) -> None:
        # Intentionally no OS actions here. This prevents accidental Win+D,
        # Alt+Tab, Win+Tab, Win-key, Ctrl, and navigation events.
        return

    def reset(self) -> None:
        return
