# core/gestures/two_hand.py
"""Deliberate two-hand gestures.

The only automatic two-hand power gesture in the safe profile is zoom. It
requires both hands to hold a stable thumb-index pinch and then move apart or
together far enough to create one zoom event.
"""

from __future__ import annotations

import logging
import time

from config import (
    GESTURE_COOLDOWN_SECONDS,
    ZOOM_COOLDOWN_S,
    ZOOM_STABILITY_FRAMES,
    ZOOM_WRIST_DELTA,
)
from core.actuator import MouseActuator
from core.gestures.utils import normalized_distance
from core.tracker import Landmark

logger = logging.getLogger(__name__)

_PINCH_RATIO = 0.24


class TwoHandProcessor:
    def __init__(self, actuator: MouseActuator) -> None:
        self._actuator = actuator
        self._stable_frames = 0
        self._baseline_distance: float | None = None
        self._fired = False
        self._last_action = 0.0

    def reset(self) -> None:
        self._stable_frames = 0
        self._baseline_distance = None
        self._fired = False

    def process(self, left_lm: list[Landmark], right_lm: list[Landmark]) -> bool:
        now = time.perf_counter()
        left_pinched = normalized_distance(left_lm, 4, 8) <= _PINCH_RATIO
        right_pinched = normalized_distance(right_lm, 4, 8) <= _PINCH_RATIO

        if not (left_pinched and right_pinched):
            self.reset()
            return False

        self._stable_frames += 1
        if self._stable_frames < ZOOM_STABILITY_FRAMES:
            return True

        wrist_distance = abs(right_lm[0].x - left_lm[0].x)
        if self._baseline_distance is None:
            self._baseline_distance = wrist_distance
            return True

        delta = wrist_distance - self._baseline_distance
        if (
            not self._fired
            and abs(delta) >= ZOOM_WRIST_DELTA
            and now - self._last_action >= max(ZOOM_COOLDOWN_S, GESTURE_COOLDOWN_SECONDS)
        ):
            if delta > 0:
                self._actuator.zoom_in()
                logger.info("Two-hand zoom in")
            else:
                self._actuator.zoom_out()
                logger.info("Two-hand zoom out")
            self._last_action = now
            self._fired = True

        # Returning True suppresses single-hand click processing while both
        # hands are deliberately pinching.
        return True
