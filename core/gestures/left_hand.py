# core/gestures/left_hand.py
"""AirNav-style left-hand cursor control plus persistent mouse locking.

When both hands are visible, the left index finger moves the cursor and the
right hand is reserved for click/drag actions. A left-hand fist held briefly
toggles the cursor lock, so the user can reposition or rest their hand without
moving the Windows cursor.
"""

from __future__ import annotations

import logging
import time

from config import (
    CAMERA_FPS,
    GESTURE_STABILITY_FRAMES,
    ONE_EURO_BETA,
    ONE_EURO_DCUTOFF,
    ONE_EURO_MINCUTOFF,
)
from core.actuator import MouseActuator
from core.display import TrackpadZone, VirtualDesktop, map_to_desktop
from core.filter import OneEuroFilter
from core.gestures.utils import is_fist
from core.tracker import Landmark

logger = logging.getLogger(__name__)

# Deliberate long-enough hold to avoid accidental locking while still feeling
# much easier than the old gesture timings.
LOCK_HOLD_SECONDS = 0.40


class LeftHandProcessor:
    def __init__(
        self,
        actuator: MouseActuator,
        desktop: VirtualDesktop | None = None,
        trackpad: TrackpadZone | None = None,
    ) -> None:
        self._actuator = actuator
        self._desktop = desktop
        self._trackpad = trackpad

        self._filter_x = OneEuroFilter(
            freq=float(CAMERA_FPS),
            mincutoff=ONE_EURO_MINCUTOFF,
            beta=ONE_EURO_BETA,
            dcutoff=ONE_EURO_DCUTOFF,
        )
        self._filter_y = OneEuroFilter(
            freq=float(CAMERA_FPS),
            mincutoff=ONE_EURO_MINCUTOFF,
            beta=ONE_EURO_BETA,
            dcutoff=ONE_EURO_DCUTOFF,
        )

        self._fist_started: float | None = None
        self._lock_latched = False
        self._fist_stable = 0

    def process(
        self,
        landmarks: list[Landmark] | None,
        suppress_cursor: bool = False,
    ) -> None:
        if not landmarks or len(landmarks) < 21:
            self._fist_started = None
            self._lock_latched = False
            self._fist_stable = 0
            self._filter_x.reset()
            self._filter_y.reset()
            return

        # Preserve the legacy lightweight constructor for callers/tests that
        # instantiate LeftHandProcessor with only an actuator.
        if self._desktop is None or self._trackpad is None:
            return

        now = time.perf_counter()
        fist = is_fist(landmarks)

        if fist:
            self._fist_stable += 1
            if self._fist_started is None:
                self._fist_started = now

            stable_ready = self._fist_stable >= max(2, min(GESTURE_STABILITY_FRAMES, 4))
            if (
                stable_ready
                and not self._lock_latched
                and now - self._fist_started >= LOCK_HOLD_SECONDS
            ):
                if self._actuator.is_dragging:
                    self._actuator.drag_end()
                self._actuator.toggle_cursor_lock()
                self._lock_latched = True
                self._filter_x.reset()
                self._filter_y.reset()
                logger.info(
                    "Cursor lock %s by left-fist gesture",
                    "enabled" if self._actuator.cursor_locked else "disabled",
                )
            return

        # Release arms the same gesture again. The next held left fist toggles
        # lock, making unlock as simple as repeating the lock gesture.
        self._fist_started = None
        self._lock_latched = False
        self._fist_stable = 0

        if suppress_cursor or self._actuator.cursor_locked:
            self._filter_x.reset()
            self._filter_y.reset()
            return

        now = time.perf_counter()
        filtered_x = self._filter_x(landmarks[8].x, now)
        filtered_y = self._filter_y(landmarks[8].y, now)
        screen_x, screen_y = map_to_desktop(
            filtered_x,
            filtered_y,
            self._trackpad,
            self._desktop,
        )
        self._actuator.move(screen_x, screen_y)

    def reset(self) -> None:
        self._fist_started = None
        self._lock_latched = False
        self._fist_stable = 0
        self._filter_x.reset()
        self._filter_y.reset()
