# core/gestures/right_hand.py
"""Conservative right-hand mouse processor with optional AI semantic gating."""

from __future__ import annotations

import logging
import time
from enum import Enum, auto

from config import (
    CAMERA_FPS,
    DOUBLE_CLICK_WINDOW_S,
    DRAG_HOLD_SECONDS,
    GESTURE_COOLDOWN_SECONDS,
    GESTURE_RELEASE_FRAMES,
    GESTURE_STABILITY_FRAMES,
    ONE_EURO_BETA,
    ONE_EURO_DCUTOFF,
    ONE_EURO_MINCUTOFF,
    PINCH_RELEASE_DIST,
    SCROLL_COOLDOWN_S,
    SCROLL_TICK_SCALE,
    THUMB_INDEX_CLICK_DIST,
    THUMB_MIDDLE_CLICK_DIST,
    WRIST_VELOCITY_THRESHOLD,
)
from core.actuator import MouseActuator
from core.display import TrackpadZone, VirtualDesktop, map_to_desktop
from core.filter import OneEuroFilter
from core.gestures.utils import (
    is_fist,
    is_peace_sign,
    normalized_distance,
)
from core.tracker import Landmark

logger = logging.getLogger(__name__)


class _State(Enum):
    IDLE = auto()
    LEFT_PINCH_PENDING_DRAG = auto()
    RIGHT_PINCH = auto()
    DRAGGING = auto()
    SCROLLING = auto()
    LOCKED = auto()


class _StableGate:
    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.count = 0

    def update(self, active: bool) -> bool:
        self.count = self.count + 1 if active else 0
        return self.count >= self.frames

    def reset(self) -> None:
        self.count = 0


class RightHandProcessor:
    def __init__(self, actuator: MouseActuator, desktop: VirtualDesktop, trackpad: TrackpadZone) -> None:
        self._actuator = actuator
        self._desktop = desktop
        self._trackpad = trackpad
        self._state = _State.IDLE

        self._pinch_start_time: float | None = None
        self._right_pinch_start_time: float | None = None
        self._last_click_time = 0.0
        self._last_action_time = 0.0
        self._last_scroll_time = 0.0
        self._prev_wrist_y: float | None = None
        self._last_frame_time: float | None = None

        self._left_pinch_gate = _StableGate(GESTURE_STABILITY_FRAMES)
        self._right_pinch_gate = _StableGate(GESTURE_STABILITY_FRAMES)
        self._left_release_gate = _StableGate(GESTURE_RELEASE_FRAMES)
        self._right_release_gate = _StableGate(GESTURE_RELEASE_FRAMES)
        self._peace_gate = _StableGate(GESTURE_STABILITY_FRAMES)

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

    def _reset_gates(self) -> None:
        self._left_pinch_gate.reset()
        self._right_pinch_gate.reset()
        self._left_release_gate.reset()
        self._right_release_gate.reset()
        self._peace_gate.reset()

    @staticmethod
    def _ai_allows(ai_gesture: str | None, required: bool, expected: str) -> bool:
        return not required or ai_gesture == expected

    def process(
        self,
        landmarks: list[Landmark] | None,
        suppress_actions: bool = False,
        ai_gesture: str | None = None,
        ai_required: bool = False,
    ) -> None:
        if not landmarks or len(landmarks) < 21:
            if self._state == _State.DRAGGING:
                self._actuator.drag_end()
            self._state = _State.IDLE
            self._pinch_start_time = None
            self._right_pinch_start_time = None
            self._prev_wrist_y = None
            self._last_frame_time = None
            self._filter_x.reset()
            self._filter_y.reset()
            self._reset_gates()
            return

        now = time.perf_counter()
        dt = max(
            now - self._last_frame_time if self._last_frame_time is not None else 1.0 / CAMERA_FPS,
            1e-3,
        )

        wrist_vel_y = 0.0
        if self._prev_wrist_y is not None:
            wrist_vel_y = (landmarks[0].y - self._prev_wrist_y) / dt
        self._prev_wrist_y = landmarks[0].y
        self._last_frame_time = now

        if suppress_actions:
            if self._state == _State.DRAGGING:
                self._actuator.drag_end()
            self._state = _State.IDLE
            self._pinch_start_time = None
            self._right_pinch_start_time = None
            self._filter_x.reset()
            self._filter_y.reset()
            self._reset_gates()
            return

        if is_fist(landmarks):
            if self._state == _State.DRAGGING:
                self._actuator.drag_end()
            self._state = _State.LOCKED
            self._pinch_start_time = None
            self._right_pinch_start_time = None
            self._reset_gates()
            return

        if self._state == _State.LOCKED:
            self._state = _State.IDLE
            if is_fist(landmarks):
                return

        index_pinch = normalized_distance(landmarks, 4, 8) <= THUMB_INDEX_CLICK_DIST
        middle_pinch = normalized_distance(landmarks, 4, 12) <= THUMB_MIDDLE_CLICK_DIST
        index_released = normalized_distance(landmarks, 4, 8) >= PINCH_RELEASE_DIST
        middle_released = normalized_distance(landmarks, 4, 12) >= PINCH_RELEASE_DIST
        peace = is_peace_sign(landmarks)

        index_stable = self._left_pinch_gate.update(index_pinch)
        middle_stable = self._right_pinch_gate.update(middle_pinch)

        if self._state == _State.IDLE and middle_stable and not index_pinch:
            self._state = _State.RIGHT_PINCH
            self._right_pinch_start_time = now

        if self._state == _State.RIGHT_PINCH:
            if self._right_release_gate.update(middle_released):
                if self._ai_allows(ai_gesture, ai_required, "right_click") and now - self._last_action_time >= GESTURE_COOLDOWN_SECONDS:
                    self._actuator.right_click()
                    self._last_action_time = now
                self._state = _State.IDLE
                self._right_pinch_start_time = None
                self._right_release_gate.reset()

        elif self._state == _State.IDLE and index_stable:
            self._state = _State.LEFT_PINCH_PENDING_DRAG
            self._pinch_start_time = now
            self._left_release_gate.reset()

        elif self._state == _State.LEFT_PINCH_PENDING_DRAG:
            if self._left_release_gate.update(index_released):
                if (
                    self._pinch_start_time is not None
                    and now - self._pinch_start_time < DRAG_HOLD_SECONDS
                    and now - self._last_action_time >= GESTURE_COOLDOWN_SECONDS
                ):
                    if self._ai_allows(ai_gesture, ai_required, "left_click"):
                        if now - self._last_click_time <= DOUBLE_CLICK_WINDOW_S and self._last_click_time > 0:
                            self._actuator.double_click()
                            self._last_click_time = 0.0
                        else:
                            self._actuator.left_click()
                            self._last_click_time = now
                        self._last_action_time = now
                self._state = _State.IDLE
                self._pinch_start_time = None
                self._left_release_gate.reset()

            elif (
                self._pinch_start_time is not None
                and now - self._pinch_start_time >= DRAG_HOLD_SECONDS
            ):
                if self._ai_allows(ai_gesture, ai_required, "drag"):
                    self._actuator.drag_start()
                    self._state = _State.DRAGGING

        elif self._state == _State.DRAGGING and self._left_release_gate.update(index_released):
            self._actuator.drag_end()
            self._state = _State.IDLE
            self._pinch_start_time = None
            self._left_release_gate.reset()

        if self._state == _State.IDLE and self._peace_gate.update(peace):
            if abs(wrist_vel_y) >= WRIST_VELOCITY_THRESHOLD and now - self._last_scroll_time >= SCROLL_COOLDOWN_S:
                expected = "scroll_up" if wrist_vel_y < 0 else "scroll_down"
                if self._ai_allows(ai_gesture, ai_required, expected):
                    ticks = int(min(5, max(1, abs(wrist_vel_y) / SCROLL_TICK_SCALE)))
                    self._actuator.scroll(ticks if wrist_vel_y < 0 else -ticks)
                    self._last_scroll_time = now

        if self._state in (_State.IDLE, _State.DRAGGING):
            filtered_x = self._filter_x(landmarks[8].x, now)
            filtered_y = self._filter_y(landmarks[8].y, now)
            screen_x, screen_y = map_to_desktop(
                filtered_x, filtered_y, self._trackpad, self._desktop
            )
            self._actuator.move(screen_x, screen_y)
