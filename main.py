# main.py
"""AI Air Mouse entry point with safe two-hand control and optional AI vision."""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import time
import traceback
from typing import Optional

import cv2
import numpy as np
import psutil

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEBUG_GESTURES,
    GESTURE_STABILITY_FRAMES,
    KEYBOARD_TOGGLE_COOLDOWN_S,
    KEYBOARD_TOGGLE_HOLD_S,
    PROCESS_PRIORITY,
    AI_GESTURE_MAX_AGE_S,
    AI_GESTURE_MIN_CONFIDENCE,
    AI_GESTURE_GRACE_S,
    THUMB_INDEX_CLICK_DIST,
    SHOW_CAMERA_UI,
    TIMER_RESOLUTION_MS,
)
from core.ai_assist import AIAssistant
from core.actuator import MouseActuator
from core.camera import AsyncCamera
from core.display import build_trackpad_zone, build_virtual_desktop
from core.gestures import GestureOrchestrator
from core.gesture_ai import SemanticGestureAI
from core.gestures.utils import (
    is_three_finger_keyboard_pose,
    is_peace_sign,
    normalized_distance,
)
from core.tracker import HandTracker
from core.virtual_keyboard import VirtualKeyboard

if DEBUG_GESTURES:
    from core.debug_overlay import draw_debug_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airmouse.main")


def _apply_windows_performance() -> dict:
    original: dict = {}

    try:
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(TIMER_RESOLUTION_MS)
        original["timer_period"] = TIMER_RESOLUTION_MS
        logger.info("Windows timer resolution set to %dms", TIMER_RESOLUTION_MS)
    except Exception as exc:
        logger.warning("Failed to set timer resolution: %s", exc)

    try:
        proc = psutil.Process(os.getpid())
        proc.nice(PROCESS_PRIORITY)
        logger.info("Process priority set to HIGH_PRIORITY_CLASS (0x%X)", PROCESS_PRIORITY)
    except psutil.AccessDenied:
        logger.warning("Could not elevate process priority")

    try:
        avrt = ctypes.windll.avrt
        task_index = ctypes.c_ulong(0)
        avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
        handle = avrt.AvSetMmThreadCharacteristicsW("Games", ctypes.byref(task_index))
        if handle:
            original["mmcss_handle"] = handle
            logger.info("MMCSS thread registered as 'Games'")
    except Exception as exc:
        logger.warning("MMCSS registration failed: %s", exc)

    params = (ctypes.c_int * 3)(0, 0, 0)
    if ctypes.windll.user32.SystemParametersInfoW(3, 0, params, 0):
        original["mouse_params"] = list(params)
        no_accel = (ctypes.c_int * 3)(0, 0, 0)
        if ctypes.windll.user32.SystemParametersInfoW(4, 0, no_accel, 2):
            logger.info("Windows pointer acceleration disabled")

    return original


def _restore_windows_settings(original: dict) -> None:
    if "timer_period" in original:
        ctypes.windll.winmm.timeEndPeriod(original["timer_period"])
        logger.info("Windows timer resolution restored")

    if "mmcss_handle" in original:
        try:
            ctypes.windll.avrt.AvRevertMmThreadCharacteristics(original["mmcss_handle"])
            logger.info("MMCSS thread unregistered")
        except Exception as exc:
            logger.warning("MMCSS revert failed: %s", exc)

    if "mouse_params" in original:
        restored = (ctypes.c_int * 3)(*original["mouse_params"])
        ctypes.windll.user32.SystemParametersInfoW(4, 0, restored, 2)
        logger.info("Windows pointer acceleration restored")


def _finger_up(lms, tip: int, pip: int) -> bool:
    return lms[tip].y < lms[pip].y


def _keyboard_toggle_pose(lms: Optional[list]) -> bool:
    """Three raised fingers on the left hand; thumb orientation is ignored."""
    return bool(lms) and len(lms) >= 21 and is_three_finger_keyboard_pose(lms)


class KeyboardToggle:
    """Require a stable pose plus release before another toggle can occur."""

    def __init__(
        self,
        hold_seconds: float = KEYBOARD_TOGGLE_HOLD_S,
        cooldown_seconds: float = KEYBOARD_TOGGLE_COOLDOWN_S,
    ):
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.started: float | None = None
        self.last_toggle = -999.0
        self.armed = True
        self.stable_frames = 0

    def update(self, lms: Optional[list], authorized: bool = False) -> bool:
        now = time.perf_counter()
        pose = _keyboard_toggle_pose(lms)

        if not pose:
            self.started = None
            self.stable_frames = 0
            self.armed = True
            return False

        self.stable_frames += 1
        if not self.armed or self.stable_frames < GESTURE_STABILITY_FRAMES:
            return False

        if self.started is None:
            self.started = now

        if (
            authorized
            and now - self.started >= self.hold_seconds
            and now - self.last_toggle >= self.cooldown_seconds
        ):
            self.last_toggle = now
            self.started = None
            self.stable_frames = 0
            self.armed = False
            return True

        return False

    def progress(self) -> float:
        if self.started is None:
            return 0.0
        return max(0.0, min(1.0, (time.perf_counter() - self.started) / self.hold_seconds))


def _gesture_candidate(
    hands,
    keyboard_visible: bool,
    wrist_dy: float,
    two_hand_delta: float,
) -> tuple[str | None, str]:
    """Return a semantic AI candidate and a non-authoritative local hint."""
    left = hands.left
    right = hands.right
    left_keyboard = _keyboard_toggle_pose(left)

    if left_keyboard:
        return "keyboard_toggle", "MediaPipe reports the left hand in the three-finger keyboard pose."

    if keyboard_visible:
        if right and len(right) >= 21:
            index_pinch = normalized_distance(right, 4, 8) <= THUMB_INDEX_CLICK_DIST
            if index_pinch:
                return "keyboard_type", "Right-hand thumb-index pinch is present over the virtual keyboard."
        return None, "No deliberate keyboard gesture candidate."

    if left and right and len(left) >= 21 and len(right) >= 21:
        left_pinch = normalized_distance(left, 4, 8) <= THUMB_INDEX_CLICK_DIST
        right_pinch = normalized_distance(right, 4, 8) <= THUMB_INDEX_CLICK_DIST
        if left_pinch and right_pinch:
            direction = (
                "apart" if two_hand_delta > 0
                else "together" if two_hand_delta < 0
                else "stationary"
            )
            return (
                "two_hand_pinch",
                f"Both hands are holding thumb-index pinches; wrist separation is moving {direction} "
                f"(delta={two_hand_delta:.4f}).",
            )

    if right and len(right) >= 21:
        index_pinch = normalized_distance(right, 4, 8) <= THUMB_INDEX_CLICK_DIST
        middle_pinch = normalized_distance(right, 4, 12) <= THUMB_INDEX_CLICK_DIST
        if middle_pinch and not index_pinch:
            return "middle_pinch", "Right thumb-middle pinch is present."
        if index_pinch:
            return "index_pinch", "Right thumb-index pinch is present; hold duration is supplied separately."
        if is_peace_sign(right):
            direction = "upward" if wrist_dy < 0 else "downward" if wrist_dy > 0 else "stationary"
            return "scroll_sign", f"Right hand shows the peace/scroll sign with local wrist movement {direction}."

    return None, "No deliberate gesture candidate."

def _ai_frame_crop(frame, hands):
    """Crop the AI input around visible hands while retaining generous context."""
    if frame is None or not getattr(frame, "size", 0):
        return frame
    landmarks = []
    for hand in (getattr(hands, "left", None), getattr(hands, "right", None)):
        if hand and len(hand) >= 21:
            landmarks.extend(hand)

    if not landmarks:
        return frame

    height, width = frame.shape[:2]
    xs = [max(0.0, min(1.0, lm.x)) for lm in landmarks]
    ys = [max(0.0, min(1.0, lm.y)) for lm in landmarks]
    x0 = int(min(xs) * width)
    x1 = int(max(xs) * width)
    y0 = int(min(ys) * height)
    y1 = int(max(ys) * height)

    pad_x = max(40, int((x1 - x0) * 0.45))
    pad_y = max(40, int((y1 - y0) * 0.45))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else frame

def _mirror(landmark, width: int, height: int) -> tuple[int, int]:
    return int((1.0 - landmark.x) * width), int(landmark.y * height)


def _draw_status(
    frame,
    mode: str,
    fps: float,
    keyboard_toggle: KeyboardToggle,
    ai: AIAssistant,
    ai_text: str,
    ai_gesture: str,
    ai_confidence: float,
    mouse_locked: bool = False,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (6, 6), (w - 6, 86), (18, 18, 22), -1)
    title = "VIRTUAL KEYBOARD" if mode == "keyboard" else "AIR MOUSE"
    cv2.putText(frame, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 225, 255), 2, cv2.LINE_AA)

    if mode == "keyboard":
        hint = "Right hand: hover + thumb/index pinch to type | Left 3-finger hold = toggle | K = toggle | Q = quit"
    else:
        hint = "Left index = cursor | Right pinch = click | thumb+middle = right-click | pinch-hold = drag | Left fist hold = lock/unlock"
    if mouse_locked and mode == "mouse":
        hint = "MOUSE LOCKED | Left fist hold again to unlock"
    cv2.putText(frame, hint, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (195, 200, 208), 1, cv2.LINE_AA)

    if keyboard_toggle.started is not None:
        cv2.putText(
            frame,
            f"Keyboard arming: {keyboard_toggle.progress() * 100:.0f}%",
            (18, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"FPS {fps:.0f} | {ai.status}",
            (w - 390, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (195, 200, 208),
            1,
            cv2.LINE_AA,
        )
        if ai_gesture:
            cv2.putText(
                frame,
                f"AI gesture: {ai_gesture} ({ai_confidence:.0%})",
                (18, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    if ai_text:
        panel_top = 96
        cv2.rectangle(frame, (8, panel_top), (w - 8, panel_top + 86), (18, 18, 22), -1)
        cv2.putText(frame, "AI DIAGNOSTIC", (18, panel_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 225, 255), 1, cv2.LINE_AA)
        remaining = ai_text.replace("\n", " ")
        for idx in range(2):
            chunk = remaining[idx * 92:(idx + 1) * 92]
            if not chunk:
                break
            cv2.putText(frame, chunk, (18, panel_top + 45 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)


def run() -> None:
    original_settings: dict = {}
    keyboard = VirtualKeyboard(CAMERA_WIDTH, CAMERA_HEIGHT)
    toggle = KeyboardToggle(hold_seconds=min(KEYBOARD_TOGGLE_HOLD_S, 0.65))
    ai = AIAssistant()
    semantic_ai = SemanticGestureAI(ai, min_confidence=AI_GESTURE_MIN_CONFIDENCE)
    ai_text = ""
    ai_gesture = ""
    ai_confidence = 0.0
    previous_right_wrist_y: float | None = None
    previous_two_hand_distance: float | None = None
    pinch_candidate_started: float | None = None
    previous_candidate: str | None = None

    def emergency_restore():
        semantic_ai.close()
        keyboard.close()
        if original_settings:
            _restore_windows_settings(original_settings)

    atexit.register(emergency_restore)

    logger.info("=== Safe Unified Airmouse + Virtual Keyboard starting ===")
    original_settings.update(_apply_windows_performance())

    desktop = build_virtual_desktop()
    trackpad = build_trackpad_zone()
    actuator = MouseActuator(
        desktop.total_width,
        desktop.total_height,
        origin_x=desktop.origin_x,
        origin_y=desktop.origin_y,
    )
    processor = GestureOrchestrator(
        actuator,
        desktop,
        trackpad,
        ai_required=semantic_ai.enabled,
    )

    fps_clock = time.perf_counter()
    fps_frames = 0
    fps = 0.0
    window = "Unified Air Control"
    keyboard_window = "AirMouse Keyboard"
    keyboard_window_created = False

    def _draw_hand_overlay(canvas, landmarks):
        """Draw the AirNav-style dots/lines without exposing the live camera."""
        if not landmarks or len(landmarks) < 21:
            return
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        )
        h, w = canvas.shape[:2]
        points = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in landmarks]
        for a, b in connections:
            cv2.line(canvas, points[a], points[b], (120, 180, 255), 2, cv2.LINE_AA)
        for px, py in points:
            cv2.circle(canvas, (px, py), 4, (80, 220, 255), -1, cv2.LINE_AA)

    try:
        with AsyncCamera() as camera, HandTracker() as tracker:
            if SHOW_CAMERA_UI:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, CAMERA_WIDTH, CAMERA_HEIGHT)
                keyboard_window_created = False

            while True:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.002)
                    continue

                hands = tracker.process(frame)

                now = time.perf_counter()
                wrist_dy = 0.0
                two_hand_delta = 0.0
                if hands.right and len(hands.right) >= 21 and previous_right_wrist_y is not None:
                    wrist_dy = hands.right[0].y - previous_right_wrist_y
                if (
                    hands.left and hands.right
                    and len(hands.left) >= 21
                    and len(hands.right) >= 21
                    and previous_two_hand_distance is not None
                ):
                    current_two_hand_distance = abs(
                        hands.right[0].x - hands.left[0].x
                    )
                    two_hand_delta = current_two_hand_distance - previous_two_hand_distance
                if hands.right and len(hands.right) >= 21:
                    previous_right_wrist_y = hands.right[0].y
                else:
                    previous_right_wrist_y = None

                if (
                    hands.left and hands.right
                    and len(hands.left) >= 21
                    and len(hands.right) >= 21
                ):
                    previous_two_hand_distance = abs(
                        hands.right[0].x - hands.left[0].x
                    )
                else:
                    previous_two_hand_distance = None

                candidate, hands_hint = _gesture_candidate(
                    hands,
                    keyboard.visible,
                    wrist_dy,
                    two_hand_delta,
                )

                if candidate != previous_candidate:
                    pinch_candidate_started = (
                        now if candidate in {"index_pinch", "middle_pinch"} else None
                    )
                    # Keep a completed AI decision alive through the release edge.
                    # This is critical for click gestures whose action occurs on release.
                    if candidate is not None:
                        semantic_ai.clear()
                    previous_candidate = candidate
                elif candidate in {"index_pinch", "middle_pinch"} and pinch_candidate_started is None:
                    pinch_candidate_started = now
                if candidate is None:
                    pinch_candidate_started = None

                if candidate in {"index_pinch", "middle_pinch"} and pinch_candidate_started is not None:
                    held = now - pinch_candidate_started
                    hands_hint += f" Pinch has been held for {held:.2f}s."
                if candidate == "scroll_sign":
                    hands_hint += f" Current frame-to-frame wrist dy={wrist_dy:.4f}."

                semantic_ai.submit(
                    _ai_frame_crop(frame, hands) if semantic_ai.enabled else None,
                    candidate or "",
                    "keyboard" if keyboard.visible else "mouse",
                    hands_hint,
                )
                semantic_ai.poll()

                if candidate is not None:
                    decision = semantic_ai.current(
                        candidate,
                        max_age=AI_GESTURE_MAX_AGE_S,
                        grace=AI_GESTURE_GRACE_S,
                    )
                else:
                    decision = semantic_ai.current(
                        None,
                        max_age=AI_GESTURE_MAX_AGE_S,
                        grace=AI_GESTURE_GRACE_S,
                    )
                if decision is not None:
                    ai_gesture = decision.gesture
                    ai_confidence = decision.confidence
                else:
                    ai_gesture = ""
                    ai_confidence = 0.0

                def ai_authorizes(expected: str, target: str) -> bool:
                    if not semantic_ai.enabled:
                        return True
                    return (
                        decision is not None
                        and decision.gesture == expected
                        and decision.target_hand in {target, "both"}
                    )

                keyboard_toggle_authorized = ai_authorizes("keyboard_toggle", "left")

                authorized_mouse_gesture = ai_gesture or None
                if semantic_ai.enabled and decision is not None:
                    gesture_target = {
                        "left_click": "right",
                        "right_click": "right",
                        "drag": "right",
                        "scroll_up": "right",
                        "scroll_down": "right",
                        "zoom_in": "both",
                        "zoom_out": "both",
                    }.get(decision.gesture)
                    if gesture_target is not None:
                        target_ok = (
                            decision.target_hand == gesture_target
                            or (
                                gesture_target == "right"
                                and decision.target_hand == "both"
                            )
                        )
                        if not target_ok:
                            authorized_mouse_gesture = None

                if toggle.update(hands.left, authorized=keyboard_toggle_authorized):
                    keyboard.toggle()
                    processor.reset()
                    semantic_ai.clear()
                    previous_candidate = None
                    logger.info(
                        "Virtual keyboard %s%s",
                        "enabled" if keyboard.visible else "disabled",
                        " (AI authorized)" if semantic_ai.enabled else "",
                    )

                if keyboard.visible:
                    active = hands.right
                    if not keyboard_window_created:
                        cv2.namedWindow(keyboard_window, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(keyboard_window, CAMERA_WIDTH, CAMERA_HEIGHT)
                        keyboard_window_created = True

                    if SHOW_CAMERA_UI:
                        display = cv2.flip(frame, 1)
                    else:
                        display = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                        cv2.putText(
                            display,
                            "CAMERA PREVIEW OFF",
                            (18, 112),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (150, 160, 175),
                            1,
                            cv2.LINE_AA,
                        )
                    _draw_hand_overlay(display, hands.left)
                    _draw_hand_overlay(display, hands.right)
                    if active and len(active) >= 21:
                        ix, iy = _mirror(active[8], frame.shape[1], frame.shape[0])
                        tx, ty = _mirror(active[4], frame.shape[1], frame.shape[0])
                        keyboard.update_hover(ix, iy)
                        keyboard.handle_pinch_type(
                            (tx, ty),
                            (ix, iy),
                            ai_allowed=ai_authorizes("keyboard_type", "right"),
                        )
                        keyboard.update_gesture((ix, iy))
                        keyboard.draw(display, finger_pos=(ix, iy))
                    else:
                        keyboard.update_hover(-1, -1)
                        keyboard.update_gesture(None)
                        keyboard.draw(display, finger_pos=None)
                    frame = display
                else:
                    processor.process(
                        hands,
                        ai_gesture=authorized_mouse_gesture,
                    )
                    frame = cv2.flip(frame, 1)

                fps_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_clock
                if elapsed >= 1.0:
                    fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_clock = now

                _draw_status(
                    frame,
                    "keyboard" if keyboard.visible else "mouse",
                    fps,
                    toggle,
                    ai,
                    ai_text,
                    ai_gesture,
                    ai_confidence,
                    mouse_locked=processor.mouse_locked,
                )

                if DEBUG_GESTURES:
                    cv2.imshow("AirMouse Debug", draw_debug_frame(frame, hands, processor.right_state))

                if keyboard.visible:
                    cv2.imshow(keyboard_window, frame)
                    key = cv2.waitKey(1) & 0xFF
                elif SHOW_CAMERA_UI:
                    if keyboard_window_created:
                        cv2.destroyWindow(keyboard_window)
                        keyboard_window_created = False
                    cv2.imshow(window, frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("k"), ord("K")):
                        keyboard.toggle()
                        processor.reset()
                        logger.info("Virtual keyboard %s (manual)", "enabled" if keyboard.visible else "disabled")
                    if key in (ord("a"), ord("A")):
                        ai_text = ai.analyze(frame, "keyboard" if keyboard.visible else "mouse")
                        logger.info("AI diagnostic requested")
    except KeyboardInterrupt:
        logger.info("Ctrl+C received — shutting down cleanly")
    except Exception:
        logger.error("Unhandled exception:\n%s", traceback.format_exc())
        raise
    finally:
        if actuator.is_dragging:
            actuator.drag_end()
        keyboard.close()
        if keyboard_window_created:
            cv2.destroyWindow(keyboard_window)
        if SHOW_CAMERA_UI or DEBUG_GESTURES:
            cv2.destroyAllWindows()
        _restore_windows_settings(original_settings)
        original_settings.clear()
        logger.info("=== Safe Unified Airmouse + Virtual Keyboard stopped ===")


if __name__ == "__main__":
    run()
