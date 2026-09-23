# main.py
"""
AI Air Mouse — Entry point.

Startup sequence (ORDER MATTERS):
  1. Configure logging
  2. Set Windows timer resolution to 1ms
  3. Elevate process to HIGH_PRIORITY_CLASS
  4. Register main thread with MMCSS ("Games" profile)
  5. Build virtual desktop + trackpad zone (DPI awareness already set at display.py import)
  6. Disable Windows pointer acceleration (restore on exit)
  7. Start async camera
  8. Start MediaPipe hand tracker
  9. Instantiate gesture processor + actuator
  10. Run inference loop
  11. On exit: restore timer resolution, restore pointer precision, release all resources

Exit handling:
  - Ctrl+C (KeyboardInterrupt) -> clean shutdown
  - Any unhandled exception -> log traceback, then clean shutdown
  - atexit handler as final safety net to ensure pointer precision is restored
"""

import atexit
import ctypes
import logging
import os
import time
import traceback
from typing import Optional

import psutil
import cv2

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEBUG_GESTURES,
    PROCESS_PRIORITY,
    SHOW_CAMERA_UI,
    TIMER_RESOLUTION_MS,
)
from core.camera import AsyncCamera
from core.tracker import HandTracker
from core.gestures import GestureOrchestrator
from core.actuator import MouseActuator
from core.display import build_virtual_desktop, build_trackpad_zone
from core.virtual_keyboard import VirtualKeyboard

if DEBUG_GESTURES:
    from core.debug_overlay import draw_debug_frame

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airmouse.main")


# ── Windows OS Setup ──────────────────────────────────────────────────────────

def _apply_windows_performance() -> dict:
    """
    Apply all Windows OS performance settings.
    Returns a dict of original values so they can be restored on exit.
    """
    original = {}

    # 1. Timer resolution: set to 1ms floor (default is 15.6ms)
    try:
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(TIMER_RESOLUTION_MS)
        original["timer_period"] = TIMER_RESOLUTION_MS
        logger.info("Windows timer resolution set to %dms", TIMER_RESOLUTION_MS)
    except Exception as e:
        logger.warning("Failed to set timer resolution: %s", e)

    # 2. Process priority: HIGH_PRIORITY_CLASS
    try:
        proc = psutil.Process(os.getpid())
        proc.nice(PROCESS_PRIORITY)
        logger.info("Process priority set to HIGH_PRIORITY_CLASS (0x%X)", PROCESS_PRIORITY)
    except psutil.AccessDenied:
        logger.warning("Could not elevate process priority — run as administrator for best performance")

    # 3. MMCSS: register main thread as "Games" workload class
    # This tells the Windows scheduler to give us CPU time more aggressively
    # and deprioritize background tasks (Defender, Windows Update, etc.) relative to us.
    try:
        avrt = ctypes.windll.avrt
        task_index = ctypes.c_ulong(0)
        avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
        handle = avrt.AvSetMmThreadCharacteristicsW("Games", ctypes.byref(task_index))
        if handle:
            original["mmcss_handle"] = handle
            logger.info("MMCSS thread registered as 'Games' (handle=%d)", handle)
        else:
            logger.warning("MMCSS registration returned NULL handle")
    except Exception as e:
        logger.warning("MMCSS registration failed: %s", e)

    # 4. Disable "Enhance Pointer Precision" (Windows mouse acceleration)
    # This ballistic curve applied by Windows corrupts the 1€ Filter output.
    # SPI_GETMOUSE = 3, SPI_SETMOUSE = 4
    # The parameter is [threshold1, threshold2, acceleration]
    # acceleration=0 disables enhance pointer precision
    original_mouse_params = (ctypes.c_int * 3)(0, 0, 0)
    get_result = ctypes.windll.user32.SystemParametersInfoW(3, 0, original_mouse_params, 0)
    if not get_result:
        logger.warning("SystemParametersInfoW(SPI_GETMOUSE) failed — cannot save/restore mouse settings")
    else:
        original["mouse_params"] = list(original_mouse_params)
        no_accel = (ctypes.c_int * 3)(0, 0, 0)
        set_result = ctypes.windll.user32.SystemParametersInfoW(4, 0, no_accel, 2)
        if not set_result:
            logger.warning("SystemParametersInfoW(SPI_SETMOUSE) failed — pointer acceleration may not be disabled")
        else:
            logger.info("Windows pointer acceleration disabled")

    return original


def _restore_windows_settings(original: dict) -> None:
    """Restore all Windows settings mutated during startup."""

    # Restore timer resolution
    if "timer_period" in original:
        ctypes.windll.winmm.timeEndPeriod(original["timer_period"])
        logger.info("Windows timer resolution restored")

    # Unregister MMCSS
    if "mmcss_handle" in original:
        try:
            ctypes.windll.avrt.AvRevertMmThreadCharacteristics(original["mmcss_handle"])
            logger.info("MMCSS thread unregistered")
        except Exception as e:
            logger.warning("MMCSS revert failed: %s", e)

    # Restore pointer precision
    if "mouse_params" in original:
        restored = (ctypes.c_int * 3)(*original["mouse_params"])
        ctypes.windll.user32.SystemParametersInfoW(4, 0, restored, 2)
        logger.info("Windows pointer acceleration restored")


# ── Main Loop ─────────────────────────────────────────────────────────────────


def _finger_up(lms, tip: int, pip: int) -> bool:
    return lms[tip].y < lms[pip].y


def _keyboard_toggle_pose(lms: Optional[list]) -> bool:
    """Left hand: index+middle+ring up; pinky down; thumb curled."""
    if not lms or len(lms) < 21:
        return False
    thumb_near = ((lms[4].x - lms[0].x) ** 2 + (lms[4].y - lms[0].y) ** 2) ** 0.5 < 0.30
    return (
        _finger_up(lms, 8, 6)
        and _finger_up(lms, 12, 10)
        and _finger_up(lms, 16, 14)
        and not _finger_up(lms, 20, 18)
        and thumb_near
    )


class KeyboardToggle:
    def __init__(self, hold_seconds: float = 0.75, cooldown_seconds: float = 1.25):
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.started: float | None = None
        self.last_toggle = -999.0

    def update(self, lms: Optional[list]) -> bool:
        now = time.perf_counter()
        if not _keyboard_toggle_pose(lms):
            self.started = None
            return False
        if self.started is None:
            self.started = now
            return False
        if now - self.started >= self.hold_seconds and now - self.last_toggle >= self.cooldown_seconds:
            self.last_toggle = now
            self.started = None
            return True
        return False


def _mirror(landmark, width: int, height: int) -> tuple[int, int]:
    return int((1.0 - landmark.x) * width), int(landmark.y * height)


def _draw_mouse_ui(frame, hands, fps: float) -> None:
    h, w = frame.shape[:2]
    if hands.right and len(hands.right) >= 9:
        p = _mirror(hands.right[8], w, h)
        cv2.drawMarker(frame, p, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.rectangle(frame, (7, 7), (w - 7, 61), (18, 18, 22), -1)
    cv2.putText(
        frame, "AIR MOUSE", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
        (180, 225, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "Left 3-finger hold = virtual keyboard  |  K = toggle  |  Q = quit",
        (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 200, 208), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{fps:.0f} FPS", (w - 86, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        (215, 220, 225), 1, cv2.LINE_AA
    )


def run(stop_event=None) -> None:
    original_settings = {}
    keyboard = VirtualKeyboard(CAMERA_WIDTH, CAMERA_HEIGHT)
    toggle = KeyboardToggle()

    def emergency_restore():
        keyboard.close()
        if original_settings:
            _restore_windows_settings(original_settings)

    atexit.register(emergency_restore)

    logger.info("=== Unified Airmouse + Virtual Keyboard starting ===")
    original_settings.update(_apply_windows_performance())

    desktop = build_virtual_desktop()
    trackpad = build_trackpad_zone()
    actuator = MouseActuator(desktop.total_width, desktop.total_height)
    processor = GestureOrchestrator(actuator, desktop, trackpad)

    fps_clock = time.perf_counter()
    fps_frames = 0
    fps = 0.0
    window = "Unified Air Control"

    try:
        with AsyncCamera() as camera, HandTracker() as tracker:
            if SHOW_CAMERA_UI:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, CAMERA_WIDTH, CAMERA_HEIGHT)

            while stop_event is None or not stop_event.is_set():
                frame = camera.read()
                if frame is None:
                    time.sleep(0.002)
                    continue

                hands = tracker.process(frame)

                if toggle.update(hands.left):
                    keyboard.toggle()
                    processor.reset()
                    logger.info(
                        "Virtual keyboard %s",
                        "enabled" if keyboard.visible else "disabled",
                    )

                if keyboard.visible:
                    active = hands.right or hands.left
                    display = cv2.flip(frame, 1)
                    if active and len(active) >= 21:
                        ix, iy = _mirror(active[8], frame.shape[1], frame.shape[0])
                        tx, ty = _mirror(active[4], frame.shape[1], frame.shape[0])
                        keyboard.update_hover(ix, iy)
                        keyboard.handle_pinch_type((tx, ty), (ix, iy))
                        keyboard.update_gesture((ix, iy))
                        keyboard.draw(display, finger_pos=(ix, iy))
                    else:
                        keyboard.update_hover(-1, -1)
                        keyboard.update_gesture(None)
                        keyboard.draw(display, finger_pos=None)
                    frame = display
                else:
                    processor.process(hands)
                    frame = cv2.flip(frame, 1)
                    _draw_mouse_ui(frame, hands, fps)

                fps_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_clock
                if elapsed >= 1.0:
                    fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_clock = now

                if DEBUG_GESTURES:
                    cv2.imshow("AirMouse Debug", draw_debug_frame(frame, hands, processor.right_state))

                if SHOW_CAMERA_UI:
                    cv2.imshow(window, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        break
                    if key in (ord("k"), ord("K")):
                        keyboard.toggle()
                        processor.reset()

    except KeyboardInterrupt:
        logger.info("Ctrl+C received — shutting down cleanly")
    except Exception:
        logger.error("Unhandled exception:\n%s", traceback.format_exc())
        raise
    finally:
        if actuator.is_dragging:
            actuator.drag_end()
        keyboard.close()
        if SHOW_CAMERA_UI or DEBUG_GESTURES:
            cv2.destroyAllWindows()
        _restore_windows_settings(original_settings)
        original_settings.clear()
        logger.info("=== Unified Airmouse + Virtual Keyboard stopped ===")


if __name__ == "__main__":
    run()
