# core/actuator.py
"""
Windows mouse actuation via SendInput.

Movement uses SendInput directly for low overhead. Click/drag/keyboard
operations use pynput for reliable button/key state handling.
"""

import ctypes
import logging

from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Key

logger = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_u", _INPUT_UNION),
    ]


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000


def _send_input(inp: INPUT) -> None:
    result = _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if result == 0:
        err = ctypes.get_last_error()
        logger.warning("SendInput returned 0 (error=%d)", err)


class MouseActuator:
    """Centralized mouse/keyboard output with a persistent cursor-lock mode."""

    def __init__(
        self,
        total_width: int,
        total_height: int,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> None:
        self._total_w = total_width
        self._total_h = total_height
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._pynput = MouseController()
        self._dragging = False
        self._keyboard = KeyboardController()
        self._cursor_locked = False
        self._locked_position: tuple[int, int] | None = None
        logger.info("MouseActuator ready (%dx%d desktop)", total_width, total_height)

    def move(self, x: int, y: int) -> None:
        """Move to absolute desktop coordinates unless cursor lock is active."""
        if self._cursor_locked:
            return

        norm_x = int((x - self._origin_x) * 65535 / max(self._total_w - 1, 1))
        norm_y = int((y - self._origin_y) * 65535 / max(self._total_h - 1, 1))
        norm_x = max(0, min(65535, norm_x))
        norm_y = max(0, min(65535, norm_y))

        inp = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=norm_x,
                dy=norm_y,
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                time=0,
                dwExtraInfo=0,
            ),
        )
        _send_input(inp)

    def _get_cursor_position(self) -> tuple[int, int] | None:
        point = POINT()
        if _user32.GetCursorPos(ctypes.byref(point)):
            return int(point.x), int(point.y)
        return None

    def lock_cursor(self) -> None:
        """Freeze virtual cursor movement at its current Windows position."""
        if self._cursor_locked:
            return
        self._locked_position = self._get_cursor_position()
        self._cursor_locked = True
        logger.info("Cursor locked at %s", self._locked_position)

    def unlock_cursor(self) -> None:
        """Release the cursor lock without moving the cursor."""
        if not self._cursor_locked:
            return
        self._cursor_locked = False
        logger.info("Cursor unlocked")

    def toggle_cursor_lock(self) -> bool:
        """Toggle cursor lock and return the new lock state."""
        if self._cursor_locked:
            self.unlock_cursor()
        else:
            self.lock_cursor()
        return self._cursor_locked

    @property
    def cursor_locked(self) -> bool:
        return self._cursor_locked

    @property
    def locked_position(self) -> tuple[int, int] | None:
        return self._locked_position

    def left_click(self) -> None:
        self._pynput.click(Button.left)

    def right_click(self) -> None:
        self._pynput.click(Button.right)

    def drag_start(self) -> None:
        if not self._dragging:
            self._pynput.press(Button.left)
            self._dragging = True
            logger.debug("Drag started")

    def drag_end(self) -> None:
        if self._dragging:
            self._pynput.release(Button.left)
            self._dragging = False
            logger.debug("Drag ended")

    def scroll(self, ticks: int) -> None:
        self._pynput.scroll(0, ticks)

    def double_click(self) -> None:
        self._pynput.click(Button.left, 2)

    def win_d(self) -> None:
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press("d")
            self._keyboard.release("d")

    def alt_tab(self) -> None:
        with self._keyboard.pressed(Key.alt):
            self._keyboard.press(Key.tab)
            self._keyboard.release(Key.tab)

    def win_tab(self) -> None:
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press(Key.tab)
            self._keyboard.release(Key.tab)

    def win_key(self) -> None:
        self._keyboard.press(Key.cmd)
        self._keyboard.release(Key.cmd)

    def alt_left(self) -> None:
        with self._keyboard.pressed(Key.alt):
            self._keyboard.press(Key.left)
            self._keyboard.release(Key.left)

    def win_l(self) -> None:
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press("l")
            self._keyboard.release("l")

    def win_snap_left(self) -> None:
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press(Key.left)
            self._keyboard.release(Key.left)

    def win_snap_right(self) -> None:
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press(Key.right)
            self._keyboard.release(Key.right)

    def ctrl_down(self) -> None:
        self._keyboard.press(Key.ctrl)

    def ctrl_up(self) -> None:
        self._keyboard.release(Key.ctrl)

    def zoom_in(self) -> None:
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("=")
            self._keyboard.release("=")

    def zoom_out(self) -> None:
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("-")
            self._keyboard.release("-")

    @property
    def is_dragging(self) -> bool:
        return self._dragging
