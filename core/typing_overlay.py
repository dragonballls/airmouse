"""Compact contextual keyboard overlay anchored to the Windows cursor."""

from __future__ import annotations

import ctypes
import logging
from typing import Any

import cv2
import numpy as np

from config import (
    TYPING_KEYBOARD_BOTTOM_MARGIN,
    TYPING_KEYBOARD_HEIGHT,
    TYPING_KEYBOARD_MARGIN,
    TYPING_KEYBOARD_WIDTH,
)
from core.display import VirtualDesktop
from core.virtual_keyboard import VirtualKeyboard

logger = logging.getLogger(__name__)

_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TOPMOST = 0x00000008
_GWL_EXSTYLE = -20
_HWND_TOPMOST = -1
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_WM_NCHITTEST = 0x0084
_HTTRANSPARENT = -1


class TypingKeyboardOverlay:
    """A small non-activating keyboard controlled by the right hand."""

    def __init__(self) -> None:
        self.width = TYPING_KEYBOARD_WIDTH
        self.height = TYPING_KEYBOARD_HEIGHT
        self.window_name = "AirMouse Typing Keyboard"
        self.keyboard = VirtualKeyboard(
            self.width,
            self.height,
            key_width=54,
            key_height=42,
            key_margin=5,
            compact=True,
        )
        self.visible = False
        self._anchor: tuple[int, int] | None = None
        self._desktop: VirtualDesktop | None = None
        self._hwnd: int | None = None

    def _native_setup(self) -> None:
        if self._hwnd is not None or not hasattr(ctypes, "windll"):
            return
        hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
        if not hwnd:
            return

        get_style = ctypes.windll.user32.GetWindowLongPtrW
        set_style = ctypes.windll.user32.SetWindowLongPtrW
        style = int(get_style(hwnd, _GWL_EXSTYLE))
        style |= _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW | _WS_EX_TOPMOST
        set_style(hwnd, _GWL_EXSTYLE, style)

        ctypes.windll.user32.SetWindowPos(
            hwnd,
            _HWND_TOPMOST,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
        )
        self._hwnd = int(hwnd)

    def _place(self) -> None:
        if not self.visible or self._anchor is None or self._desktop is None:
            return

        ax, ay = self._anchor
        d = self._desktop
        x = ax + TYPING_KEYBOARD_MARGIN
        y = ay + TYPING_KEYBOARD_MARGIN

        right = d.origin_x + d.total_width
        bottom = d.origin_y + d.total_height

        if x + self.width > right:
            x = ax - self.width - TYPING_KEYBOARD_MARGIN
        if y + self.height > bottom:
            y = ay - self.height - TYPING_KEYBOARD_MARGIN

        x = max(d.origin_x + 2, min(x, right - self.width - 2))
        y = max(d.origin_y + 2, min(y, bottom - self.height - 2))

        cv2.moveWindow(self.window_name, int(x), int(y))

    def show(self, anchor: tuple[int, int], desktop: VirtualDesktop) -> None:
        self._anchor = anchor
        self._desktop = desktop
        self.visible = True
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
        self._native_setup()
        self._place()

    def hide(self) -> None:
        self.visible = False
        self.keyboard.close()
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self._hwnd = None
        self._anchor = None
        self._desktop = None

    def _pointer_position(self, hand: Any) -> tuple[int, int] | None:
        if not hand or len(hand) < 21:
            return None
        return (
            int((1.0 - hand[8].x) * self.width),
            int(hand[8].y * self.height),
        )

    def update(
        self,
        right_hand: Any,
        ai_allowed: bool = True,
    ) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2.putText(
            canvas,
            "TYPE MODE",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 225, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Right index = typing pointer  |  thumb+index pinch = select",
            (14, self.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (175, 185, 200),
            1,
            cv2.LINE_AA,
        )

        pointer = self._pointer_position(right_hand)
        if pointer is not None:
            x, y = pointer
            self.keyboard.update_hover(x, y)

            thumb = (
                int((1.0 - right_hand[4].x) * self.width),
                int(right_hand[4].y * self.height),
            )
            self.keyboard.handle_pinch_type(
                thumb,
                pointer,
                ai_allowed=ai_allowed,
            )
            self.keyboard.update_gesture(pointer)
            self.keyboard.draw_pinch_feedback(canvas, thumb, pointer)
            self.keyboard.draw(canvas, finger_pos=pointer)
        else:
            self.keyboard.update_hover(-1, -1)
            self.keyboard.update_gesture(None)
            self.keyboard.draw(canvas, finger_pos=None)

        if self._anchor is not None:
            self._place()

        return canvas
