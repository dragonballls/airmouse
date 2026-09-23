from __future__ import annotations

"""
Virtual QWERTY keyboard overlay rendered with OpenCV.

Keys are stored as structured button objects and drawn as rectangles
on top of a webcam feed. Index fingertip hover highlights keys in real time.
"""

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pyautogui

from core.keyboard_ui import draw_key_press_ripple, draw_typed_preview
from core.keyboard_utils import pinch_distance

# QWERTY layout: each row is a list of key labels
QWERTY_LAYOUT = [
    list("1234567890-=") + ["BKSP"],
    ["TAB"] + list("QWERTYUIOP") + ["[", "]", "\\"],
    ["CAPS"] + list("ASDFGHJKL") + [";", "'", "ENTER"],
    ["SHIFT"] + list("ZXCVBNM") + [",", ".", "/"] + ["SHIFT"],
    ["CTRL", "ALT", "WIN", "LEFT", "DOWN", "UP", "RIGHT"],
]


@dataclass(frozen=True)
class KeyButton:
    """Represents a single keyboard key with label and rectangular bounds."""

    label: str
    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def contains(self, px: int, py: int, padding: int = 0) -> bool:
        return (
            self.x - padding <= px <= self.x + self.width + padding
            and self.y - padding <= py <= self.y + self.height + padding
        )


class VirtualKeyboard:
    """Builds and draws a QWERTY keyboard overlay on a video frame."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        key_width: int = 70,
        key_height: int = 70,
        key_margin: int = 8,
        pinch_on_threshold: float = 40.0,
        pinch_off_threshold: float = 54.0,
        key_switch_delay: float = 0.15,
        hover_alpha: float = 0.32,
        hover_stable_frames: int = 3,
        key_hit_padding: int = 14,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.visible = False
        self.key_width = key_width
        self._held_modifiers: set[str] = set()
        self.key_height = key_height
        self.key_margin = key_margin
        self.keys: list[KeyButton] = self._build_layout()

        self.highlighted_label: str | None = None
        self.hovered_key: KeyButton | None = None
        self.pinch_on_threshold = pinch_on_threshold
        self.pinch_off_threshold = pinch_off_threshold
        self.key_switch_delay = key_switch_delay
        self.hover_alpha = hover_alpha
        self.hover_stable_frames = hover_stable_frames
        self.key_hit_padding = key_hit_padding
        self.is_pinching = False
        self.last_press_time = 0.0
        self.last_pressed_label: str | None = None
        self.press_feedback_until = 0.0
        self.press_anim_start = 0.0
        self.last_typed = ""
        self.active_gesture = "No Hand Detected"

        # Glide-typing: track which key was typed during current pinch
        self._pinch_active_key: str | None = None

        # Smoothed fingertip position (EMA)
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self.smooth_finger_pos: tuple[int, int] | None = None

        # Hover stability — prevents flicker between adjacent keys
        self._pending_key_label: str | None = None
        self._hover_stable_count = 0

        # Smoothed pinch distance
        self._pinch_dist_buffer: deque[float] = deque(maxlen=5)
        self._smooth_pinch_dist: float | None = None

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        self.is_pinching = False
        self._pinch_active_key = None
        if not visible:
            self._reset_smoothing()
            for modifier in tuple(self._held_modifiers):
                pyautogui.keyUp(modifier)
            self._held_modifiers.clear()

    def toggle(self) -> bool:
        self.set_visible(not self.visible)
        return self.visible

    def close(self) -> None:
        self.set_visible(False)

    def _build_layout(self) -> list[KeyButton]:
        """Create KeyButton objects from the QWERTY layout with computed positions."""
        keys: list[KeyButton] = []
        keyboard_height = len(QWERTY_LAYOUT) * (self.key_height + self.key_margin) + self.key_height
        start_y = self.frame_height - keyboard_height - 20

        for row_idx, row in enumerate(QWERTY_LAYOUT):
            row_width = len(row) * self.key_width + (len(row) - 1) * self.key_margin
            start_x = (self.frame_width - row_width) // 2
            y = start_y + row_idx * (self.key_height + self.key_margin)

            for col_idx, label in enumerate(row):
                x = start_x + col_idx * (self.key_width + self.key_margin)
                keys.append(KeyButton(label=label, x=x, y=y, width=self.key_width, height=self.key_height))

        space_width = min(self.frame_width - 80, 420)
        space_x = (self.frame_width - space_width) // 2
        space_y = start_y + len(QWERTY_LAYOUT) * (self.key_height + self.key_margin)
        keys.append(KeyButton(label="SPACE", x=space_x, y=space_y, width=space_width, height=self.key_height))

        return keys

    def get_key_at(self, px: int, py: int) -> KeyButton | None:
        """Return the key under the given point with expanded hit area."""
        for key in self.keys:
            if key.contains(px, py, padding=self.key_hit_padding):
                return key
        return None

    def _smooth_finger(self, finger_x: int, finger_y: int) -> tuple[int, int]:
        """EMA smoothing for fluid fingertip movement."""
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = float(finger_x)
            self._smooth_y = float(finger_y)
        else:
            a = self.hover_alpha
            self._smooth_x = a * finger_x + (1.0 - a) * self._smooth_x
            self._smooth_y = a * finger_y + (1.0 - a) * self._smooth_y

        pos = (int(self._smooth_x), int(self._smooth_y))
        self.smooth_finger_pos = pos
        return pos

    def _update_stable_hover(self, candidate: KeyButton | None) -> None:
        """Only switch hovered key after it stays stable for several frames."""
        candidate_label = candidate.label if candidate else None

        if candidate_label == self._pending_key_label:
            self._hover_stable_count += 1
        else:
            self._pending_key_label = candidate_label
            self._hover_stable_count = 1

        if self._hover_stable_count >= self.hover_stable_frames:
            self.hovered_key = candidate
            self.highlighted_label = candidate_label

    def _reset_smoothing(self) -> None:
        """Clear all smoothing state when hand leaves the frame."""
        self._smooth_x = None
        self._smooth_y = None
        self.smooth_finger_pos = None
        self._pending_key_label = None
        self._hover_stable_count = 0
        self._pinch_dist_buffer.clear()
        self._smooth_pinch_dist = None
        self.hovered_key = None
        self.highlighted_label = None

    def update_hover(self, finger_x: int, finger_y: int) -> KeyButton | None:
        """Detect which key the smoothed, stabilized fingertip is hovering over."""
        if finger_x < 0:
            self._reset_smoothing()
            return None

        smooth_x, smooth_y = self._smooth_finger(finger_x, finger_y)
        candidate = self.get_key_at(smooth_x, smooth_y)
        self._update_stable_hover(candidate)
        return self.hovered_key

    def _smooth_pinch_distance(self, thumb_tip: tuple[int, int], index_tip: tuple[int, int]) -> float:
        """Smooth pinch distance to avoid rapid pinch on/off flicker."""
        raw = pinch_distance(thumb_tip, index_tip)
        self._pinch_dist_buffer.append(raw)
        self._smooth_pinch_dist = float(np.mean(self._pinch_dist_buffer))
        return self._smooth_pinch_dist

    def _is_pinching_now(self, distance: float) -> bool:
        """Hysteresis: harder to release than to trigger, keeps pinch state stable."""
        if self.is_pinching:
            return distance < self.pinch_off_threshold
        return distance < self.pinch_on_threshold

    def _key_to_char(self, label: str) -> str:
        return " " if label == "SPACE" else label.lower()

    def _type_key(self, label: str, current_time: float) -> bool:
        """Send a real Windows key event and update visual typing feedback."""
        modifier_map = {
            "SHIFT": "shift",
            "CTRL": "ctrl",
            "ALT": "alt",
            "WIN": "winleft",
        }
        special_map = {
            "BKSP": "backspace",
            "ENTER": "enter",
            "TAB": "tab",
            "CAPS": "capslock",
            "ESC": "esc",
            "LEFT": "left",
            "RIGHT": "right",
            "UP": "up",
            "DOWN": "down",
        }

        if label in modifier_map:
            modifier = modifier_map[label]
            if modifier in self._held_modifiers:
                pyautogui.keyUp(modifier)
                self._held_modifiers.remove(modifier)
            else:
                pyautogui.keyDown(modifier)
                self._held_modifiers.add(modifier)
            char = ""
        elif label in special_map:
            pyautogui.press(special_map[label])
            if label == "BKSP":
                self.last_typed = self.last_typed[:-1]
            elif label == "ENTER":
                self.last_typed += "\n"
            elif label == "CAPS":
                pass
            char = ""
        else:
            char = self._key_to_char(label)
            pyautogui.write(char, interval=0)
            self.last_typed += char

        self.last_pressed_label = label
        self.press_feedback_until = current_time + 0.35
        self.press_anim_start = current_time
        self.last_press_time = current_time
        self.last_typed = self.last_typed[-100:]
        return True

    def handle_pinch_type(
        self,
        thumb_tip: tuple[int, int],
        index_tip: tuple[int, int],
    ) -> bool:
        """
        Type keys with glide support while pinching.

        - Pinch on a key → types immediately
        - Keep pinching and move to another key → types the new key (no release needed)
        - Same key won't repeat until pinch is released and pressed again
        """
        distance = self._smooth_pinch_distance(thumb_tip, index_tip)
        pinching = self._is_pinching_now(distance)
        typed = False
        current_time = time.time()
        current_label = self.hovered_key.label if self.hovered_key else None
        hover_ready = self._hover_stable_count >= self.hover_stable_frames

        if not pinching:
            self._pinch_active_key = None
        elif self.hovered_key and hover_ready and current_time - self.last_press_time >= self.key_switch_delay:
            new_pinch = not self.is_pinching
            moved_to_new_key = (
                current_label != self._pinch_active_key and self._pinch_active_key is not None
            )

            if new_pinch or moved_to_new_key:
                typed = self._type_key(current_label, current_time)
                self._pinch_active_key = current_label

        self.is_pinching = pinching
        return typed

    def _draw_key(self, frame: np.ndarray, key: KeyButton, now: float) -> None:
        """Draw a single rectangular key button onto the frame."""
        x, y, w, h = key.rect
        is_hovered = key.label == self.highlighted_label
        is_pressed = key.label == self.last_pressed_label and now < self.press_feedback_until

        if is_pressed:
            fill_color = (60, 60, 200)
            border_color = (120, 120, 255)
            draw_key_press_ripple(frame, x, y, w, h, self.press_anim_start)
        elif is_hovered:
            fill_color = (0, 140, 255)
            border_color = (0, 200, 255)
            if self.is_pinching:
                cv2.rectangle(frame, (x, y + h - 6), (x + w, y + h), (0, 80, 200), -1)
        else:
            fill_color = (45, 45, 45)
            border_color = (180, 180, 180)

        cv2.rectangle(frame, (x, y), (x + w, y + h), fill_color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)

        font_scale = 0.55 if key.label != "SPACE" else 0.5
        text_size = cv2.getTextSize(key.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.putText(
            frame,
            key.label,
            (x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )

    def draw_fingertip(self, frame: np.ndarray, finger_x: int, finger_y: int) -> None:
        """Draw a smooth marker trail on the index fingertip."""
        draw_x, draw_y = self.smooth_finger_pos if self.smooth_finger_pos else (finger_x, finger_y)
        color = (0, 0, 255) if self.is_pinching else (0, 255, 0)

        cv2.circle(frame, (draw_x, draw_y), 16, color, 2)
        cv2.circle(frame, (draw_x, draw_y), 6, color, cv2.FILLED)
        cv2.circle(frame, (finger_x, finger_y), 4, (255, 255, 255), cv2.FILLED)

    def draw_pinch_feedback(
        self,
        frame: np.ndarray,
        thumb_tip: tuple[int, int],
        index_tip: tuple[int, int],
    ) -> np.ndarray:
        """Draw pinch distance line between thumb and index."""
        line_color = (0, 0, 255) if self.is_pinching else (180, 180, 180)
        tip = self.smooth_finger_pos if self.smooth_finger_pos else index_tip
        cv2.circle(frame, thumb_tip, 10, line_color, cv2.FILLED)
        cv2.line(frame, thumb_tip, tip, line_color, 2)
        return frame

    def update_gesture(self, finger_pos: tuple[int, int] | None) -> None:
        """Update the active gesture label for the HUD."""
        if not finger_pos:
            self.active_gesture = "No Hand Detected"
        elif self.is_pinching and self.hovered_key:
            self.active_gesture = f"Glide to type — {self.hovered_key.label}"
        elif self.hovered_key:
            self.active_gesture = f"Hover — {self.hovered_key.label}"
        elif self.is_pinching:
            self.active_gesture = "Glide to keys"
        else:
            self.active_gesture = "Pinch & glide to type"

    def draw(self, frame: np.ndarray, finger_pos: tuple[int, int] | None = None) -> np.ndarray:
        """Draw the QWERTY keyboard overlay directly on the frame."""
        if not self.visible:
            return frame

        draw_typed_preview(frame, self.last_typed, y_offset=200)
        now = time.time()

        if self.is_pinching:
            cv2.putText(
                frame,
                "Glide finger across keys while pinching",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
            )

        for key in self.keys:
            self._draw_key(frame, key, now)

        if finger_pos:
            self.draw_fingertip(frame, finger_pos[0], finger_pos[1])

        if self.smooth_finger_pos and self.hovered_key:
            sx, sy = self.smooth_finger_pos
            cv2.line(frame, (sx, sy - 20), (sx, sy + 20), (0, 200, 255), 1)
            cv2.line(frame, (sx - 20, sy), (sx + 20, sy), (0, 200, 255), 1)

        return frame


def main():
    """Standalone demo entry point for virtual keyboard."""
    from hand_tracker import HandTracker
    from utils import (
        build_landmark_map,
        configure_pyautogui,
        get_fingertips,
        has_complete_landmarks,
        open_webcam,
        read_frame,
        release_camera,
        MAX_FRAME_FAILURES,
    )

    configure_pyautogui()

    cap = open_webcam(0)
    if cap is None:
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keyboard = VirtualKeyboard(frame_width, frame_height)
    tracker = HandTracker()
    frame_failures = 0

    print("Virtual Keyboard demo started. Press Q to quit.")

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                frame_failures += 1
                if frame_failures >= MAX_FRAME_FAILURES:
                    print("Error: Lost connection to webcam after repeated read failures.")
                    break
                continue
            frame_failures = 0

            frame = cv2.flip(frame, 1)
            frame, landmarks = tracker.findHands(frame, draw=True)

            finger_pos = thumb_pos = None
            if has_complete_landmarks(landmarks):
                index_tip, thumb_tip, _ = get_fingertips(build_landmark_map(landmarks))
                finger_pos, thumb_pos = index_tip, thumb_tip

            if finger_pos:
                keyboard.update_hover(finger_pos[0], finger_pos[1])
            else:
                keyboard.update_hover(-1, -1)

            if finger_pos and thumb_pos:
                keyboard.handle_pinch_type(thumb_pos, finger_pos)
                frame = keyboard.draw_pinch_feedback(frame, thumb_pos, finger_pos)

            keyboard.update_gesture(finger_pos)
            frame = keyboard.draw(frame, finger_pos=finger_pos)
            cv2.imshow("Virtual Keyboard", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        tracker.release()
        release_camera(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
