from __future__ import annotations

"""
Shared UI overlay helpers for the virtual mouse and keyboard application.

Provides a consistent, minimal HUD with gesture labels and animated feedback.
"""

import time

import cv2
import numpy as np


# Professional dark-theme palette (BGR)
PANEL_COLOR = (30, 30, 30)
PANEL_BORDER = (70, 70, 70)
TEXT_WHITE = (235, 235, 235)
TEXT_MUTED = (160, 160, 160)
ACCENT_MOUSE = (90, 210, 90)
ACCENT_KEYBOARD = (50, 180, 255)
ACCENT_GESTURE = (200, 200, 120)
CLICK_RIPPLE = (220, 120, 80)
KEY_PRESS_GLOW = (100, 100, 255)


def _draw_translucent_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.65,
) -> None:
    """Draw a semi-transparent filled rectangle over the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PANEL_BORDER, 1)


def draw_notice(
    frame: np.ndarray,
    message: str,
    color: tuple[int, int, int] = TEXT_MUTED,
    y: int = 110,
) -> None:
    """Draw an informational or warning message below the status bar."""
    cv2.putText(
        frame,
        message,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
    )


def draw_status_hud(
    frame: np.ndarray,
    mode: str,
    gesture: str,
    shortcuts: str = "M = Mouse   K = Keyboard   A = AI Analyze   Q = Quit",
) -> None:
    """Draw the top status bar showing active mode and current gesture."""
    height, width = frame.shape[:2]
    mode_label = "MOUSE MODE" if mode == "mouse" else "KEYBOARD MODE"
    mode_color = ACCENT_MOUSE if mode == "mouse" else ACCENT_KEYBOARD

    _draw_translucent_rect(frame, 0, 0, width, 72, PANEL_COLOR)
    _draw_translucent_rect(frame, 0, height - 36, width, height, PANEL_COLOR)

    cv2.putText(
        frame,
        mode_label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        mode_color,
        2,
    )
    cv2.putText(
        frame,
        f"|  {gesture}",
        (200, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ACCENT_GESTURE,
        2,
    )
    cv2.putText(
        frame,
        shortcuts,
        (16, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_MUTED,
        1,
    )


def draw_click_ripple(
    frame: np.ndarray,
    center: tuple[int, int],
    anim_start: float,
    duration: float = 0.45,
) -> None:
    """
    Draw an expanding ripple animation at the click location.

    Three concentric rings expand outward and fade, giving clear
    visual confirmation that a mouse click was registered.
    """
    elapsed = time.time() - anim_start
    if elapsed > duration or center is None:
        return

    for ring in range(3):
        ring_delay = ring * 0.1
        ring_elapsed = elapsed - ring_delay
        if ring_elapsed <= 0:
            continue

        progress = min(ring_elapsed / (duration - ring_delay), 1.0)
        radius = int(12 + progress * 55)
        fade = max(0.0, 1.0 - progress)
        color = (
            int(CLICK_RIPPLE[0] * fade),
            int(CLICK_RIPPLE[1] * fade),
            int(CLICK_RIPPLE[2] * fade),
        )
        thickness = max(1, int(3 * fade))
        cv2.circle(frame, center, radius, color, thickness)

    # Brief center flash at the start of the animation
    if elapsed < 0.12:
        cv2.circle(frame, center, 10, CLICK_RIPPLE, cv2.FILLED)


def draw_key_press_ripple(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    anim_start: float,
    duration: float = 0.45,
) -> None:
    """
    Draw an expanding glow around a key when it is pressed.

    A bright border expands outward from the key rectangle and fades,
    providing animated confirmation of the keystroke.
    """
    elapsed = time.time() - anim_start
    if elapsed > duration:
        return

    progress = elapsed / duration
    expand = int(14 * (1.0 - progress))
    fade = max(0.0, 1.0 - progress)

    glow_color = (
        int(KEY_PRESS_GLOW[0] * fade),
        int(KEY_PRESS_GLOW[1] * fade),
        int(KEY_PRESS_GLOW[2] * fade),
    )

    cv2.rectangle(
        frame,
        (x - expand, y - expand),
        (x + w + expand, y + h + expand),
        glow_color,
        max(1, int(3 * fade)),
    )

    if elapsed < 0.1:
        inner = frame.copy()
        cv2.rectangle(inner, (x, y), (x + w, y + h), KEY_PRESS_GLOW, -1)
        cv2.addWeighted(inner, 0.35, frame, 0.65, 0, frame)


def draw_ai_insight(
    frame: np.ndarray,
    insight: dict,
    enabled: bool = True,
    error: str | None = None,
    y_offset: int = 130,
) -> None:
    """Draw the Groq AI gesture analysis panel."""
    if error:
        _draw_translucent_rect(frame, 10, y_offset, frame.shape[1] - 10, y_offset + 70, PANEL_COLOR, 0.7)
        cv2.putText(frame, "AI: Disabled", (18, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.putText(frame, error[:70], (18, y_offset + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1)
        return

    if not enabled:
        return

    gesture = insight.get("gesture", "idle")
    confidence = insight.get("confidence", 0.0)
    action = insight.get("action", "")
    explanation = insight.get("explanation", "")

    _draw_translucent_rect(frame, 10, y_offset, frame.shape[1] - 10, y_offset + 70, PANEL_COLOR, 0.7)
    cv2.putText(
        frame,
        f"AI: {gesture}  ({confidence:.0%})",
        (18, y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Action: {action}"[:55],
        (18, y_offset + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_WHITE,
        1,
    )
    cv2.putText(
        frame,
        explanation[:65],
        (18, y_offset + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEXT_MUTED,
        1,
    )


def draw_typed_preview(
    frame: np.ndarray,
    text: str,
    y_offset: int = 90,
) -> None:
    """Draw a typed-text preview panel below the status bar."""
    if not text:
        return

    display = text[-35:]
    panel_width = min(len(display) * 14 + 30, frame.shape[1] - 20)
    _draw_translucent_rect(frame, 10, y_offset, 10 + panel_width, y_offset + 32, PANEL_COLOR, 0.7)
    cv2.putText(
        frame,
        f"Typed: {display}",
        (18, y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_WHITE,
        1,
    )
