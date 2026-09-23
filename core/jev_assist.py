"""Optional Jev semantic decision layer for live gesture interpretation."""

from __future__ import annotations

import os
from typing import Any


class JevGestureClassifier:
    def __init__(self, min_confidence: float = 0.68) -> None:
        self.min_confidence = min_confidence
        self.model = os.getenv("TYPESAFE_DEFAULT_MODEL", "").strip() or "jev-latest"
        self._api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self._available = False
        self._error: str | None = None
        self._client: Any | None = None

        if not self._api_key:
            return

        try:
            from typesafe_sdk import Choice, TypeSafeClient

            self._choice = Choice
            self._client = TypeSafeClient(
                api_key=self._api_key,
                model=self.model,
                timeout=3.0,
            )
            self._available = True
        except Exception as exc:
            self._error = str(exc)

    def close(self) -> None:
        """Release the optional SDK client when the application shuts down."""
        client = self._client
        self._client = None
        self._available = False
        if client is not None:
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    @property
    def enabled(self) -> bool:
        return self._available and bool(self._api_key)

    @property
    def status(self) -> str:
        if self.enabled:
            return f"Jev ready ({self.model})"
        if self._error:
            return f"Jev unavailable: {self._error}"
        return "Jev not configured: set TYPESAFE_API_KEY"

    def classify(
        self,
        candidate: str,
        mode: str,
        hands_hint: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "gesture": "unknown",
                "target_hand": "unknown",
                "confidence": 0.0,
                "provider": "jev",
            }

        criteria = {
            "none": "No deliberate action should occur.",
            "left_click": "The right hand intentionally performs a short thumb-index click.",
            "right_click": "The right hand intentionally performs a thumb-middle click.",
            "drag": "The right hand intentionally sustains a thumb-index drag.",
            "scroll_up": "The right hand intentionally makes the scroll sign and moves upward.",
            "scroll_down": "The right hand intentionally makes the scroll sign and moves downward.",
            "zoom_in": "Both hands intentionally perform the protected zoom gesture and move apart.",
            "zoom_out": "Both hands intentionally perform the protected zoom gesture and move together.",
            "keyboard_toggle": "The left hand intentionally holds the dedicated three-finger keyboard-toggle sign.",
            "keyboard_type": "The right hand intentionally pinches thumb and index to select a virtual-keyboard key.",
            "unknown": "The observation is ambiguous, contradictory, transitional, or insufficient.",
        }

        state = {
            "application_mode": mode,
            "candidate": candidate,
            "local_hand_tracker_observation": hands_hint,
            "instruction": "Choose one action only. Respect the performing hand. For uncertainty choose unknown.",
        }

        try:
            response = self._client.system_one(
                state=state,
                questions={
                    "gesture": self._choice(
                        instructions=(
                            "Select the single semantic air-mouse action that best "
                            "matches the observed state. This is a closed-set decision."
                        ),
                        criteria=criteria,
                    )
                },
            )

            answer = response.choices["gesture"]
            gesture = str(answer.choice).strip().lower()
            confidence = float(answer.confidence or 0.0)
            target_hand = {
                "keyboard_toggle": "left",
                "keyboard_type": "right",
                "left_click": "right",
                "right_click": "right",
                "drag": "right",
                "scroll_up": "right",
                "scroll_down": "right",
                "zoom_in": "both",
                "zoom_out": "both",
            }.get(gesture, "none")

            return {
                "gesture": gesture,
                "target_hand": target_hand,
                "confidence": max(0.0, min(1.0, confidence)),
                "probabilities": getattr(answer, "probabilities", {}) or {},
                "provider": "jev",
                "model": getattr(response, "model", self.model),
            }
        except Exception as exc:
            return {
                "gesture": "unknown",
                "target_hand": "unknown",
                "confidence": 0.0,
                "provider": "jev",
                "error": str(exc),
            }
