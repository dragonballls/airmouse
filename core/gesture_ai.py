"""AI semantic gesture classification for the live air-mouse pipeline.

The local tracker remains responsible for low-latency landmarks and pointer
movement. Gemini is used to classify deliberate gesture candidates into a small,
closed set of safe semantic labels. AI output never contains OS commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any


GESTURE_LABELS = frozenset(
    {
        "none",
        "left_click",
        "right_click",
        "drag",
        "scroll_up",
        "scroll_down",
        "zoom_in",
        "zoom_out",
        "keyboard_toggle",
        "unknown",
    }
)


@dataclass(frozen=True)
class GestureDecision:
    gesture: str
    target_hand: str
    confidence: float
    candidate: str
    created_at: float


class SemanticGestureAI:
    """Async Gemini classifier with latest-request-wins behavior."""

    def __init__(self, assistant: Any, min_confidence: float = 0.68) -> None:
        self._assistant = assistant
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._future: Any | None = None
        self._executor: Any | None = None
        self._last_request_at = 0.0
        self._last_submitted_candidate = ""
        self._decision: GestureDecision | None = None

        if self.enabled:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="airmouse-gemini",
            )

    @property
    def enabled(self) -> bool:
        return bool(self._assistant.enabled and self._assistant.provider == "gemini")

    @property
    def status(self) -> str:
        if self.enabled:
            return "AI gesture gate ready"
        return "AI gesture gate unavailable"

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def clear(self) -> None:
        with self._lock:
            self._decision = None
            self._last_submitted_candidate = ""

    def submit(
        self,
        frame: Any,
        candidate: str,
        mode: str,
        hands_hint: str,
    ) -> None:
        """Submit a candidate only when it changes or enough time has elapsed."""
        if not self.enabled or frame is None:
            return

        now = time.perf_counter()
        with self._lock:
            if (
                candidate == self._last_submitted_candidate
                and now - self._last_request_at < 0.45
            ):
                return
            if self._future is not None and not self._future.done():
                return
            self._last_submitted_candidate = candidate
            self._last_request_at = now
            executor = self._executor

        if executor is None:
            return

        snapshot = frame.copy()
        self._future = executor.submit(
            self._classify,
            snapshot,
            candidate,
            mode,
            hands_hint,
        )

    def poll(self) -> GestureDecision | None:
        future = self._future
        if future is None or not future.done():
            return None

        self._future = None
        try:
            decision = future.result()
        except Exception:
            return None

        with self._lock:
            self._decision = decision
        return decision

    def current(
        self,
        candidate: str,
        max_age: float = 1.0,
    ) -> GestureDecision | None:
        with self._lock:
            decision = self._decision
        if decision is None:
            return None
        if decision.candidate != candidate:
            return None
        if time.perf_counter() - decision.created_at > max_age:
            return None
        if decision.gesture == "unknown" or decision.confidence < self._min_confidence:
            return None
        return decision

    def _classify(
        self,
        frame: Any,
        candidate: str,
        mode: str,
        hands_hint: str,
    ) -> GestureDecision:
        result = self._assistant.classify_semantic_gesture(
            frame=frame,
            candidate=candidate,
            mode=mode,
            hands_hint=hands_hint,
        )
        gesture = str(result.get("gesture", "unknown")).strip().lower()
        if gesture not in GESTURE_LABELS:
            gesture = "unknown"

        target_hand = str(result.get("target_hand", "unknown")).strip().lower()
        if target_hand not in {"left", "right", "both", "none", "unknown"}:
            target_hand = "unknown"

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return GestureDecision(
            gesture=gesture,
            target_hand=target_hand,
            confidence=confidence,
            candidate=candidate,
            created_at=time.perf_counter(),
        )
