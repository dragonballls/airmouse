"""Optional AI vision assistant for explicit, user-triggered analysis.

The AI layer never sends mouse or keyboard events. It only analyzes a frame when
the user explicitly requests it, so network latency or model mistakes cannot
directly cause OS input.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np


class AIAssistant:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6").strip() or "gpt-5.6"
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        self._client: Any | None = None
        self._error: str | None = None

        if not self.api_key:
            return

        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": 15.0,
                "max_retries": 1,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            self._error = str(exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._error is None

    @property
    def status(self) -> str:
        if self.enabled:
            return f"AI ready ({self.model})"
        if self._error:
            return f"AI unavailable: {self._error}"
        return "AI not configured: set OPENAI_API_KEY"

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        if frame is None or frame.size == 0:
            raise ValueError("empty camera frame")

        image = frame
        height, width = image.shape[:2]
        max_width = 768
        if width > max_width:
            scale = max_width / width
            image = cv2.resize(
                image,
                (max_width, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 65],
        )
        if not ok:
            raise RuntimeError("could not encode camera frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def analyze(self, frame: np.ndarray, mode: str) -> str:
        """Analyze a single frame only when explicitly requested by the user."""
        if not self.enabled:
            return self.status

        try:
            image_b64 = self._encode_frame(frame)
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You are the visual diagnostic assistant for an air-mouse. "
                                    f"The application is currently in {mode} mode. "
                                    "Describe what the hand positions appear to be doing, "
                                    "whether the pose looks intentional or ambiguous, and one "
                                    "calibration suggestion when useful. Never request or "
                                    "perform a mouse click, key press, window switch, lock, "
                                    "or other OS action. Keep the response under 80 words."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        ],
                    }
                ],
                max_output_tokens=180,
            )
            text = getattr(response, "output_text", "") or ""
            return text.strip() or "AI returned no diagnostic text."
        except Exception as exc:
            return f"AI request failed: {exc}"
