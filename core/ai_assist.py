"""Optional AI vision assistant for explicit, user-triggered analysis.

The AI layer never sends mouse or keyboard events. It only analyzes a frame when
the user explicitly requests it, so network latency or model mistakes cannot
directly cause OS input.

Both Google Gemini and OpenAI are supported. Gemini is selected automatically
when GEMINI_API_KEY or GOOGLE_API_KEY is configured; otherwise OpenAI is used
when OPENAI_API_KEY is configured.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np


class AIAssistant:
    def __init__(self) -> None:
        self.provider = ""
        self.api_key = ""
        self.model = ""
        self.base_url = ""
        self._client: Any | None = None
        self._types: Any | None = None
        self._error: str | None = None

        gemini_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            self.model = (
                os.getenv("GEMINI_MODEL", "").strip()
                or "gemini-2.5-flash-lite"
            )
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=self.api_key)
                self._types = types
            except Exception as exc:
                self._error = str(exc)
            return

        if openai_key:
            self.provider = "openai"
            self.api_key = openai_key
            self.model = (
                os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.6"
            )
            self.base_url = os.getenv("OPENAI_BASE_URL", "").strip()
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
            return

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._error is None

    @property
    def status(self) -> str:
        if self.enabled:
            return f"AI ready ({self.provider}/{self.model})"
        if self._error:
            return f"AI unavailable: {self._error}"
        return "AI not configured: set GEMINI_API_KEY or OPENAI_API_KEY"

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
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
        return encoded.tobytes()

    def analyze(self, frame: np.ndarray, mode: str) -> str:
        """Analyze a single frame only when explicitly requested by the user."""
        if not self.enabled:
            return self.status

        try:
            image_bytes = self._encode_frame(frame)
            prompt = (
                "You are the visual diagnostic assistant for an air-mouse. "
                f"The application is currently in {mode} mode. "
                "Describe what the hand positions appear to be doing, "
                "whether the pose looks intentional or ambiguous, and one "
                "calibration suggestion when useful. Never request or "
                "perform a mouse click, key press, window switch, lock, "
                "or other OS action. Keep the response under 80 words."
            )

            if self.provider == "gemini":
                image_part = self._types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[image_part, prompt],
                )
                return (getattr(response, "text", "") or "").strip() or (
                    "AI returned no diagnostic text."
                )

            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:image/jpeg;base64,{image_b64}"
                                ),
                            },
                        ],
                    }
                ],
                max_output_tokens=180,
            )
            return (
                (getattr(response, "output_text", "") or "").strip()
                or "AI returned no diagnostic text."
            )
        except Exception as exc:
            return f"AI request failed: {exc}"
