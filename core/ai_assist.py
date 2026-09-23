"""AI vision assistant and semantic gesture classifier.

Both Google Gemini and OpenAI are supported. Gemini can act as the live semantic
gesture gate for the air-mouse: it classifies deliberate candidates into a
closed set of safe labels. It never receives permission to issue OS commands.
"""

from __future__ import annotations

import base64
import json
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
                or "gemini-3.5-flash-lite"
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
            self.model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.6"
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
        max_width = 640
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

    def classify_semantic_gesture(
        self,
        frame: np.ndarray,
        candidate: str,
        mode: str,
        hands_hint: str,
    ) -> dict[str, Any]:
        """Classify one deliberate gesture candidate into a closed safe label."""
        if not self.enabled:
            return {
                "gesture": "unknown",
                "target_hand": "unknown",
                "confidence": 0.0,
            }

        if self.provider != "gemini":
            return {
                "gesture": "unknown",
                "target_hand": "unknown",
                "confidence": 0.0,
            }

        image_bytes = self._encode_frame(frame)
        image_part = self._types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        allowed = [
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
        ]
        schema = {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": allowed,
                    "description": "The single semantic gesture visible now.",
                },
                "target_hand": {
                    "type": "string",
                    "enum": ["left", "right", "both", "none", "unknown"],
                    "description": "Which hand or hands perform the gesture.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["gesture", "target_hand", "confidence"],
            "additionalProperties": False,
        }

        prompt = (
            "Classify an air-mouse hand gesture from the camera image. "
            "This is a closed-set classifier, not an assistant. "
            f"Operating mode: {mode}. Local candidate: {candidate}. "
            f"Local tracker hint (not authoritative): {hands_hint}. "
            "Inspect the visible hands yourself. Distinguish left and right "
            "hands and recognize deliberate signs. Only choose one allowed "
            "gesture label. Do not invent labels or OS commands. "
            "For an uncertain, occluded, transitional, or absent gesture use "
            "unknown with low confidence. A click label means the corresponding "
            "pinch is intentional; drag means a sustained intentional pinch; "
            "scroll means the deliberate scroll sign plus motion; zoom means "
            "both hands performing the deliberate zoom gesture; keyboard_toggle "
            "means the dedicated three-finger left-hand sign. "
            "Return only the requested JSON object."
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=[image_part, prompt],
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                media_resolution=self._types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                thinking_config=self._types.ThinkingConfig(
                    thinking_level="minimal"
                ),
            ),
        )

        raw = (getattr(response, "text", "") or "").strip()
        if not raw:
            return {"gesture": "unknown", "target_hand": "unknown", "confidence": 0.0}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"gesture": "unknown", "target_hand": "unknown", "confidence": 0.0}

        return parsed

    def analyze(self, frame: np.ndarray, mode: str) -> str:
        """Analyze one frame when explicitly requested with A."""
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
