"""MediaPipe Hand Landmarker with temporal video tracking."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from config import (
    FLIP_HANDEDNESS,
    MP_DETECTION_CONFIDENCE,
    MP_MAX_HANDS,
    MP_MODEL_PATH,
    MP_MODEL_URL,
    MP_TRACKING_CONFIDENCE,
)

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 0.0


@dataclass
class HandsResult:
    left: list[Landmark] | None = None
    right: list[Landmark] | None = None


def resolve_model_path() -> Path:
    p = Path(MP_MODEL_PATH)
    if p.is_absolute():
        return p

    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        bundled = bundle_root / p
        if bundled.exists() and bundled.stat().st_size > 100_000:
            return bundled
        return Path(sys.executable).resolve().parent / p

    return Path(__file__).resolve().parents[1] / p


def ensure_model() -> Path:
    p = resolve_model_path()
    if p.exists() and p.stat().st_size > 100_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".download")
    try:
        req = Request(MP_MODEL_URL, headers={"User-Agent": "Unified-Airmouse/1.0"})
        with urlopen(req, timeout=60) as src, tmp.open("wb") as dst:
            while True:
                chunk = src.read(262144)
                if not chunk:
                    break
                dst.write(chunk)
        if tmp.stat().st_size <= 100_000:
            raise RuntimeError("model download too small")
        tmp.replace(p)
        logger.info("Downloaded MediaPipe hand model to %s", p)
        return p
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not provision MediaPipe model: {exc}") from exc


class HandTracker:
    def __init__(self) -> None:
        base = python.BaseOptions(model_asset_path=str(ensure_model()))
        opts = HandLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            num_hands=MP_MAX_HANDS,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self._detector = HandLandmarker.create_from_options(opts)
        self._rgb_buffer: np.ndarray | None = None
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def _timestamp_ms(self) -> int:
        current = int((time.monotonic() - self._start_time) * 1000)
        self._last_timestamp_ms = max(current, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def process(self, frame: np.ndarray) -> HandsResult:
        out = HandsResult()
        if frame is None or frame.size == 0:
            return out
        try:
            if self._rgb_buffer is None or self._rgb_buffer.shape != frame.shape or self._rgb_buffer.dtype != np.uint8:
                self._rgb_buffer = np.empty_like(frame)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
            result = self._detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb_buffer),
                self._timestamp_ms(),
            )
        except Exception as exc:
            logger.warning("MediaPipe processing error: %s", exc)
            return out

        for i, hand in enumerate(result.hand_landmarks or []):
            lms = [
                Landmark(
                    l.x,
                    l.y,
                    l.z,
                    getattr(l, "visibility", 0.0) or 0.0,
                )
                for l in hand
            ]
            label = result.handedness[i][0].category_name
            is_right = (
                label == "Right"
                if not FLIP_HANDEDNESS
                else label == "Left"
            )
            if is_right:
                out.right = lms
            else:
                out.left = lms
        return out

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
