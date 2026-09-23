# core/tracker.py
"""MediaPipe Hand Landmarker with automatic official model provisioning."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request,urlopen
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
from config import FLIP_HANDEDNESS,MP_DETECTION_CONFIDENCE,MP_MAX_HANDS,MP_MODEL_PATH,MP_MODEL_URL,MP_TRACKING_CONFIDENCE,INFERENCE_WIDTH,INFERENCE_HEIGHT
logger=logging.getLogger(__name__)
@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float=0.0
@dataclass
class HandsResult:
    left: list[Landmark]|None=None
    right: list[Landmark]|None=None
def resolve_model_path()->Path:
    p=Path(MP_MODEL_PATH)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1]/p
def ensure_model()->Path:
    p=resolve_model_path()
    if p.exists() and p.stat().st_size>100_000: return p
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".download")
    try:
        req=Request(MP_MODEL_URL,headers={"User-Agent":"Unified-Airmouse/1.0"})
        with urlopen(req,timeout=60) as src,tmp.open("wb") as dst:
            while True:
                chunk=src.read(262144)
                if not chunk: break
                dst.write(chunk)
        if tmp.stat().st_size<=100_000: raise RuntimeError("model download too small")
        tmp.replace(p)
        logger.info("Downloaded MediaPipe hand model to %s",p)
        return p
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not provision MediaPipe model: {exc}") from exc
class HandTracker:
    def __init__(self)->None:
        base=python.BaseOptions(model_asset_path=str(ensure_model()))
        opts=HandLandmarkerOptions(base_options=base,running_mode=RunningMode.VIDEO,
            num_hands=MP_MAX_HANDS,min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,min_tracking_confidence=MP_TRACKING_CONFIDENCE)
        self._detector=HandLandmarker.create_from_options(opts)
    def process(self,frame:np.ndarray)->HandsResult:
        out=HandsResult()
        if frame is None or frame.size==0:return out
        try:
            if frame.shape[1] != INFERENCE_WIDTH or frame.shape[0] != INFERENCE_HEIGHT:
                frame=cv2.resize(frame,(INFERENCE_WIDTH,INFERENCE_HEIGHT),interpolation=cv2.INTER_AREA)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            timestamp_ms=time.monotonic_ns()//1_000_000
            result=self._detector.detect_for_video(image,timestamp_ms)
        except Exception as exc:
            logger.warning("MediaPipe processing error: %s",exc)
            return out
        for i,hand in enumerate(result.hand_landmarks or []):
            lms=[Landmark(l.x,l.y,l.z,getattr(l,"visibility",0.0) or 0.0) for l in hand]
            label=result.handedness[i][0].category_name
            is_right=(label=="Right") if not FLIP_HANDEDNESS else (label=="Left")
            if is_right: out.right=lms
            else: out.left=lms
        return out
    def close(self)->None:self._detector.close()
    def __enter__(self)->"HandTracker":return self
    def __exit__(self,*_)->None:self.close()
