# core/gestures/utils.py
"""Shared geometry and conservative gesture helpers."""

from __future__ import annotations

import numpy as np

from core.tracker import Landmark


def dist3d(a: Landmark, b: Landmark) -> float:
    """3D Euclidean distance in normalized MediaPipe coordinate space."""
    return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))


def hand_scale(lm: list[Landmark]) -> float:
    """Return a stable palm-size estimate used to normalize distances."""
    if len(lm) < 13:
        return 1.0
    return max(dist3d(lm[0], lm[9]), 1e-4)


def normalized_distance(lm: list[Landmark], a: int, b: int) -> float:
    """Distance between two landmarks relative to palm size."""
    return dist3d(lm[a], lm[b]) / hand_scale(lm)


def is_extended(lm_or_tip, pip_or_index) -> bool:
    """
    Determine extension using distance from the wrist rather than only y.
    This is substantially more tolerant of hand rotation.
    """
    if isinstance(lm_or_tip, list):
        lm = lm_or_tip
        tip_index = int(pip_or_index)
        pip_index = {8: 6, 12: 10, 16: 14, 20: 18}.get(tip_index)
        if pip_index is None:
            raise ValueError("unsupported finger index")
        tip = lm[tip_index]
        pip = lm[pip_index]
        wrist = lm[0]
    else:
        tip = lm_or_tip
        pip = pip_or_index
        # Compatibility path for callers that already have landmarks.
        # A direct comparison is less rotation-safe but remains useful.
        return tip.y < pip.y

    return dist3d(tip, wrist) > dist3d(pip, wrist) * 1.08


def is_curled(lm_or_tip, pip_or_index) -> bool:
    """Inverse of is_extended with the same compatibility behavior."""
    if isinstance(lm_or_tip, list):
        lm = lm_or_tip
        return not is_extended(lm, pip_or_index)
    return lm_or_tip.y > pip_or_index.y


def finger_extended(lm: list[Landmark], tip: int, pip: int) -> bool:
    return dist3d(lm[tip], lm[0]) > dist3d(lm[pip], lm[0]) * 1.08


def finger_curled(lm: list[Landmark], tip: int, pip: int) -> bool:
    return not finger_extended(lm, tip, pip)


def is_fist(lm: list[Landmark]) -> bool:
    return (
        finger_curled(lm, 8, 6)
        and finger_curled(lm, 12, 10)
        and finger_curled(lm, 16, 14)
        and finger_curled(lm, 20, 18)
        and normalized_distance(lm, 4, 8) > 0.35
    )


def is_peace_sign(lm: list[Landmark]) -> bool:
    return (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 12, 10)
        and finger_curled(lm, 16, 14)
        and finger_curled(lm, 20, 18)
    )


def is_open_palm(lm: list[Landmark]) -> bool:
    return (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 12, 10)
        and finger_extended(lm, 16, 14)
        and finger_extended(lm, 20, 18)
    )


def is_v_sign(lm: list[Landmark]) -> bool:
    return (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 12, 10)
        and finger_curled(lm, 16, 14)
        and finger_curled(lm, 20, 18)
    )


def is_four_fingers(lm: list[Landmark]) -> bool:
    return (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 12, 10)
        and finger_extended(lm, 16, 14)
        and finger_extended(lm, 20, 18)
    )


def is_three_finger_keyboard_pose(lm: list[Landmark]) -> bool:
    """Index+middle+ring up, pinky down; thumb intentionally ignored."""
    if len(lm) < 21:
        return False
    return (
        finger_extended(lm, 8, 6)
        and finger_extended(lm, 12, 10)
        and finger_extended(lm, 16, 14)
        and finger_curled(lm, 20, 18)
    )
