# core/gestures/orchestrator.py
"""
GestureOrchestrator — routes HandsResult to per-hand processors.

Exposes process(hands: HandsResult) as a drop-in replacement for
the old GestureProcessor.process(landmarks: list[Landmark]).
"""

import logging

from core.tracker import HandsResult
from core.display import VirtualDesktop, TrackpadZone
from core.actuator import MouseActuator
from core.gestures.right_hand import RightHandProcessor
from core.gestures.left_hand import LeftHandProcessor
from core.gestures.two_hand import TwoHandProcessor
from config import SAFE_DUAL_HAND_MODE

logger = logging.getLogger(__name__)


class GestureOrchestrator:
    """
    Coordinates right, left, and two-hand processors.

    Args:
        actuator: MouseActuator instance
        desktop:  VirtualDesktop for coordinate mapping
        trackpad: TrackpadZone for coordinate mapping
    """

    def __init__(
        self,
        actuator: MouseActuator,
        desktop: VirtualDesktop,
        trackpad: TrackpadZone,
    ) -> None:
        self._right = RightHandProcessor(actuator, desktop, trackpad)
        self._left = LeftHandProcessor(actuator)
        self._two = TwoHandProcessor(actuator)
        logger.info("GestureOrchestrator ready (dual-hand mode)")

    def process(self, hands: HandsResult) -> None:
        """Process one frame. Call once per frame regardless of hand visibility."""
        both_hands = hands.left is not None and hands.right is not None
        self._right.process(hands.right)

        if both_hands and SAFE_DUAL_HAND_MODE:
            # Safe mode: the right hand remains the mouse, while the left hand
            # cannot emit Windows shortcuts or modifier keys when both hands
            # are visible. This keeps ordinary two-hand positioning harmless.
            self._left.process(None)
            self._two.reset()
            return

        if hands.left is not None:
            self._left.process(hands.left)
        else:
            self._left.process(None)

        if both_hands:
            self._two.process(hands.left, hands.right)

    def reset(self) -> None:
        """Release any held OS input and reset all gesture state."""
        self._right.process(None)
        self._left.process(None)

    @property
    def right_state(self) -> str:
        return self._right._state.name
