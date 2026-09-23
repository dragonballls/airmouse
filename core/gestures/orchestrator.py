# core/gestures/orchestrator.py
"""Routes right-hand, left-hand, and deliberate two-hand processing."""

from __future__ import annotations

import logging

from core.actuator import MouseActuator
from core.display import TrackpadZone, VirtualDesktop
from core.gestures.left_hand import LeftHandProcessor
from core.gestures.right_hand import RightHandProcessor
from core.gestures.two_hand import TwoHandProcessor
from core.tracker import HandsResult

logger = logging.getLogger(__name__)


class GestureOrchestrator:
    def __init__(
        self,
        actuator: MouseActuator,
        desktop: VirtualDesktop,
        trackpad: TrackpadZone,
    ) -> None:
        self._right = RightHandProcessor(actuator, desktop, trackpad)
        self._left = LeftHandProcessor(actuator)
        self._two = TwoHandProcessor(actuator)
        logger.info("GestureOrchestrator ready (safe dual-hand mode)")

    def process(self, hands: HandsResult) -> None:
        both = hands.left is not None and hands.right is not None
        two_hand_exclusive = False

        if both:
            two_hand_exclusive = self._two.process(hands.left, hands.right)
        else:
            self._two.reset()

        self._right.process(hands.right, suppress_actions=two_hand_exclusive)
        self._left.process(hands.left)

    def reset(self) -> None:
        self._right.process(None)
        self._left.reset()
        self._two.reset()

    @property
    def right_state(self) -> str:
        return self._right._state.name
