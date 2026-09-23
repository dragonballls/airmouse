# core/gestures/orchestrator.py
"""Routes the AirNav-style two-hand interaction model.

When both hands are visible:
- left index finger = cursor movement
- right hand = click / double-click / right-click / drag / scroll
- both-hand pinch = explicit two-hand zoom

When only the right hand is visible, it also moves the cursor as a fallback.
The left-fist lock gesture is always handled by the left processor.
"""

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
        ai_required: bool = False,
    ) -> None:
        # Basic input is deliberately local-first. ai_required remains in the
        # public signature for compatibility with the existing application.
        self._right = RightHandProcessor(actuator, desktop, trackpad)
        self._left = LeftHandProcessor(actuator, desktop, trackpad)
        self._two = TwoHandProcessor(actuator)
        self._ai_required = bool(ai_required)
        logger.info(
            "GestureOrchestrator ready (AirNav dual-hand mode, AI gate=%s)",
            "on" if self._ai_required else "off",
        )

    def process(
        self,
        hands: HandsResult,
        ai_gesture: str | None = None,
    ) -> None:
        both = hands.left is not None and hands.right is not None
        two_hand_exclusive = False

        if both:
            two_hand_exclusive = self._two.process(
                hands.left,
                hands.right,
                ai_gesture=ai_gesture,
                ai_required=self._ai_required,
            )
        else:
            self._two.reset()

        # Left hand owns cursor movement when both hands are present. A
        # deliberate two-hand gesture temporarily suspends cursor movement.
        self._left.process(
            hands.left,
            suppress_cursor=two_hand_exclusive,
        )

        # Right hand owns click/drag actions. It gets cursor movement only when
        # it is the sole visible hand.
        self._right.process(
            hands.right,
            suppress_actions=two_hand_exclusive,
            ai_gesture=ai_gesture,
            ai_required=self._ai_required,
            cursor_enabled=not both and not two_hand_exclusive,
        )

    def reset(self) -> None:
        self._right.process(None)
        self._left.reset()
        self._two.reset()

    @property
    def right_state(self) -> str:
        return self._right._state.name

    @property
    def mouse_locked(self) -> bool:
        return self._left._actuator.cursor_locked
