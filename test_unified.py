"""CI-safe tests for the unified keyboard/control integration."""

from unittest.mock import Mock

import core.virtual_keyboard as vk
from core.actuator import MouseActuator
from core.display import build_trackpad_zone, build_virtual_desktop
from core.gestures import GestureOrchestrator
from core.gestures.left_hand import LeftHandProcessor
from core.gestures.utils import is_three_finger_keyboard_pose, normalized_distance
from core.tracker import HandsResult, Landmark
from core.virtual_keyboard import VirtualKeyboard


def _landmarks() -> list[Landmark]:
    # Neutral synthetic hand. Individual gesture geometry is changed in the
    # focused tests below; no camera is required.
    return [Landmark(0.5, 0.5, 0.0) for _ in range(21)]


def test_keyboard_contains_practical_keys():
    keyboard = VirtualKeyboard(1280, 720)
    labels = {key.label for key in keyboard.keys}
    required = {
        "ESC", "F1", "F12", "BACKSPACE", "TAB", "CAPS", "ENTER",
        "SHIFT", "SPACE", "CTRL", "ALT", "WIN", "LEFT", "RIGHT",
        "Q", "A", "Z", "0",
    }
    assert required <= labels
    keyboard.close()


def test_keyboard_sends_key_events():
    keyboard = VirtualKeyboard(1280, 720)
    vk.pyautogui.write = Mock()
    vk.pyautogui.press = Mock()
    vk.pyautogui.keyDown = Mock()
    vk.pyautogui.keyUp = Mock()

    q = next(key for key in keyboard.keys if key.label == "Q")
    space = next(key for key in keyboard.keys if key.label == "SPACE")
    backspace = next(key for key in keyboard.keys if key.label == "BACKSPACE")

    keyboard._type_key(q.label, 0.0)
    keyboard._type_key(space.label, 0.1)
    keyboard._type_key(backspace.label, 0.2)

    assert vk.pyautogui.write.called
    assert vk.pyautogui.press.called
    assert keyboard.last_typed == "q"
    keyboard.close()


def test_keyboard_pose_uses_three_fingers_without_thumb_requirement():
    lm = _landmarks()
    lm[0] = Landmark(0.5, 0.8, 0.0)
    lm[6] = Landmark(0.48, 0.55, 0.0)
    lm[8] = Landmark(0.48, 0.25, 0.0)
    lm[10] = Landmark(0.50, 0.55, 0.0)
    lm[12] = Landmark(0.50, 0.22, 0.0)
    lm[14] = Landmark(0.52, 0.55, 0.0)
    lm[16] = Landmark(0.52, 0.24, 0.0)
    lm[18] = Landmark(0.54, 0.55, 0.0)
    lm[20] = Landmark(0.54, 0.70, 0.0)
    assert is_three_finger_keyboard_pose(lm)


def test_normalized_distance_changes_with_hand_scale():
    lm = _landmarks()
    lm[0] = Landmark(0.5, 0.5, 0.0)
    lm[9] = Landmark(0.5, 0.4, 0.0)
    lm[4] = Landmark(0.50, 0.45, 0.0)
    lm[8] = Landmark(0.52, 0.45, 0.0)
    small = normalized_distance(lm, 4, 8)
    lm[9] = Landmark(0.5, 0.2, 0.0)
    large = normalized_distance(lm, 4, 8)
    assert large < small


def test_left_hand_is_safe_by_default():
    actuator = Mock(spec=MouseActuator)
    processor = LeftHandProcessor(actuator)
    processor.process(_landmarks())
    actuator.win_d.assert_not_called()
    actuator.alt_tab.assert_not_called()
    actuator.win_tab.assert_not_called()
    actuator.win_key.assert_not_called()
    actuator.alt_left.assert_not_called()


def test_orchestrator_handles_empty_and_dual_hand_input():
    desktop = build_virtual_desktop()
    trackpad = build_trackpad_zone()
    actuator = MouseActuator(desktop.total_width, desktop.total_height)
    processor = GestureOrchestrator(actuator, desktop, trackpad)
    processor.process(HandsResult())
    hand = _landmarks()
    processor.process(HandsResult(left=hand, right=hand))
    processor.reset()


def test_ai_assistant_is_optional_without_any_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from core.ai_assist import AIAssistant
    assistant = AIAssistant()
    assert assistant.enabled is False


def test_semantic_gesture_ai_accepts_closed_set_result():
    from core.gesture_ai import SemanticGestureAI

    class FakeAssistant:
        enabled = True
        provider = "gemini"

        def classify_semantic_gesture(self, **_kwargs):
            return {
                "gesture": "right_click",
                "target_hand": "right",
                "confidence": 0.91,
            }

    semantic = SemanticGestureAI(FakeAssistant())
    decision = semantic._classify(
        _landmarks(),
        "middle_pinch",
        "mouse",
        "right hand",
    )
    assert decision.gesture == "right_click"
    assert decision.target_hand == "right"
    assert decision.confidence == 0.91
    semantic.close()


def test_keyboard_does_not_type_without_ai_authorization():
    keyboard = VirtualKeyboard(1280, 720)
    vk.pyautogui.write = Mock()
    vk.pyautogui.press = Mock()
    keyboard.hovered_key = next(key for key in keyboard.keys if key.label == "Q")
    keyboard.highlighted_label = "Q"
    keyboard._hover_stable_count = keyboard.hover_stable_frames
    keyboard.handle_pinch_type((100, 100), (105, 105), ai_allowed=False)
    assert not vk.pyautogui.write.called
    assert not vk.pyautogui.press.called
    keyboard.close()


def test_jev_classifier_is_optional_without_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from core.jev_assist import JevGestureClassifier

    classifier = JevGestureClassifier()
    assert classifier.enabled is False


def test_jev_classifier_uses_typed_choice(monkeypatch):
    import sys
    import types as pytypes

    class FakeAnswer:
        choice = "right_click"
        confidence = 0.93
        probabilities = {"right_click": 0.93, "none": 0.07}

    class FakeResponse:
        model = "jev-test"

        @property
        def choices(self):
            return {"gesture": FakeAnswer()}

    class FakeChoice:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def system_one(self, state, questions):
            assert state["candidate"] == "middle_pinch"
            assert "right_click" in questions["gesture"].criteria
            return FakeResponse()

    fake = pytypes.ModuleType("typesafe_sdk")
    fake.Choice = FakeChoice
    fake.TypeSafeClient = FakeClient
    monkeypatch.setitem(sys.modules, "typesafe_sdk", fake)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    from core import jev_assist
    classifier = jev_assist.JevGestureClassifier()
    result = classifier.classify("middle_pinch", "mouse", "right hand")
    assert result["gesture"] == "right_click"
    assert result["target_hand"] == "right"
    assert result["confidence"] == 0.93


def test_semantic_decision_survives_release_grace():
    from core.gesture_ai import GestureDecision, SemanticGestureAI
    import time

    class FakeAssistant:
        enabled = False
        provider = ""

    semantic = SemanticGestureAI(FakeAssistant())
    semantic._decision = GestureDecision(
        gesture="left_click",
        target_hand="right",
        confidence=0.95,
        candidate="index_pinch",
        created_at=time.perf_counter(),
    )
    decision = semantic.current(None, max_age=0.1, grace=0.9)
    assert decision is not None
    assert decision.gesture == "left_click"
    assert decision.target_hand == "right"
    semantic.close()


def test_jev_configuration_is_optional(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from core.jev_assist import JevGestureClassifier

    classifier = JevGestureClassifier()
    assert classifier.enabled is False
    assert "TYPESAFE_API_KEY" in classifier.status


def test_video_tracker_blank_frame_returns_empty_result(monkeypatch):
    import numpy as np
    import core.tracker as tracker_module

    class FakeDetector:
        def detect_for_video(self, image, timestamp_ms):
            assert timestamp_ms >= 0
            return type("Result", (), {"hand_landmarks": [], "handedness": []})()

        def close(self):
            return None

    monkeypatch.setattr(tracker_module.HandLandmarker, "create_from_options", lambda _opts: FakeDetector())
    monkeypatch.setattr(tracker_module, "ensure_model", lambda: object())

    tracker = tracker_module.HandTracker()
    result = tracker.process(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.left is None
    assert result.right is None
    tracker.close()


def test_camera_handedness_is_swapped_for_unmirrored_input():
    from config import FLIP_HANDEDNESS
    assert FLIP_HANDEDNESS is True


def test_stale_ai_generation_cannot_be_accepted():
    from core.gesture_ai import GestureDecision, SemanticGestureAI
    import time

    class FakeAssistant:
        enabled = False
        provider = ""

    semantic = SemanticGestureAI(FakeAssistant())
    semantic._decision = GestureDecision(
        gesture="right_click",
        target_hand="right",
        confidence=0.95,
        candidate="middle_pinch",
        created_at=time.perf_counter(),
    )
    semantic._generation = 8
    semantic._future_generation = 7
    semantic._future = None
    # A generation mismatch is represented as an invalidated decision after clear.
    semantic.clear()
    assert semantic.current(None, grace=1.0) is None
    semantic.close()



def test_semantic_ai_retains_valid_decision_through_unknown_refresh():
    from core.gesture_ai import GestureDecision, SemanticGestureAI

    class FakeAssistant:
        enabled = False
        provider = ""

    class FakeFuture:
        def done(self):
            return True

        def result(self):
            return GestureDecision(
                gesture="unknown",
                target_hand="unknown",
                confidence=0.0,
                candidate="index_pinch",
                created_at=0.0,
            )

    import time
    semantic = SemanticGestureAI(FakeAssistant())
    semantic._decision = GestureDecision(
        gesture="left_click",
        target_hand="right",
        confidence=0.95,
        candidate="index_pinch",
        created_at=time.perf_counter(),
    )
    semantic._generation = 4
    semantic._future_generation = 4
    semantic._future = FakeFuture()
    decision = semantic.poll()
    assert decision is not None
    assert decision.gesture == "left_click"
    semantic.close()


def test_semantic_ai_rejects_wrong_hand_for_candidate():
    from core.gesture_ai import SemanticGestureAI

    class FakeAssistant:
        enabled = True
        provider = "gemini"

        def classify_semantic_gesture(self, **_kwargs):
            return {
                "gesture": "keyboard_type",
                "target_hand": "left",
                "confidence": 0.95,
            }

    semantic = SemanticGestureAI(FakeAssistant())
    decision = semantic._classify(_landmarks(), "keyboard_type", "keyboard", "left hand")
    assert decision.gesture == "unknown"
    assert decision.confidence == 0.0
    semantic.close()


def test_virtual_desktop_preserves_negative_monitor_origin():
    from core.display import TrackpadZone, VirtualDesktop, map_to_desktop

    desktop = VirtualDesktop(
        total_width=3840,
        total_height=1080,
        monitor_count=2,
        origin_x=-1920,
        origin_y=0,
    )
    zone = TrackpadZone(100, 1180, 100, 620)
    left = map_to_desktop(1.0, 0.5, zone, desktop)
    right = map_to_desktop(0.0, 0.5, zone, desktop)
    assert left[0] == -1920
    assert right[0] == 1919


def test_ai_left_click_is_authorized_on_release_edge(monkeypatch):
    from core.gestures import right_hand as rh

    class FakeActuator:
        def __init__(self):
            self.clicks = 0
            self.is_dragging = False

        def left_click(self):
            self.clicks += 1

        def double_click(self):
            self.clicks += 2

        def right_click(self):
            pass

        def drag_start(self):
            self.is_dragging = True

        def drag_end(self):
            self.is_dragging = False

        def scroll(self, _ticks):
            pass

        def move(self, _x, _y):
            pass

    class FakeFilter:
        def __call__(self, value, _timestamp=None):
            return value

        def reset(self):
            pass

    actuator = FakeActuator()
    processor = rh.RightHandProcessor(
        actuator,
        type("Desktop", (), {"total_width": 1920, "total_height": 1080, "origin_x": 0, "origin_y": 0})(),
        type("Zone", (), {"x_min": 100, "x_max": 1180, "y_min": 100, "y_max": 620})(),
    )
    processor._filter_x = FakeFilter()
    processor._filter_y = FakeFilter()
    monkeypatch.setattr(rh, "is_fist", lambda _lm: False)
    monkeypatch.setattr(rh, "is_peace_sign", lambda _lm: False)

    phase = {"pinch": True}

    def fake_distance(_lm, a, b):
        if (a, b) == (4, 8):
            return 0.10 if phase["pinch"] else 0.50
        return 0.50

    monkeypatch.setattr(rh, "normalized_distance", fake_distance)

    lms = _landmarks()
    for _ in range(8):
        processor.process(lms, ai_gesture="left_click", ai_required=True)

    assert actuator.clicks == 0

    phase["pinch"] = False
    for _ in range(4):
        processor.process(lms, ai_gesture="left_click", ai_required=True)

    assert actuator.clicks == 1


def test_filter_reset_forgets_previous_sample():
    from core.filter import OneEuroFilter

    filt = OneEuroFilter(freq=60.0)
    first = filt(0.0, 0.0)
    _ = filt(1.0, 1.0 / 60.0)
    filt.reset()
    after_reset = filt(0.75, 2.0)
    assert first == 0.0
    assert after_reset == 0.75
