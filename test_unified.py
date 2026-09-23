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


def test_ai_assistant_is_optional_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from core.ai_assist import AIAssistant
    assistant = AIAssistant()
    assert assistant.enabled is False
