"""CI-safe tests for the unified keyboard/control integration."""
from unittest.mock import Mock

import core.virtual_keyboard as vk
from core.virtual_keyboard import VirtualKeyboard


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


def test_hands_only_toggle_helper():
    from main import (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        SHOW_CAMERA_UI,
        _keyboard_toggle_pose,
    )
    assert callable(_keyboard_toggle_pose)
    assert CAMERA_WIDTH > 0
    assert CAMERA_HEIGHT > 0
    assert isinstance(SHOW_CAMERA_UI, bool)



def _landmarks_for_pose(extended: set[int]) -> list:
    from core.tracker import Landmark

    lm = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18), (4, 2))
    for tip, pip in finger_pairs:
        if tip == 4:
            lm[tip].x, lm[tip].y = (0.7, 0.30) if tip in extended else (0.52, 0.52)
            lm[pip].x, lm[pip].y = (0.5, 0.45)
        else:
            lm[tip].x, lm[tip].y = (0.5, 0.30) if tip in extended else (0.5, 0.65)
            lm[pip].x, lm[pip].y = (0.5, 0.45)
    lm[9].x, lm[9].y = 0.5, 0.40
    return lm


def test_dual_hand_safe_mode_blocks_left_windows_shortcuts():
    from unittest.mock import Mock
    from config import SAFE_DUAL_HAND_MODE
    from core.display import build_virtual_desktop, build_trackpad_zone
    from core.gestures import GestureOrchestrator
    from core.tracker import HandsResult

    assert SAFE_DUAL_HAND_MODE is True

    actuator = Mock()
    desktop = build_virtual_desktop()
    trackpad = build_trackpad_zone()
    orchestrator = GestureOrchestrator(actuator, desktop, trackpad)

    left_open_palm = _landmarks_for_pose({8, 12, 16, 20, 4})
    right_fist = _landmarks_for_pose(set())

    orchestrator.process(HandsResult(left=left_open_palm, right=right_fist))

    actuator.win_d.assert_not_called()
    actuator.alt_tab.assert_not_called()
    actuator.win_tab.assert_not_called()
    actuator.win_key.assert_not_called()


def test_performance_configuration():
    from config import INFERENCE_FPS, INFERENCE_WIDTH, INFERENCE_HEIGHT
    assert INFERENCE_FPS == 30
    assert (INFERENCE_WIDTH, INFERENCE_HEIGHT) == (640, 360)
