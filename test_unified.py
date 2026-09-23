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
