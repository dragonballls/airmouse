"""CI-safe tests for the unified keyboard/control integration."""
from unittest.mock import Mock
from core.virtual_keyboard import VirtualKeyboard

def test_keyboard_contains_practical_keys():
    keyboard=VirtualKeyboard(1280,720)
    labels={key.label for key in keyboard.keys}
    required={"Esc","F1","F12","Backspace","Tab","Caps","Enter","Shift","Space","Q","A","Z"}
    assert required <= labels
    keyboard.close()

def test_keyboard_sends_real_key_events():
    keyboard=VirtualKeyboard(1280,720)
    keyboard.controller=Mock()
    q=next(key for key in keyboard.keys if key.label=="Q")
    space=next(key for key in keyboard.keys if key.label=="SPACE")
    keyboard.press_key(q)
    keyboard.press_key(space)
    assert keyboard.controller.press.called
    assert keyboard.controller.release.called
    keyboard.close()

def test_hands_only_toggle_helper():
    from main import _keyboard_toggle_pose
    assert callable(_keyboard_toggle_pose)
