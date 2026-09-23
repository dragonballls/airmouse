# Unified Airmouse configuration.
# Tuned for responsive two-hand control on a typical Windows laptop webcam.
CAMERA_INDEX=0
CAMERA_WIDTH=960
CAMERA_HEIGHT=540
CAMERA_FPS=60
CAMERA_BUFFER_SIZE=1

MP_MODEL_PATH="models/hand_landmarker.task"
MP_MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MP_MAX_HANDS=2
MP_DETECTION_CONFIDENCE=0.60
MP_TRACKING_CONFIDENCE=0.50

# Larger usable gesture area without forcing the hand to camera edges.
TRACKPAD_MARGIN=55
ONE_EURO_MINCUTOFF=0.90
ONE_EURO_BETA=0.020
ONE_EURO_DCUTOFF=1.0

# Short confirmation / release windows keep pinches deliberate without
# requiring the user to hold their hand unnaturally still.
GESTURE_STABILITY_FRAMES=4
GESTURE_RELEASE_FRAMES=2
GESTURE_COOLDOWN_SECONDS=0.30

# AirNav-style easier pinch interaction with release hysteresis.
THUMB_INDEX_CLICK_DIST=0.28
THUMB_MIDDLE_CLICK_DIST=0.28
PINCH_RELEASE_DIST=0.40
DRAG_HOLD_SECONDS=0.35
DRAG_MOVE_THRESHOLD=0.018
DOUBLE_CLICK_WINDOW_S=0.45

WRIST_VELOCITY_THRESHOLD=0.010
SCROLL_TICK_SCALE=0.0035
SCROLL_COOLDOWN_S=0.18

ZOOM_WRIST_DELTA=0.08
ZOOM_STABILITY_FRAMES=6
ZOOM_COOLDOWN_S=0.75

# Keyboard activation is deliberately separate from mouse gestures.
KEYBOARD_TOGGLE_HOLD_S=0.65
KEYBOARD_TOGGLE_COOLDOWN_S=1.2

# AI semantic classification is optional and disabled by default in the
# real-time input path so gesture latency never depends on network/model time.
AI_GESTURE_MIN_CONFIDENCE=0.68
AI_GESTURE_MAX_AGE_S=0.75
AI_GESTURE_GRACE_S=0.95
AI_REQUEST_INTERVAL_S=0.25
AI_VISION_MAX_WIDTH=640

ENABLE_LEFT_SYSTEM_SHORTCUTS=False

FLIP_HANDEDNESS=True
DEBUG_GESTURES=False

# Do not show the live camera feed. Keyboard mode creates its own non-video HUD.
SHOW_CAMERA_UI=False

# Contextual typing keyboard. It appears only after the pointer dwells over a
# Windows UI Automation text field, then stays anchored near that cursor.
TEXT_INPUT_AUTO_KEYBOARD=True
TEXT_INPUT_DWELL_SECONDS=0.65
TEXT_INPUT_POLL_SECONDS=0.20
TYPING_KEYBOARD_WIDTH=720
TYPING_KEYBOARD_HEIGHT=320
TYPING_KEYBOARD_MARGIN=16
TYPING_KEYBOARD_BOTTOM_MARGIN=18

TIMER_RESOLUTION_MS=1
PROCESS_PRIORITY=0x80
