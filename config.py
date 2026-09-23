# Unified Airmouse configuration.
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=60
CAMERA_BUFFER_SIZE=1

MP_MODEL_PATH="models/hand_landmarker.task"
MP_MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MP_MAX_HANDS=2
MP_DETECTION_CONFIDENCE=0.7
MP_TRACKING_CONFIDENCE=0.5

TRACKPAD_MARGIN=100
ONE_EURO_MINCUTOFF=0.7
ONE_EURO_BETA=0.012
ONE_EURO_DCUTOFF=1.0

# Gesture safety is intentionally conservative.
GESTURE_STABILITY_FRAMES=8
GESTURE_RELEASE_FRAMES=4
GESTURE_COOLDOWN_SECONDS=0.65

THUMB_INDEX_CLICK_DIST=0.24
THUMB_MIDDLE_CLICK_DIST=0.24
PINCH_RELEASE_DIST=0.36
DRAG_HOLD_SECONDS=0.55
DOUBLE_CLICK_WINDOW_S=0.45

WRIST_VELOCITY_THRESHOLD=0.010
SCROLL_TICK_SCALE=0.0035
SCROLL_COOLDOWN_S=0.18

ZOOM_WRIST_DELTA=0.08
ZOOM_STABILITY_FRAMES=6
ZOOM_COOLDOWN_S=0.75

# Keyboard activation is deliberately separate from mouse gestures.
KEYBOARD_TOGGLE_HOLD_S=1.0
KEYBOARD_TOGGLE_COOLDOWN_S=1.5

# AI semantic gate. When Gemini is configured, deliberate actions require a
# matching high-confidence semantic label. Pointer movement remains local.
AI_GESTURE_MIN_CONFIDENCE=0.68
AI_GESTURE_MAX_AGE_S=0.75
AI_GESTURE_GRACE_S=0.95
AI_REQUEST_INTERVAL_S=0.25
AI_VISION_MAX_WIDTH=640

# The old left-hand Windows shortcuts are disabled because they were too easy
# to trigger accidentally. Windows shortcuts can be added later behind explicit
# arming/confirmation.
ENABLE_LEFT_SYSTEM_SHORTCUTS=False

FLIP_HANDEDNESS=True  # Camera frames are unmirrored before display mirroring
DEBUG_GESTURES=False
SHOW_CAMERA_UI=True
TIMER_RESOLUTION_MS=1
PROCESS_PRIORITY=0x80
