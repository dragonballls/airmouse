# AI gesture control

The air-mouse keeps latency-sensitive MediaPipe hand tracking local for cursor
movement, while Gemini provides semantic classification for deliberate gesture
candidates.

## Configure

Google Gemini is supported directly. Set the key as a Windows user variable
without printing it:

    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY","YOUR_KEY","User")

Default live gesture model:

    gemini-3.5-flash-lite

Override it with GEMINI_MODEL when needed.

OpenAI remains available for the explicit A-key diagnostic fallback:

    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY","YOUR_KEY","User")
    [Environment]::SetEnvironmentVariable("OPENAI_MODEL","gpt-5.6","User")

Never commit an API key to the repository.

## Live AI gesture layer

When Gemini is configured, deliberate actions use a closed semantic label set:

- left_click
- right_click
- drag
- scroll_up
- scroll_down
- zoom_in
- zoom_out
- keyboard_toggle
- keyboard_type
- unknown

The camera frame and a small local-tracker hint are sent asynchronously to
Gemini only for deliberate gesture candidates, rather than on every camera
frame. The classifier returns structured JSON containing the gesture,
target hand, and confidence.

The AI does not receive OS-control tools or arbitrary commands. The application
accepts only the predefined labels and also checks the expected hand before
executing an action. Low-confidence, stale, unknown, or wrong-hand decisions
are rejected.

Pointer movement remains local so network latency does not make the cursor
jerky. Click/drag/scroll/zoom and keyboard actions are AI-authorized when the
Gemini gate is enabled.

## Controls

### Mouse mode

- Right index finger: point/move
- Thumb + index pinch: AI distinguishes left click versus sustained drag
- Thumb + middle pinch: AI-authorized right click
- Peace sign + wrist movement: AI determines scroll direction
- Fist: local safety lock

### Two hands

Both hands holding a deliberate thumb-index pinch create a protected zoom
candidate. Gemini classifies zoom direction and the local two-hand state gate
requires both hands.

### Keyboard

Raise three fingers (index, middle, ring) on the left hand and hold for about
one second. Gemini must identify the left-hand keyboard-toggle sign.

Once visible, only the right hand types. Gemini authorizes the right-hand
thumb-index typing sign before a key is sent.

K manually toggles the keyboard for emergency/manual control.
A requests one explicit visual diagnostic.
Q or Esc quits.

## Reliability model

MediaPipe is still the continuous tracker. Gemini is the semantic layer: it
helps distinguish the intended sign/action and the performing hand, while
local stable-frame, release, cooldown, and action-boundary checks remain in
place.
