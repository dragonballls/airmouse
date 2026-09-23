# AI gesture control

The air-mouse combines local MediaPipe tracking with two AI layers:

1. Jev is the fast primary semantic decision layer. It consumes compact,
   structured hand-tracker state and returns a typed action choice.
2. Gemini is the visual fallback for ambiguous or low-confidence candidates and
   can inspect the actual camera image.

Pointer tracking stays local for responsiveness. AI only authorizes bounded,
predefined actions.

## Configure Jev

The official TypeSafe Python SDK is included in requirements.txt.

Set a TypeSafe API key as a Windows user environment variable without printing
it:

    [Environment]::SetEnvironmentVariable("TYPESAFE_API_KEY","YOUR_KEY","User")

The official SDK reads TYPESAFE_API_KEY and defaults to jev-latest. Do not commit
the key to the repository.

Jev direct access may require a TypeSafe account. The application remains
functional with Gemini alone when a TypeSafe key is not configured.

## Configure Gemini

Set the Google key as a Windows user environment variable without printing it:

    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY","YOUR_KEY","User")

Default model:

    gemini-3.5-flash-lite

Override with GEMINI_MODEL when needed.

## Live semantic behavior

When Jev is available, it is asked to select exactly one action from:

- none
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

The state includes current mode, local hand ownership, deliberate candidate,
hold duration, and temporal two-hand motion. Jev returns a typed choice plus
probabilities and confidence.

When Jev is unavailable or uncertain, Gemini receives the compressed camera
frame and performs structured visual classification. Unknown or low-confidence
decisions are rejected.

The action executor remains hard-coded and closed-set. AI cannot supply a
Windows command, application name, key sequence, or arbitrary function call.

## Gestures

### Mouse

Right index finger controls the pointer locally.

Thumb + index:
- short intentional pinch -> AI-authorized left click
- sustained intentional pinch -> AI-authorized drag

Thumb + middle:
- intentional pinch -> AI-authorized right click

Peace sign + wrist movement:
- AI-authorized scroll direction

Fist:
- local safety lock

### Two hands

Both hands holding thumb-index pinches create the protected zoom candidate.
Temporal wrist separation is supplied to Jev so it can distinguish zoom in
(apart) from zoom out (together).

### Keyboard

Raise index + middle + ring on the left hand and hold for about one second.
Jev or Gemini must recognize the left-hand keyboard-toggle sign before the
keyboard opens.

Only the right hand types. A right-hand thumb-index pinch over a stabilized key
requires AI authorization before a key event is sent.

K remains available as the explicit manual keyboard toggle.
A requests a one-shot visual diagnostic.
Q or Esc quits.

## Reliability

MediaPipe handles continuous low-latency tracking. Jev handles fast semantic
choice. Gemini handles visual ambiguity. Stable-frame, release, cooldown,
freshness, hand-ownership, and closed-set action checks remain in code.
