# Optional AI vision assistant

The air-mouse performs latency-sensitive hand tracking locally. The AI API is called only when you explicitly press A in the camera window.

## Configure

Google Gemini is supported directly. Google documents GEMINI_API_KEY or GOOGLE_API_KEY as environment-variable configuration for the Gemini SDK.

On Windows PowerShell, store the key as a user environment variable without printing it:

    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY","YOUR_KEY","User")

Then open a new PowerShell session.

Default Gemini model:

    gemini-2.5-flash-lite

Override it when needed:

    [Environment]::SetEnvironmentVariable("GEMINI_MODEL","YOUR_MODEL","User")

OpenAI remains supported as a fallback:

    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY","YOUR_KEY","User")
    [Environment]::SetEnvironmentVariable("OPENAI_MODEL","gpt-5.6","User")

Never commit an API key to the repository.

## What the AI does

A single compressed JPEG camera frame is sent for an explicit diagnostic request. The assistant describes the apparent hand pose and can suggest a calibration change.

The AI response is advisory only. It cannot directly issue mouse clicks, keyboard events, window switches, or lock the computer.

## Controls

### Mouse mode

- Right index finger: local cursor
- Stable thumb + index pinch, then release: left click
- Stable thumb + middle pinch, then release: right click
- Deliberate pinch hold: drag
- Stable peace pose + wrist movement: scroll
- Fist: pause right-hand mouse actions

### Two hands

Both hands holding a deliberate thumb/index pinch enters a protected zoom gesture. Moving the wrists apart or together far enough produces one zoom event. The gesture must be released before another zoom can fire.

### Keyboard

Raise three fingers (index, middle, ring) on the left hand and hold for about one second. The pose must become stable first, and the hand must be released before another toggle can occur.

Once visible, only the right hand types:

- hover an on-screen key with the index finger
- pinch thumb + index to type
- move while pinching to glide between keys

K toggles the keyboard manually.

A requests one explicit AI diagnostic. Q or Esc quits.
