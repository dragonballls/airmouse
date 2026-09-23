# Optional AI vision assistant

The air-mouse still performs latency-sensitive hand tracking locally. The API is only called when you explicitly press A in the camera window.

## Configure

On Windows PowerShell:

    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_KEY", "User")
    [Environment]::SetEnvironmentVariable("OPENAI_MODEL", "gpt-5.6", "User")

Restart PowerShell after setting the variable.

An OpenAI-compatible gateway can also be selected without changing the code:

    [Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", "https://your-compatible-endpoint/v1", "User")

Never commit the API key to the repository.

## What the AI does

A single JPEG frame is sent for an explicit diagnostic request. The assistant describes the apparent hand pose and can suggest a calibration change.

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
