# Conductor

Webcam-driven video conductor. Move your hand up and down to play the
loaded MP4; hold still to pause. Throw the right-angle fist (forearm
vertical, fingers curled) to halt and rewind to the beginning.

## How it works

Each frame:

1. **MediaPipe Hands + Pose** locate the wrist and elbow.
2. **Tempo estimator** tracks the wrist's vertical position, marks each
   direction reversal as a beat, and computes BPM as `60 / mean_interval`
   over a rolling 4-second window.
3. **Gesture classifier** flags a closed fist (fingertips curled toward
   the wrist) and the elbow angle (within ±25° of 90° on the more visible
   arm). Both together = halt.
4. **State machine** drives the player:
   - hand moving → `play()`
   - hand still → `pause()`
   - halt sign → `stop_and_reset()`
5. **HUD** renders status, current/target BPM, the history graph, and
   flavor toasts on top of the mirrored webcam frame. When the current
   BPM strays more than 5 from the target, the centered banner reads
   *NOT QUITE MY TEMPO* in red.

## Setup

VLC must be installed on the system (python-vlc binds to it).

```sh
pip install -r requirements.txt
```

Drop one or more `.mp4` files into `library/`.

## Run

```sh
python -m conductor
```

The terminal will list songs in `library/`, ask which one, and prompt
for a target BPM. Then a window opens with the webcam feed and HUD.
Press `q` or `esc` to quit.

## Layout

| module        | responsibility                              |
| ------------- | ------------------------------------------- |
| `tracker.py`  | MediaPipe Hands + Pose, pixel-space output  |
| `tempo.py`    | rolling peak-detection BPM estimator        |
| `gestures.py` | fist + right-angle arm classifiers          |
| `player.py`   | python-vlc play / pause / stop+rewind       |
| `hud.py`      | OpenCV HUD overlay                          |
| `quotes.py`   | flavor lines indexed by event               |
| `events.py`   | TTL-bound toaster the HUD reads             |
| `main.py`     | terminal picker + main loop + state machine |
