# Conductor

Webcam-driven media conductor. Conduct an oscillating motion — move,
stop, move, stop — to set the tempo on the loaded track (MP4 or MP3).
Throw the right-angle fist (forearm vertical, fingers curled) to halt
and rewind to the beginning. Once started, only the fist halts; brief
stillness or a momentarily lost hand will not pause playback.

## How it works

Each frame:

1. **MediaPipe Hands + Pose** locate the wrist and elbow. If the hand
   model loses lock, the tempo estimator falls back to the most-visible
   pose-wrist so playback rides out the gap on the previous tempo.
2. **Tempo estimator** tracks vertical wrist position, watches the
   speed signal, and marks each *motion → stop* transition as a beat.
   BPM is `60 / mean_interval` between recent stops over a rolling
   4-second window. Hysteresis between the move/stop thresholds keeps
   jitter from spawning fake beats.
3. **Gesture classifier** flags a closed fist (fingertips curled toward
   the wrist) and the elbow angle (within ±25° of 90° on the more visible
   arm). Both together = halt.
4. **State machine** drives the player:
   - first detected motion → `play()`
   - halt sign → `stop_and_reset()`
   - released halt → ready for next start
   Brief stillness or a lost hand do **not** pause; only the fist halts.
5. **HUD** renders status, current/target BPM, the history graph, and
   flavor toasts on top of the mirrored webcam frame. The detected hand
   skeleton and active arm (shoulder→elbow→wrist with elbow-angle
   readout) are drawn live so you can see the classifier's view —
   bones turn green on a fist, red on the halt sign. When the current
   BPM strays more than 5 from the target, the centered banner reads
   *NOT QUITE MY TEMPO* in red.
6. **Tempo-driven playback rate**: while playing, libVLC's playback
   rate is set to `current_bpm / target_bpm`, with extreme bounds
   (~0.05×–16×) so wildly off-tempo conducting actually warps the
   track. VLC scales pitch with speed, so dragging the beat slows the
   track *and* pitches it down; rushing speeds it up *and* pitches it
   up.

## Setup

VLC must be installed on the system (python-vlc binds to it). On macOS:

```sh
brew install --cask vlc
```

Or download VLC from [videolan.org](https://www.videolan.org/) and drop
it into `/Applications`. Then install Python deps:

```sh
pip install -r requirements.txt
```

Drop one or more `.mp4` or `.mp3` files into `library/`. MP3s play
audio-only; the webcam HUD is still the visible window.

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
