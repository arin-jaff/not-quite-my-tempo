"""Conductor entry point.

Terminal flow:
    1. Pick a song from library/.
    2. Enter target BPM.
    3. Step in front of the webcam and conduct.

Main loop reads a webcam frame, runs MediaPipe, updates the tempo
estimator, decides whether to play / pause / halt the video, and renders
the HUD on top of the mirrored webcam image.
"""
import os
import sys
import time

import cv2

from . import quotes
from .events import EventLog
from .gestures import is_fist, is_halt
from .hud import draw_hud
from .player import VideoPlayer
from .tempo import TempoEstimator
from .tracker import Tracker

LIBRARY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "library",
)


def _list_songs():
    if not os.path.isdir(LIBRARY_DIR):
        print(f"library/ directory not found at {LIBRARY_DIR}", file=sys.stderr)
        sys.exit(1)
    files = sorted(
        f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(".mp4")
    )
    if not files:
        print(f"No MP4 files in {LIBRARY_DIR}. Drop one in and try again.")
        sys.exit(1)
    return files


def pick_song():
    files = _list_songs()
    print("\nLibrary:")
    for i, name in enumerate(files):
        print(f"  [{i + 1}] {name}")
    while True:
        choice = input("Pick a song (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            chosen = files[int(choice) - 1]
            return os.path.join(LIBRARY_DIR, chosen), chosen
        print("Try again.")


def prompt_bpm():
    while True:
        raw = input("Target BPM: ").strip()
        try:
            bpm = float(raw)
        except ValueError:
            print("Enter a number between 20 and 320.")
            continue
        if 20 <= bpm <= 320:
            return bpm
        print("Enter a number between 20 and 320.")


def main():
    song_path, song_name = pick_song()
    target_bpm = prompt_bpm()
    print(f"\nLoaded {song_name} @ {target_bpm} BPM. Step on stage.\n")

    tracker = Tracker()
    tempo = TempoEstimator(target_bpm=target_bpm)
    player = VideoPlayer(song_path)
    events = EventLog()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    started = False           # has the song been kicked off this run?
    halt_active = False       # currently holding the halt sign
    status = "READY"
    last_off_tempo_quote = 0.0
    last_on_tempo_quote = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)
            now = time.time()

            classification = "no hand"
            if res.has_hand:
                _, wy = res.wrist_xy
                tempo.update(now, wy)
                classification = "fist" if is_fist(res.hand_pts) else "open hand"

            halt_now = res.has_hand and is_halt(res.hand_pts, res.pose_pts)
            if halt_now and not halt_active:
                halt_active = True
                player.stop_and_reset()
                started = False
                status = "HALTED"
                events.add(quotes.pick(quotes.HALT))
            elif not halt_now and halt_active:
                halt_active = False  # released; main loop will re-evaluate

            if not halt_active:
                if res.has_hand and tempo.is_moving:
                    if not started:
                        started = True
                        events.add(quotes.pick(quotes.START))
                    player.play()
                    status = "PLAYING"
                else:
                    if started:
                        player.pause()
                        status = "PAUSED"
                    else:
                        status = "READY"

            # off-tempo nag (cooldown 4s)
            if (
                status == "PLAYING"
                and tempo.bpm is not None
                and not tempo.on_tempo
                and now - last_off_tempo_quote > 4.0
            ):
                events.add(quotes.pick(quotes.OFF_TEMPO))
                last_off_tempo_quote = now
            # rare praise when locked in (cooldown 8s)
            if (
                status == "PLAYING"
                and tempo.on_tempo
                and now - last_on_tempo_quote > 8.0
            ):
                events.add(quotes.pick(quotes.ON_TEMPO))
                last_on_tempo_quote = now

            draw_hud(
                frame,
                status=status,
                classification=classification,
                current_bpm=tempo.bpm,
                target_bpm=target_bpm,
                on_tempo=tempo.on_tempo,
                history=tempo.history(),
                peaks=tempo.recent_peaks(),
                events=events.active(),
                song_name=song_name,
            )
            cv2.imshow("Conductor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        player.release()


if __name__ == "__main__":
    main()
