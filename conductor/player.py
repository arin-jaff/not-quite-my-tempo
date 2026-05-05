"""Tiny wrapper around python-vlc for the conductor.

Plays anything libVLC can decode — MP4 video and MP3 audio are both fine.

Three operations matter:
    play()             — start (or resume) playback
    pause()            — freeze where we are
    stop_and_reset()   — halt; the next play() restarts from t=0

We track our own state because vlc.MediaPlayer state transitions are
asynchronous and we want O(1) decisions inside the main loop.
"""
import vlc


class VideoPlayer:
    def __init__(self, path):
        self.path = path
        self._instance = vlc.Instance("--quiet")
        self._media = self._instance.media_new(path)
        self._player = self._instance.media_player_new()
        self._player.set_media(self._media)
        self._state = "stopped"

    def play(self):
        if self._state == "playing":
            return
        if self._state == "paused":
            self._player.set_pause(0)
        else:
            self._player.play()
        self._state = "playing"

    def pause(self):
        if self._state != "playing":
            return
        self._player.set_pause(1)
        self._state = "paused"

    def stop_and_reset(self):
        self._player.stop()
        # vlc.stop() rewinds to 0; the next play() begins fresh
        self._state = "stopped"

    @property
    def state(self):
        return self._state

    @property
    def position_s(self):
        ms = self._player.get_time()
        return ms / 1000.0 if ms is not None and ms >= 0 else 0.0

    def release(self):
        self._player.stop()
        self._player.release()
        self._instance.release()
