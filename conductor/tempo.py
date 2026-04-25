"""Rolling tempo estimator from vertical hand motion.

Each direction change in the wrist Y signal counts as one beat. BPM is
60 / mean_interval over a short window. We require a minimum amplitude
between direction changes so jitter at rest doesn't spawn fake beats.
"""
from collections import deque

import numpy as np


class TempoEstimator:
    def __init__(
        self,
        target_bpm,
        window_s=4.0,
        min_amplitude_px=18.0,
        stationary_speed_px_s=40.0,
        stationary_window_s=0.5,
        on_tempo_tolerance=5.0,
    ):
        self.target_bpm = float(target_bpm)
        self.window_s = window_s
        self.min_amplitude_px = min_amplitude_px
        self.stationary_speed = stationary_speed_px_s
        self.stationary_window = stationary_window_s
        self.tolerance = on_tempo_tolerance

        self.samples = deque()      # (t, y)
        self.peaks = deque()        # t of confirmed direction changes
        self._last_dir = 0          # +1 down, -1 up (image y grows downward)
        self._last_extreme_y = None

    def update(self, t, y):
        self.samples.append((t, y))
        cutoff = t - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.peaks and self.peaks[0] < cutoff:
            self.peaks.popleft()

        if len(self.samples) < 5:
            return

        # smoothed slope from last 5 samples
        recent_y = np.array([s[1] for s in list(self.samples)[-5:]])
        slope = float(recent_y[-1] - recent_y[0])
        if abs(slope) < 1.0:
            return
        d = 1 if slope > 0 else -1
        if self._last_dir == 0:
            self._last_dir = d
            self._last_extreme_y = y
            return
        if d != self._last_dir:
            # direction reversed — confirm via amplitude since last extreme
            if (
                self._last_extreme_y is not None
                and abs(y - self._last_extreme_y) >= self.min_amplitude_px
            ):
                self.peaks.append(t)
                self._last_extreme_y = y
            self._last_dir = d

    @property
    def bpm(self):
        if len(self.peaks) < 3:
            return None
        intervals = np.diff(np.array(self.peaks))
        mean = float(np.mean(intervals))
        if mean <= 0:
            return None
        return 60.0 / mean

    @property
    def on_tempo(self):
        b = self.bpm
        return b is not None and abs(b - self.target_bpm) <= self.tolerance

    @property
    def is_moving(self):
        if len(self.samples) < 2:
            return False
        t_now = self.samples[-1][0]
        cutoff = t_now - self.stationary_window
        recent = [(t, y) for t, y in self.samples if t >= cutoff]
        if len(recent) < 2:
            return False
        ys = np.array([y for _, y in recent])
        ts = np.array([t for t, _ in recent])
        span = float(ts[-1] - ts[0])
        if span <= 0:
            return False
        amp = float(ys.max() - ys.min())
        return (amp / span) >= self.stationary_speed

    def history(self):
        return list(self.samples)

    def recent_peaks(self):
        return list(self.peaks)
