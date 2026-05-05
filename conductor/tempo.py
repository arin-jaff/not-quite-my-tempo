"""Rolling tempo estimator from oscillating hand motion.

We track a "motion → stop → motion → stop" cycle. Each transition from
moving to stopped marks one beat (the moment the conductor's hand
arrives at a downbeat or upbeat, regardless of direction). BPM is
60 / mean_interval between recent stops. Hysteresis between move/stop
thresholds keeps jitter from spawning fake beats.
"""
from collections import deque

import numpy as np


class TempoEstimator:
    def __init__(
        self,
        target_bpm,
        window_s=4.0,
        move_threshold_px_s=240.0,
        stop_threshold_px_s=80.0,
        speed_window_s=0.15,
        min_beat_gap_s=0.18,
        stationary_speed_px_s=40.0,
        stationary_window_s=0.5,
        on_tempo_tolerance=5.0,
    ):
        self.target_bpm = float(target_bpm)
        self.window_s = window_s
        self.move_threshold = move_threshold_px_s
        self.stop_threshold = stop_threshold_px_s
        self.speed_window = speed_window_s
        self.min_beat_gap = min_beat_gap_s
        self.stationary_speed = stationary_speed_px_s
        self.stationary_window = stationary_window_s
        self.tolerance = on_tempo_tolerance

        self.samples = deque()      # (t, y)
        self.peaks = deque()        # t of confirmed motion→stop transitions
        self._moving = False        # current hysteresis state

    def update(self, t, y):
        self.samples.append((t, y))
        cutoff = t - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.peaks and self.peaks[0] < cutoff:
            self.peaks.popleft()

        # Speed over the last speed_window seconds (peak-to-peak / span).
        v_cutoff = t - self.speed_window
        recent = [(ts, ys) for ts, ys in self.samples if ts >= v_cutoff]
        if len(recent) < 3:
            return
        ys = np.array([yy for _, yy in recent])
        ts = np.array([tt for tt, _ in recent])
        span = float(ts[-1] - ts[0])
        if span <= 0:
            return
        speed = float(ys.max() - ys.min()) / span

        if not self._moving:
            if speed >= self.move_threshold:
                self._moving = True
        else:
            if speed <= self.stop_threshold:
                if not self.peaks or (t - self.peaks[-1]) >= self.min_beat_gap:
                    self.peaks.append(t)
                self._moving = False

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
