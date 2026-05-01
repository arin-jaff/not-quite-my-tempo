"""Toaster for HUD-displayed event messages.

Each entry has a wall-clock timestamp; active() returns those still
within their TTL. The HUD fades them out over time.
"""
import time
from collections import deque


class EventLog:
    def __init__(self, ttl=3.5, capacity=8):
        self.ttl = ttl
        self._items = deque(maxlen=capacity)

    def add(self, msg):
        if not msg:
            return
        self._items.append((time.time(), msg))

    def active(self):
        now = time.time()
        return [(t, m) for t, m in self._items if now - t <= self.ttl]
