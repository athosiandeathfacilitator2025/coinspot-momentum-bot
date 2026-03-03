# lib/pace_tracker.py
import time
from collections import deque

class PaceTracker:
    def __init__(self, window_seconds=3600):
        self.window = window_seconds
        self.events = deque()

    def record(self):
        self.events.append(time.time())
        self._cleanup()

    def _cleanup(self):
        cutoff = time.time() - self.window
        while self.events and self.events[0] < cutoff:
            self.events.popleft()

    def hourly_rate(self):
        self._cleanup()
        return len(self.events)
