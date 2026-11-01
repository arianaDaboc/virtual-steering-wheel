"""Runtime metrics. For now: FPS and per-frame processing time."""
import time


class FrameTimer:
    """Call `start()` when a frame arrives and `stop()` when it is handled."""

    def __init__(self, smoothing: float = 0.1) -> None:
        self.smoothing = smoothing  # weight of the newest frame in the FPS average
        self.fps = 0.0
        self.proc_ms = 0.0
        self._t0 = time.perf_counter()
        self._last = self._t0

    def start(self) -> float:
        """Mark the start of processing and return the timestamp (seconds)."""
        self._t0 = time.perf_counter()
        return self._t0

    def stop(self) -> None:
        """Mark the end of processing and update `fps` and `proc_ms`."""
        now = time.perf_counter()
        self.proc_ms = (now - self._t0) * 1000
        instant_fps = 1.0 / max(now - self._last, 1e-6)
        # exponential moving average keeps the number readable on screen
        self.fps += self.smoothing * (instant_fps - self.fps)
        self._last = now
