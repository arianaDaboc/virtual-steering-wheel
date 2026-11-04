"""Runtime metrics. For now: camera FPS and capture-to-output latency."""
import time
from typing import Optional


class FrameTimer:
    """Call `start()` when a frame arrives and `stop()` when it is handled.

    Pass the capture time to `start()` and `latency_ms` becomes the real
    delay from the camera read to the controller update, hand tracking included.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        self.smoothing = smoothing  # weight of the newest frame in the FPS average
        self.fps = 0.0
        self.latency_ms = 0.0
        self._t0 = time.perf_counter()
        self._last = self._t0

    def start(self, t_capture: Optional[float] = None) -> float:
        """Mark when this frame's work began (default: now) and return it, in seconds."""
        self._t0 = time.perf_counter() if t_capture is None else t_capture
        return self._t0

    def stop(self) -> None:
        """Mark the end of processing and update `fps` and `latency_ms`."""
        now = time.perf_counter()
        self.latency_ms = (now - self._t0) * 1000
        instant_fps = 1.0 / max(now - self._last, 1e-6)
        # exponential moving average keeps the number readable on screen
        self.fps += self.smoothing * (instant_fps - self.fps)
        self._last = now
