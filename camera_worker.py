"""Camera capture + hand tracking on a background thread.

Why: MediaPipe takes most of each frame's time (often 20-40 ms on a laptop
CPU). If the main loop waited for it, the game window could only redraw as
fast as the camera is processed and would stutter. With this worker the
main loop picks up the newest result whenever one is ready and keeps
drawing smoothly in between.
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from hand_tracker import HandTracker, TwoHands


@dataclass
class TrackedFrame:
    frame: np.ndarray              # BGR, already mirrored if configured
    hands: Optional[TwoHands]      # None unless exactly two hands were found
    t_capture: float               # perf_counter() right after the camera read


class CameraWorker:
    """Reads frames and finds hands in the background; always keeps only the newest."""

    def __init__(self, cap: cv2.VideoCapture, tracker: HandTracker,
                 mirror: bool, max_read_failures: int) -> None:
        self.cap = cap
        self.tracker = tracker
        self.mirror = mirror
        self.max_read_failures = max_read_failures
        self.failed = False  # set if the camera stopped sending frames
        self._latest: Optional[TrackedFrame] = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish and wait for it (the camera read can take a moment)."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def get(self, timeout: float) -> Optional[TrackedFrame]:
        """Newest result not seen yet, or None if nothing new arrived within `timeout`."""
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            return self._latest

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                # some cameras return a few empty frames at startup (DirectShow)
                failures += 1
                if failures >= self.max_read_failures:
                    self.failed = True
                    self._new.set()  # wake the main loop so it can notice
                    return
                continue
            failures = 0
            t = time.perf_counter()
            if self.mirror:
                frame = cv2.flip(frame, 1)  # mirror view feels natural
            hands = self.tracker.find_two_hands(frame)
            with self._lock:
                # older unread results are simply replaced: only the newest matters
                self._latest = TrackedFrame(frame, hands, t)
                self._new.set()
