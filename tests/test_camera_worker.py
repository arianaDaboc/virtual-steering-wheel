import sys
import threading
import time
import types

import numpy as np

# hand_tracker imports mediapipe at module level; the worker only needs the
# tracker object we pass in, so a stub module is enough for these tests
sys.modules.setdefault("mediapipe", types.SimpleNamespace(
    solutions=types.SimpleNamespace(hands=None)))

from camera_worker import CameraWorker  # noqa: E402


class FakeCap:
    def __init__(self, frames_ok=True, delay=0.005):
        self.frames_ok, self.delay, self.count = frames_ok, delay, 0

    def read(self):
        time.sleep(self.delay)
        self.count += 1
        if not self.frames_ok:
            return False, None
        return True, np.full((4, 4, 3), self.count % 255, np.uint8)


class FakeTracker:
    def find_two_hands(self, frame):
        return None


def test_worker_delivers_newest_frames():
    w = CameraWorker(FakeCap(), FakeTracker(), mirror=False, max_read_failures=3)
    w.start()
    try:
        first = w.get(timeout=1.0)
        time.sleep(0.05)              # several frames arrive meanwhile
        newest = w.get(timeout=1.0)
        assert first is not None and newest is not None
        assert newest.t_capture > first.t_capture
        assert w.get(timeout=0.0) is None  # nothing new since the last get
    finally:
        w.stop()


def test_worker_reports_a_dead_camera():
    w = CameraWorker(FakeCap(frames_ok=False, delay=0.0), FakeTracker(),
                     mirror=False, max_read_failures=3)
    w.start()
    w.get(timeout=1.0)
    assert w.failed
    w.stop()


def test_worker_stops():
    w = CameraWorker(FakeCap(), FakeTracker(), mirror=True, max_read_failures=3)
    w.start()
    w.stop()
    assert not any(t.name == "camera" and t.is_alive() for t in threading.enumerate())
