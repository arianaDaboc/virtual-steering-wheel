"""Guided calibration: measure the straight-ahead angle and the full-lock angles.

`CalibrationSession` is a small state machine that main.py feeds one frame at
a time. It never touches the camera, so it can be tested with fake angles.
"""
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Calibration:
    """Result of a calibration, in raw hand-line degrees."""
    neutral_deg: float
    left_span_deg: Optional[float] = None    # neutral -> full left lock (positive)
    right_span_deg: Optional[float] = None   # neutral -> full right lock (positive)


@dataclass(frozen=True)
class Step:
    key: str
    instruction: str


CENTER = Step("center", "Hold the wheel STRAIGHT")
LEFT = Step("left", "Turn fully LEFT and hold")
RIGHT = Step("right", "Turn fully RIGHT and hold")


class CalibrationSession:
    """Walks the user through center (and optionally left/right) steps.

    Each step needs both hands held still for `hold_seconds`. If the hands
    disappear or clearly move, the step starts over. A single far-off frame
    is treated as a detection glitch and dropped; `motion_frames` far-off
    frames in a row mean the user is really moving (e.g. turning into the
    next pose), so the step restarts from there.
    """

    def __init__(self, hold_seconds: float, include_range: bool, min_span_deg: float,
                 max_spread_deg: float, max_jump_deg: float, motion_frames: int) -> None:
        self.steps: List[Step] = [CENTER, LEFT, RIGHT] if include_range else [CENTER]
        self.hold_seconds = hold_seconds
        self.min_span_deg = min_span_deg
        self.max_spread_deg = max_spread_deg
        self.max_jump_deg = max_jump_deg
        self.motion_frames = motion_frames
        self.index = 0
        self.aborted = False
        self.message = ""                     # why the last attempt was rejected
        self.results: Dict[str, float] = {}
        self._samples: List[float] = []
        self._outliers: List[Tuple[float, float]] = []  # (angle, t) far from the median
        self._t_start: Optional[float] = None
        self._t_last: Optional[float] = None

    # ---- state for the overlay ----
    @property
    def done(self) -> bool:
        return self.aborted or self.index >= len(self.steps)

    @property
    def step(self) -> Step:
        return self.steps[min(self.index, len(self.steps) - 1)]

    @property
    def progress(self) -> float:
        """How much of the current step is held, from 0 to 1."""
        if self._t_start is None or self._t_last is None:
            return 0.0
        return min((self._t_last - self._t_start) / self.hold_seconds, 1.0)

    # ---- driving the session ----
    def update(self, angle: Optional[float], t: float) -> None:
        """Feed the raw hand angle for this frame (None = hands not visible)."""
        if self.done:
            return
        if angle is None:
            self._restart_step()
            return
        if self._samples and abs(angle - statistics.median(self._samples)) > self.max_jump_deg:
            self._outliers.append((angle, t))
            if len(self._outliers) < self.motion_frames:
                return  # probably a glitch: skip this frame
            # real movement: start the step over from the new position
            moved = self._outliers
            self._restart_step()
            for a, ta in moved:
                self._add(a, ta)
        else:
            self._outliers = []
            self._add(angle, t)
        if t - self._t_start >= self.hold_seconds:  # type: ignore[operator]
            self._finish_step()

    def _add(self, angle: float, t: float) -> None:
        if self._t_start is None:
            self._t_start = t
        self._t_last = t
        self._samples.append(angle)

    def skip(self) -> None:
        """Skip the rest. Before the center step is measured, this cancels everything."""
        if self.index == 0:
            self.aborted = True
        else:
            self.index = len(self.steps)

    def result(self) -> Optional[Calibration]:
        """The measured calibration, or None if the center step never finished."""
        if self.aborted or "center" not in self.results:
            return None
        neutral = self.results["center"]
        left = neutral - self.results["left"] if "left" in self.results else None
        right = self.results["right"] - neutral if "right" in self.results else None
        return Calibration(neutral, left, right)

    # ---- internals ----
    def _restart_step(self) -> None:
        self._samples = []
        self._outliers = []
        self._t_start = None
        self._t_last = None

    def _finish_step(self) -> None:
        # final check against slow drift, which the per-frame jump test misses
        if statistics.pstdev(self._samples) > self.max_spread_deg:
            self.message = "Hold still"
            self._restart_step()
            return
        # median instead of mean: one glitchy detection can't drag it away
        value = statistics.median(self._samples)
        key = self.step.key
        if key != "center":
            neutral = self.results["center"]
            span = neutral - value if key == "left" else value - neutral
            if span < self.min_span_deg:
                self.message = f"Turn further {key} ({span:.0f} of {self.min_span_deg:.0f} deg)"
                self._restart_step()
                return
        self.results[key] = value
        self.message = ""
        self.index += 1
        self._restart_step()


def save_calibration(cal: Calibration, path: Path) -> None:
    """Write the calibration as JSON so the next run can skip the guided screen."""
    data = {k: None if v is None else round(v, 2) for k, v in asdict(cal).items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _optional_float(value: object) -> Optional[float]:
    return None if value is None else float(value)  # type: ignore[arg-type]


def load_calibration(path: Path) -> Optional[Calibration]:
    """Read a saved calibration; None if the file is missing or broken."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(float(data["neutral_deg"]),
                           _optional_float(data.get("left_span_deg")),
                           _optional_float(data.get("right_span_deg")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
