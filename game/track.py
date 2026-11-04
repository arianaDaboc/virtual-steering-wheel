"""Circuit geometry: a closed centerline with hills, cut into equal segments.

The layout is drawn in 2D (meters, x right, y down, so a clockwise turn is a
right turn) and then converted to what the 3D view and the physics need:
curvature and elevation per segment, plus roadside scenery.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

# Control points of the circuit in meters, driven in this order. This is the
# track *layout* (data), not a tuning value, so it lives here and not in config.
CIRCUIT: List[Point] = [
    (0, 0), (260, 0), (430, 15), (510, 80), (515, 170), (450, 235), (340, 250),
    (265, 300), (245, 395), (170, 455), (45, 445), (-35, 385), (-60, 300),
    (-140, 250), (-235, 240), (-295, 180), (-285, 90), (-205, 28), (-105, 0),
]


def catmull_rom_closed(pts: Sequence[Point], samples: int) -> List[Point]:
    """Smooth closed curve through every control point."""
    out: List[Point] = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        for k in range(samples):
            t = k / samples
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[j] + (p2[j] - p0[j]) * t
                                    + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (3 * p1[j] - p0[j] - 3 * p2[j] + p3[j]) * t3)
                             for j in (0, 1)))  # type: ignore[misc]
    return out


def resample_closed(poly: Sequence[Point], n: int) -> List[Point]:
    """`n` points spaced evenly along the closed polyline."""
    cum = [0.0]
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cum.append(cum[-1] + math.dist(a, b))
    step, out, j = cum[-1] / n, [], 0
    for k in range(n):
        target = k * step
        while cum[j + 1] < target:
            j += 1
        a, b = poly[j], poly[(j + 1) % len(poly)]
        u = (target - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    return out


def polyline_length(poly: Sequence[Point]) -> float:
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def wrap_angle(a: float) -> float:
    """Angle in [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Segment:
    curve: float                     # curvature 1/m, positive = turning right
    y: float                         # elevation at the start of the segment, m
    sprites: List[Tuple[float, str]] = field(default_factory=list)  # (offset m, kind)


class Track:
    """Closed circuit made of equal-length segments."""

    def __init__(self, cfg: Dict[str, Any], layout: Sequence[Point] = CIRCUIT) -> None:
        self.road_half = cfg["road_width"] / 2
        self.rumble = cfg["rumble_width"]
        self.runoff = cfg["runoff"]
        smooth_line = catmull_rom_closed(layout, cfg["spline_samples"])
        n = max(8, round(polyline_length(smooth_line) / cfg["segment_length"]))
        self.points = resample_closed(smooth_line, n)  # 2D centerline, for the minimap
        self.length = polyline_length(self.points)
        self.segment_length = self.length / n
        self.headings = [math.atan2(b[1] - a[1], b[0] - a[0])
                         for a, b in zip(self.points, self.points[1:] + self.points[:1])]
        raw = [wrap_angle(self.headings[(i + 1) % n] - self.headings[i]) / self.segment_length
               for i in range(n)]
        # moving average: the spline's curvature is a bit bumpy, and bumps
        # would feel like the road twitching. The average keeps the total turn.
        w = cfg["curve_smoothing"]
        curves = [sum(raw[(i + k) % n] for k in range(-w, w + 1)) / (2 * w + 1) for i in range(n)]
        self.segments = [Segment(c, self._elevation(i * self.segment_length, cfg["hills"]))
                         for i, c in enumerate(curves)]
        self._place_scenery(cfg["scenery"])

    def _elevation(self, s: float, hills: Sequence[Sequence[float]]) -> float:
        """Sum of sine waves whose periods divide the lap, so the loop closes."""
        return sum(amp * math.sin(2 * math.pi * waves * s / self.length + phase)
                   for amp, waves, phase in hills)

    def _place_scenery(self, cfg: Dict[str, Any]) -> None:
        rng = random.Random(cfg["seed"])  # same scenery on every run
        edge = self.road_half + self.rumble
        for i, seg in enumerate(self.segments):
            if i % cfg["post_every"] == 0:
                seg.sprites += [(-(edge + 1.0), "post"), (edge + 1.0, "post")]
            sharp = abs(seg.curve) > 1 / cfg["chevron_radius"]
            if sharp and i % cfg["chevron_every"] == 0:
                # warning chevrons on the outside of the corner
                side = -1 if seg.curve > 0 else 1
                seg.sprites.append((side * (edge + 3.0), "chevron_r" if seg.curve > 0 else "chevron_l"))
            for side in (-1, 1):
                if rng.random() < cfg["tree_density"]:
                    offset = edge + rng.uniform(cfg["tree_min_offset"], cfg["tree_max_offset"])
                    kind = rng.choice(("pine", "pine", "tree", "bush"))
                    seg.sprites.append((side * offset, kind))

    # ---- lookups ----
    def index(self, s: float) -> int:
        return int((s % self.length) // self.segment_length)

    def curvature(self, s: float) -> float:
        return self.segments[self.index(s)].curve

    def elevation(self, s: float) -> float:
        """Elevation at distance `s`, linearly interpolated between segments."""
        s %= self.length
        i = int(s // self.segment_length)
        u = s / self.segment_length - i
        a, b = self.segments[i].y, self.segments[(i + 1) % len(self.segments)].y
        return a + (b - a) * u

    def slope(self, s: float) -> float:
        """Road grade (rise over run) at `s`."""
        i = self.index(s)
        nxt = self.segments[(i + 1) % len(self.segments)]
        return (nxt.y - self.segments[i].y) / self.segment_length

    def heading_at(self, s: float) -> float:
        """Direction of the road at `s` in the 2D map, radians."""
        return self.headings[self.index(s)]

    def map_position(self, s: float, d: float) -> Point:
        """2D map position of a point `d` meters right of the centerline."""
        i = self.index(s)
        px, py = self.points[i]
        h = self.headings[i]
        u = s % self.length - i * self.segment_length
        # right of a heading h in a y-down map is h + 90 degrees
        return (px + math.cos(h) * u - math.sin(h) * d, py + math.sin(h) * u + math.cos(h) * d)


class LapTimer:
    """Counts laps from the car's segment index.

    A lap counts when the index wraps from the last quarter of the track to
    the first, and only if the car passed the halfway point on the way, so
    reversing over the line does nothing. Only clean laps (never off the
    road) can set the best time.
    """

    def __init__(self, n_segments: int) -> None:
        self.n = n_segments
        self.prev_idx = 0
        self.halfway = False
        self.dirty = False
        self.lap_start: Optional[float] = None
        self.laps = 0
        self.last: Optional[float] = None
        self.last_clean = True
        self.best: Optional[float] = None

    def update(self, idx: int, t: float, on_road: bool, moving: bool) -> Optional[float]:
        """Feed one frame; returns the lap time when a lap was just completed."""
        if self.lap_start is None:
            if not moving:
                return None
            self.lap_start = t  # the clock starts when the car starts
        self.dirty |= not on_road
        if abs(idx - self.n // 2) < self.n // 8:
            self.halfway = True
        crossed = self.prev_idx > self.n * 3 // 4 and idx < self.n // 4
        self.prev_idx = idx
        if not (crossed and self.halfway):
            return None
        lap = t - self.lap_start
        self.laps += 1
        self.last, self.last_clean = lap, not self.dirty
        if self.last_clean and (self.best is None or lap < self.best):
            self.best = lap
        self.lap_start, self.halfway, self.dirty = t, False, False
        return lap

    def current(self, t: float) -> float:
        return 0.0 if self.lap_start is None else t - self.lap_start


def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--.--"
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
