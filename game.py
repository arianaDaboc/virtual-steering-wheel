"""Built-in test bench: a tiny top-down racing game driven by the wheel.

The game is just another Controller, so it receives exactly the same
(steer, throttle, brake) values a real game gets from the virtual gamepad.
It also acts as the display: the camera view sits in the side panel.
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from controller import Controller  # noqa: E402
from display import QUIT_KEY  # noqa: E402

Point = Tuple[float, float]
GRASS = (54, 120, 58)
ROAD = (70, 70, 76)
PANEL = (24, 24, 28)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
CAR = (220, 40, 40)
CAR_OFF = (240, 150, 30)
GREEN = (60, 200, 90)
RED = (220, 60, 60)


# ---------------- pure logic (tested without a window) ----------------

def make_track(cx: float, cy: float, rx: float, ry: float,
               wobble: float, n: int) -> List[Point]:
    """Closed centerline: an ellipse with a wavy edge, so there are real corners."""
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t) + wobble * math.sin(3 * t))
            for t in (2 * math.pi * i / n for i in range(n))]


def nearest_segment(pts: List[Point], p: Point) -> Tuple[int, float]:
    """Index of the closest centerline segment and the distance to it."""
    best_i, best_d = 0, math.inf
    for i, (ax, ay) in enumerate(pts):
        bx, by = pts[(i + 1) % len(pts)]
        dx, dy = bx - ax, by - ay
        # projection of p onto the segment, clamped to its two ends
        u = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        d = math.hypot(ax + u * dx - p[0], ay + u * dy - p[1])
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


@dataclass
class Car:
    x: float
    y: float
    heading: float      # radians; 0 = +x, positive turns clockwise on screen (= right)
    speed: float = 0.0  # pixels per second

    def step(self, steer: float, throttle: float, brake: float, dt: float,
             on_track: bool, p: Dict[str, Any]) -> None:
        """Advance the car by `dt` seconds."""
        accel = throttle * p["accel"] - brake * p["brake_decel"] - p["drag"] * self.speed
        top = p["max_speed"] * (1.0 if on_track else p["offtrack_factor"])
        self.speed = min(max(0.0, self.speed + accel * dt), top)
        # no turning on the spot: steering authority grows with speed
        grip = min(self.speed / (p["full_steer_speed"] * p["max_speed"]), 1.0)
        self.heading += steer * math.radians(p["turn_rate_deg"]) * grip * dt
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt


class LapTimer:
    """Counts laps from the car's nearest segment index.

    A lap counts when the index wraps from the last quarter of the track to
    the first, and only if the car passed the halfway point on the way, so
    reversing over the line does nothing. Only clean laps (never off track)
    can set the best time.
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

    def update(self, idx: int, t: float, on_track: bool, moving: bool) -> Optional[float]:
        """Feed one frame; returns the lap time when a lap was just completed."""
        if self.lap_start is None:
            if not moving:
                return None
            self.lap_start = t  # the clock starts when the car starts
        self.dirty |= not on_track
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


# ---------------- the game window ----------------

class RacingGame(Controller):
    """Controller output that drives the car, and the window that shows everything."""

    name = "GAME"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        pygame.init()
        self.cfg = cfg
        self.w, self.h, self.panel_w = cfg["width"], cfg["height"], cfg["panel_width"]
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(cfg["window_title"])
        self.font = pygame.font.SysFont("consolas", 20)
        self.big_font = pygame.font.SysFont("consolas", 30, bold=True)
        self.track = make_track((self.w - self.panel_w) / 2, self.h / 2, cfg["track_rx"],
                                cfg["track_ry"], cfg["track_wobble"], cfg["track_points"])
        self.half_width = cfg["track_width"] / 2
        self.background = self._draw_track()
        self.inputs = (0.0, 0.0, 0.0)
        self.on_track = True
        self._t_last = time.perf_counter()
        self.reset()

    def reset(self) -> None:
        """Put the car back on the start line and clear the lap times."""
        (x0, y0), (x1, y1) = self.track[0], self.track[1]
        self.car = Car(x0, y0, math.atan2(y1 - y0, x1 - x0))
        self.timer = LapTimer(len(self.track))

    # ---- Controller interface ----
    def send(self, steer: float, throttle: float, brake: float) -> None:
        self.inputs = (steer, throttle, brake)

    def close(self) -> None:
        self.neutral()
        pygame.quit()  # safe to call more than once

    # ---- Display interface ----
    def show(self, frame: np.ndarray, large: bool = False) -> int:
        """Step the physics, draw the frame and return the pressed key."""
        key = self._handle_events()
        now = time.perf_counter()
        # cap dt so a slow frame (or dragging the window) can't teleport the car
        dt = min(now - self._t_last, self.cfg["max_dt"])
        self._t_last = now
        idx, dist = nearest_segment(self.track, (self.car.x, self.car.y))
        self.on_track = dist <= self.half_width
        self.car.step(*self.inputs, dt, self.on_track, self.cfg)
        self.timer.update(idx, now, self.on_track, self.car.speed > 0)
        self._render(frame, large, now)
        return key

    def _handle_events(self) -> int:
        key = -1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return QUIT_KEY
            if ev.type != pygame.KEYDOWN:
                continue
            if ev.key == pygame.K_ESCAPE:
                return QUIT_KEY
            if ev.key == pygame.K_r:
                self.reset()
            elif ev.key == pygame.K_SPACE:
                key = 32
            elif ev.unicode:
                key = ord(ev.unicode.lower())
        return key

    # ---- drawing ----
    def _draw_track(self) -> pygame.Surface:
        surf = pygame.Surface((self.w - self.panel_w, self.h))
        surf.fill(GRASS)
        width = int(self.cfg["track_width"])
        pygame.draw.lines(surf, ROAD, True, self.track, width)
        for p in self.track:  # round joints: thick pygame lines leave gaps at corners
            pygame.draw.circle(surf, ROAD, p, width // 2)
        (x0, y0), (x1, y1) = self.track[0], self.track[1]
        length = math.hypot(x1 - x0, y1 - y0)
        nx, ny = -(y1 - y0) / length * self.half_width, (x1 - x0) / length * self.half_width
        pygame.draw.line(surf, WHITE, (x0 - nx, y0 - ny), (x0 + nx, y0 + ny), 6)
        return surf

    def _frame_surface(self, frame: np.ndarray, width: int) -> pygame.Surface:
        h, w = frame.shape[:2]
        size = (width, int(h * width / w))
        rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(rgb.tobytes(), size, "RGB")

    def _draw_car(self) -> None:
        c = self.car
        fx, fy = math.cos(c.heading), math.sin(c.heading)
        hl, hw = self.cfg["car_length"] / 2, self.cfg["car_width"] / 2
        corners = [(c.x + fx * a - fy * b, c.y + fy * a + fx * b)
                   for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        pygame.draw.polygon(self.screen, CAR if self.on_track else CAR_OFF, corners)
        pygame.draw.line(self.screen, WHITE, (c.x, c.y), (c.x + fx * hl, c.y + fy * hl), 2)

    def _text(self, text: str, pos: Tuple[int, int], color: Tuple[int, int, int] = WHITE,
              big: bool = False) -> None:
        self.screen.blit((self.big_font if big else self.font).render(text, True, color), pos)

    def _render(self, frame: np.ndarray, large: bool, now: float) -> None:
        self.screen.blit(self.background, (0, 0))
        self._draw_car()
        px, pad = self.w - self.panel_w, 10
        pygame.draw.rect(self.screen, PANEL, (px, 0, self.panel_w, self.h))
        if large:
            # calibration: the camera needs to be big enough to read its panel
            cam = self._frame_surface(frame, self.w - self.panel_w)
            self.screen.blit(cam, (0, (self.h - cam.get_height()) // 2))
        else:
            cam = self._frame_surface(frame, self.panel_w - 2 * pad)
            self.screen.blit(cam, (px + pad, pad))
        y = (0 if large else cam.get_height()) + 3 * pad
        t = self.timer
        self._text(fmt_time(t.current(now)), (px + pad, y), big=True)
        rows = [(f"Lap   {t.laps + 1}", WHITE),
                (f"Last  {fmt_time(t.last)}", WHITE if t.last_clean else RED),
                (f"Best  {fmt_time(t.best)}", GREEN),
                (f"Speed {self.car.speed:5.0f} px/s", WHITE),
                ("" if self.on_track else "OFF TRACK", RED)]
        for i, (text, color) in enumerate(rows):
            self._text(text, (px + pad, y + 45 + i * 28), color)
        self._draw_inputs(px + pad, y + 200)
        self._text("r = reset   Esc = quit", (px + pad, self.h - 30), GREY)
        pygame.display.flip()

    def _draw_inputs(self, x: int, y: int) -> None:
        """Steering bar (centered) plus throttle and brake bars."""
        steer, throttle, brake = self.inputs
        label_w = 70
        bx, bw = x + label_w, self.panel_w - 20 - label_w
        for i, name in enumerate(("Steer", "Gas", "Brake")):
            self._text(name, (x, y - 3 + i * 24), GREY)
            pygame.draw.rect(self.screen, GREY, (bx, y + i * 24, bw, 14), 1)
        mid = bx + bw // 2
        end = mid + int(steer * bw / 2)
        pygame.draw.rect(self.screen, WHITE, (min(mid, end), y, abs(end - mid) + 1, 14))
        for i, (value, color) in enumerate(((throttle, GREEN), (brake, RED)), start=1):
            pygame.draw.rect(self.screen, color, (bx, y + i * 24, int(bw * value), 14))
