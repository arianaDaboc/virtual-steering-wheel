"""The game window: physics + renderer + HUD, behind the Controller interface.

RacingGame is both a Controller (it receives the same steer/throttle/brake
values a real game gets from the virtual gamepad) and the Display (the
camera view is shown inside the game window).
"""
import os
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from controller import Controller  # noqa: E402
from display import QUIT_KEY  # noqa: E402
from game.assist import apply_corner_assist, corner_speed_limit
from game.physics import CarParams, CarState, is_stuck, recover, step  # noqa: E402
from game.render import Renderer  # noqa: E402
from game.track import LapTimer, Track, fmt_time  # noqa: E402

WHITE = (240, 240, 240)
GREY = (170, 170, 170)
GREEN = (80, 220, 110)
RED = (235, 70, 60)
YELLOW = (250, 210, 60)


class RacingGame(Controller):
    """3D test track driven by the steering wheel."""

    name = "GAME"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        pygame.init()
        self.cfg = cfg
        self.refresh_interval = 1.0 / cfg["fps"]  # redraw rate, independent of the camera
        self.fps = 0.0
        self._preview: Optional[Tuple[np.ndarray, bool, pygame.Surface]] = None  # (frame, large, image)
        self.size = (cfg["width"], cfg["height"])
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(cfg["window_title"])
        self.fonts = {k: pygame.font.SysFont("consolas", s, bold=b)
                      for k, s, b in (("small", 18, False), ("mid", 24, True),
                                      ("big", 40, True), ("huge", 64, True))}
        self.track = Track(cfg["track"])
        self.params = CarParams.from_config(cfg["car"])
        self.renderer = Renderer(cfg["view"], self.track, self.size)
        self.inputs = (0.0, 0.0, 0.0)
        self.applied = (0.0, 0.0, 0.0)
        self._t_last = time.perf_counter()
        self.reset()

    def reset(self) -> None:
        """Back to the grid: car on the start line, lap times cleared."""
        self.car = CarState()
        self.cam_psi = 0.0
        self.timer = LapTimer(len(self.track.segments))
        self.stuck_time = 0.0
        self.message: Optional[Tuple[str, float]] = None  # (text, expires at)

    # ---- Controller interface ----
    def send(self, steer: float, throttle: float, brake: float) -> None:
        self.inputs = (steer, throttle, brake)

    def close(self) -> None:
        self.neutral()
        pygame.quit()  # safe to call more than once

    # ---- Display interface ----
    def show(self, frame: np.ndarray, large: bool = False) -> int:
        """Advance the simulation to now, draw everything, return the pressed key."""
        key = self._handle_events()
        now = time.perf_counter()
        # cap dt so a slow frame (or dragging the window) can't teleport the car
        dt = min(now - self._t_last, self.cfg["max_dt"])
        if now > self._t_last:
            self.fps += 0.05 * (1.0 / (now - self._t_last) - self.fps)  # smoothed for display
        self._t_last = now
        self.update(dt, now)
        self.renderer.draw(self.screen, self.car, self.cam_psi, self.applied[2] > 0.1, dt)
        self._draw_hud(frame, large, now)
        pygame.display.flip()
        return key

    def update(self, dt: float, now: float) -> None:
        """Physics, chase camera, lap timing and spin recovery for one frame."""
        car, cfg = self.car, self.cfg
        steer, throttle, brake = self.inputs
        assist = cfg["corner_assist"]
        if assist["enabled"]:
            limit = corner_speed_limit(self.track, car.s, assist)
            throttle, brake = apply_corner_assist(throttle, brake, car.vx, limit, assist)
        self.applied = (steer, throttle, brake)  # what the car really gets, for the HUD
        step(car, self.params, self.track, steer, throttle, brake, dt)
        # the camera swings after the car instead of being glued to it, which
        # is what makes slides visible from behind
        lag = cfg["view"]["camera_yaw_lag"]
        self.cam_psi += (car.psi - self.cam_psi) * min(dt / lag, 1.0)
        on_road = not car.on_grass
        self.timer.update(self.track.index(car.s), now, on_road, car.vx > 0.5)
        if car.hit_barrier:
            self.message = ("BARRIER!", now + 1.0)
        # spun out and stopped: count, then put the car back on the road
        self.stuck_time = self.stuck_time + dt if is_stuck(
            car, cfg["stuck_angle_deg"]) else 0.0
        if self.stuck_time > cfg["stuck_seconds"]:
            recover(car, self.track)
            self.cam_psi = 0.0
            self.stuck_time = 0.0
            self.message = ("RECOVERED", now + 1.5)

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

    # ---- HUD ----
    def _text(self, text: str, pos: Tuple[float, float], font: str = "small",
              color: Tuple[int, int, int] = WHITE, anchor: str = "topleft") -> None:
        img = self.fonts[font].render(text, True, color)
        shadow = self.fonts[font].render(text, True, (0, 0, 0))
        rect = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        self.screen.blit(shadow, rect.move(2, 2))
        self.screen.blit(img, rect)

    def _panel(self, rect: Tuple[int, int, int, int]) -> None:
        surf = pygame.Surface(rect[2:], pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 0, 0, 120), surf.get_rect(), border_radius=10)
        self.screen.blit(surf, rect[:2])

    def _camera_image(self, frame: np.ndarray, width: int, large: bool) -> pygame.Surface:
        """Camera frame as a pygame image, converted only when a new frame arrives.

        The game redraws ~60 times a second but the camera delivers ~30 or
        fewer frames, so converting every redraw would be wasted work.
        """
        if self._preview is None or self._preview[0] is not frame or self._preview[1] != large:
            self._preview = (frame, large, self._frame_surface(frame, width))
        return self._preview[2]

    def _frame_surface(self, frame: np.ndarray, width: int) -> pygame.Surface:
        h, w = frame.shape[:2]
        size = (width, int(h * width / w))
        rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(rgb.tobytes(), size, "RGB")

    def _draw_hud(self, frame: np.ndarray, large: bool, now: float) -> None:
        w, h = self.size
        cfg = self.cfg
        if large:
            # calibrating: the camera must be big enough to read its instructions
            cam = self._camera_image(frame, int(w * 0.6), large)
            r = cam.get_rect(center=(w // 2, h // 2))
            pygame.draw.rect(self.screen, (0, 0, 0), r.inflate(8, 8))
            self.screen.blit(cam, r)
            return
        cam = self._camera_image(frame, cfg["preview_width"], large)
        cam_pos = (w - cam.get_width() - 12, 12)
        pygame.draw.rect(self.screen, (0, 0, 0), cam.get_rect(topleft=cam_pos).inflate(6, 6))
        self.screen.blit(cam, cam_pos)

        t = self.timer
        self._panel((12, 12, 250, 150))
        self._text(f"LAP {t.laps + 1}", (24, 20), "mid", YELLOW)
        self._text(fmt_time(t.current(now)), (24, 48), "big")
        self._text(f"LAST {fmt_time(t.last)}", (24, 100), color=WHITE if t.last_clean else RED)
        self._text(f"BEST {fmt_time(t.best)}", (24, 124), color=GREEN)

        kmh = self.car.vx * 3.6
        self._panel((w - 230, h - 130, 218, 118))
        self._text(f"{kmh:3.0f}", (w - 60, h - 118), "huge", anchor="topright")
        self._text("km/h", (w - 24, h - 60), color=GREY, anchor="topright")
        bar = (w - 218, h - 34, 194, 10)
        pygame.draw.rect(self.screen, GREY, bar, 1)
        fill = min(kmh / cfg["hud_max_kmh"], 1.0)
        pygame.draw.rect(self.screen, YELLOW, (bar[0], bar[1], int(bar[2] * fill), bar[3]))

        size = cfg["view"]["minimap_size"]
        self.renderer.draw_minimap(self.screen, self.car, (12, h - size - 100))
        self._draw_inputs(12, h - 88)

        if self.car.on_grass:
            self._text("OFF TRACK", (w // 2, 70), "big", RED, "center")
        if self.message and now < self.message[1]:
            self._text(self.message[0], (w // 2, 120), "big", YELLOW, "center")
        self._text(f"r = reset   Esc = quit   {self.fps:3.0f} FPS", (w // 2, h - 14),
                   color=GREY, anchor="midbottom")

    def _draw_inputs(self, x: int, y: int) -> None:
        """Steering (centered) plus throttle and brake bars."""
        steer, throttle, brake = self.applied  # includes the corner assist
        self._panel((x, y, 250, 76))
        bx, bw = x + 70, 166
        for i, label in enumerate(("Steer", "Gas", "Brake")):
            self._text(label, (x + 10, y + 6 + i * 22), color=GREY)
            pygame.draw.rect(self.screen, GREY, (bx, y + 10 + i * 22, bw, 12), 1)
        mid = bx + bw // 2
        end = mid + int(steer * bw / 2)
        pygame.draw.rect(self.screen, WHITE, (min(mid, end), y + 10, abs(end - mid) + 1, 12))
        for i, (v, col) in enumerate(((throttle, GREEN), (brake, RED)), start=1):
            pygame.draw.rect(self.screen, col, (bx, y + 10 + i * 22, int(bw * v), 12))
