"""Pseudo-3D renderer: the road technique of classic arcade racers.

The track is a list of short flat segments. Each frame, every visible
segment's near and far edge is projected with a pinhole camera
(screen = focal * world / depth), and the segments are drawn from far to
near, so nearer road and hills cover what is behind them ("painter's
algorithm"). Curves come from bending the road a little more on every
segment; hills from each segment's elevation. No 3D library needed.
"""
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import pygame

from game.physics import CarState
from game.track import Track

Color = Tuple[int, int, int]
SKY_TOP: Color = (58, 110, 180)
SKY_BOTTOM: Color = (178, 208, 230)
FOG: Color = (190, 210, 222)
GRASS = ((72, 128, 56), (64, 116, 50))
ROAD = ((96, 96, 102), (90, 90, 96))
RUMBLE = ((200, 40, 40), (235, 235, 235))
LINE: Color = (235, 235, 235)
MOUNTAINS = ((108, 128, 156), (84, 112, 92))  # far (bluish), near (greenish)
CAR_BODY: Color = (196, 28, 36)
CAR_DARK: Color = (120, 14, 20)
GLASS: Color = (30, 38, 52)
TYRE: Color = (22, 22, 24)
TAIL: Color = (150, 0, 0)
TAIL_ON: Color = (255, 50, 40)

Projected = Tuple[float, float, float, float]  # screen x, screen y, half road width px, depth


def lerp_color(a: Color, b: Color, t: float) -> Color:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


class Renderer:
    """Draws the 3D view, the car, tire smoke and the minimap."""

    def __init__(self, cfg: Dict[str, Any], track: Track, size: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.track = track
        self.w, self.h = size
        self.focal = (self.w / 2) / math.tan(math.radians(cfg["fov_deg"]) / 2)
        self.horizon = self.h * cfg["horizon"]
        self.sky = self._make_sky()
        self.mountains = [self._make_mountains(i) for i in range(2)]
        self.car_px_w = int(cfg["car_width_m"] * self.focal / cfg["camera_back"])
        self.car_sprites = {(k, brake): self._make_car(k, brake)
                            for k in range(cfg["car_yaw_buckets"]) for brake in (False, True)}
        self.puffs = {kind: [self._make_puff(r, col) for r in range(6, 42, 4)]
                      for kind, col in (("smoke", (225, 225, 225)), ("dust", (150, 120, 80)))}
        self.particles: List[List[Any]] = []  # [x, y, vx, vy, age, life, kind]
        self.minimap = self._make_minimap(cfg["minimap_size"])
        self.rng = random.Random(0)

    # ---------------- one-time art ----------------
    def _make_sky(self) -> pygame.Surface:
        surf = pygame.Surface((self.w, int(self.horizon) + 2))
        for y in range(surf.get_height()):
            pygame.draw.line(surf, lerp_color(SKY_TOP, SKY_BOTTOM, y / surf.get_height()),
                             (0, y), (self.w, y))
        return surf

    def _make_mountains(self, layer: int) -> pygame.Surface:
        """Seamless mountain strip: one full turn of the car scrolls exactly its width."""
        parallax = self.cfg["mountain_parallax"][layer]
        width = int(2 * math.pi * self.focal * parallax)
        height = int(self.h * (0.16 if layer == 0 else 0.09))
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        rng = random.Random(11 + layer)
        waves = [(rng.uniform(0.3, 1.0), rng.randint(2, 14), rng.uniform(0, 6.3)) for _ in range(7)]
        norm = sum(a for a, _, _ in waves)
        pts = [(0, height)]
        for x in range(0, width + 1, 4):
            u = 2 * math.pi * x / width  # integer wave counts -> the strip wraps cleanly
            v = sum(a * math.sin(k * u + ph) for a, k, ph in waves) / norm
            pts.append((x, height * (0.45 - 0.45 * v)))
        pts.append((width, height))
        pygame.draw.polygon(surf, lerp_color(MOUNTAINS[layer], FOG, 0.35), pts)
        return surf

    def _make_car(self, bucket: int, braking: bool) -> pygame.Surface:
        """Rear view of the car; buckets show it turned so you see one side."""
        n = self.cfg["car_yaw_buckets"]
        yaw = (bucket / (n - 1) * 2 - 1) * self.cfg["car_max_yaw"]  # -max..max radians
        w = self.car_px_w
        h = int(w * 0.62)
        side = int(abs(math.sin(yaw)) * w * 0.9)
        surf = pygame.Surface((w + side + 4, h + 10), pygame.SRCALPHA)
        rear_w = int(w * math.cos(yaw))
        # pointing right -> we see the left flank, so the rear face shifts right
        x0 = side if yaw > 0 else 0
        pygame.draw.ellipse(surf, (0, 0, 0, 90), (x0 - 6, h - 10, rear_w + side + 12, 18))
        if side > 2:  # the visible flank: a parallelogram going into the distance
            fx = x0 - side if yaw > 0 else x0 + rear_w + side
            near_x = x0 if yaw > 0 else x0 + rear_w
            pygame.draw.polygon(surf, CAR_DARK, [(near_x, h * 0.32), (near_x, h * 0.86),
                                                 (fx, h * 0.72), (fx, h * 0.38)])
            pygame.draw.polygon(surf, GLASS, [(near_x, h * 0.12), (near_x, h * 0.32),
                                              (fx, h * 0.38), (fx, h * 0.24)])
        tw = int(rear_w * 0.17)
        for tx in (x0 + 2, x0 + rear_w - tw - 2):
            pygame.draw.rect(surf, TYRE, (tx, int(h * 0.7), tw, int(h * 0.27)), border_radius=4)
        pygame.draw.rect(surf, CAR_BODY, (x0, int(h * 0.3), rear_w, int(h * 0.52)), border_radius=8)
        pygame.draw.polygon(surf, CAR_BODY, [(x0 + rear_w * 0.12, h * 0.32), (x0 + rear_w * 0.22, h * 0.06),
                                             (x0 + rear_w * 0.78, h * 0.06), (x0 + rear_w * 0.88, h * 0.32)])
        pygame.draw.polygon(surf, GLASS, [(x0 + rear_w * 0.18, h * 0.3), (x0 + rear_w * 0.26, h * 0.1),
                                          (x0 + rear_w * 0.74, h * 0.1), (x0 + rear_w * 0.82, h * 0.3)])
        pygame.draw.rect(surf, CAR_DARK, (x0, int(h * 0.68), rear_w, int(h * 0.14)), border_radius=6)
        light = TAIL_ON if braking else TAIL
        lw, lh = int(rear_w * 0.2), int(h * 0.1)
        for lx in (x0 + int(rear_w * 0.04), x0 + rear_w - lw - int(rear_w * 0.04)):
            if braking:  # soft glow around the lit lamp
                pygame.draw.ellipse(surf, (255, 60, 40, 70), (lx - lw // 6, int(h * 0.38) - lh // 3,
                                                               lw + lw // 3, lh + 2 * (lh // 3)))
            pygame.draw.rect(surf, light, (lx, int(h * 0.38), lw, lh), border_radius=3)
        pygame.draw.rect(surf, (235, 225, 120), (x0 + rear_w // 2 - int(rear_w * 0.11), int(h * 0.55),
                                                 int(rear_w * 0.22), int(h * 0.1)))
        return surf

    @staticmethod
    def _make_puff(radius: int, color: Color) -> pygame.Surface:
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -2):  # soft edge: alpha grows toward the center
            alpha = int(70 * (1 - r / radius) + 10)
            pygame.draw.circle(surf, (*color, alpha), (radius, radius), r)
        return surf

    def _make_minimap(self, size: int) -> pygame.Surface:
        xs, ys = zip(*self.track.points)
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        self._map_scale = (size - 20) / span
        self._map_origin = (min(xs), min(ys))
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 0, 0, 110), surf.get_rect(), border_radius=10)
        pts = [self._map_point(p) for p in self.track.points]
        pygame.draw.lines(surf, (230, 230, 230), True, pts, 4)
        pygame.draw.circle(surf, (255, 255, 255), pts[0], 5)
        return surf

    def _map_point(self, p: Tuple[float, float]) -> Tuple[int, int]:
        return (int(10 + (p[0] - self._map_origin[0]) * self._map_scale),
                int(10 + (p[1] - self._map_origin[1]) * self._map_scale))

    # ---------------- per frame ----------------
    def draw(self, screen: pygame.Surface, car: CarState, cam_psi: float,
             braking: bool, dt: float) -> None:
        """Draw sky, road, scenery, car and smoke for the current state."""
        cfg, track = self.cfg, self.track
        back = cfg["camera_back"]
        # chase camera: `back` meters behind the car along the camera's heading
        cam_s = (car.s - back * math.cos(cam_psi)) % track.length
        cam_d = car.d - back * math.sin(cam_psi)
        cam_y = track.elevation(cam_s) + cfg["camera_height"]

        screen.blit(self.sky, (0, 0))
        heading = track.heading_at(cam_s) + cam_psi
        for layer, strip in enumerate(self.mountains):
            sw = strip.get_width()
            x = -(heading * self.focal * cfg["mountain_parallax"][layer]) % sw
            y = int(self.horizon) - strip.get_height() + 2
            screen.blit(strip, (x - sw, y))
            screen.blit(strip, (x, y))
            if x + sw < self.w:
                screen.blit(strip, (x + sw, y))
        # distant fields behind the last drawn segment
        pygame.draw.rect(screen, lerp_color(GRASS[0], FOG, 0.75),
                         (0, int(self.horizon), self.w, self.h - int(self.horizon)))

        projected = self._project(cam_s, cam_d, cam_y, cam_psi)
        for item in reversed(projected):
            self._draw_segment(screen, *item)

        car_y = self.horizon - (track.elevation(car.s) - cam_y) * self.focal / back
        self._draw_car(screen, car, cam_psi, braking, car_y, dt)

    def _project(self, cam_s: float, cam_d: float, cam_y: float, cam_psi: float) -> List[Any]:
        """Project segment edges near -> far; returns drawable items."""
        track, cfg = self.track, self.cfg
        n_seg, seg_len = len(track.segments), track.segment_length
        base = track.index(cam_s)
        into = cam_s - base * seg_len
        near = cfg["near_clip"]
        cx, f, hz = self.w / 2, self.focal, self.horizon
        x, dx = -cam_d, -math.tan(cam_psi)  # road center and its drift, camera space
        max_y = float(self.h)  # nearest-so-far top of the ground: lower segments are hidden
        out = []
        for n in range(cfg["draw_distance"]):
            i = (base + n) % n_seg
            seg, nxt = track.segments[i], track.segments[(i + 1) % n_seg]
            z1, z2 = n * seg_len - into, (n + 1) * seg_len - into
            x1, x2 = x, x + dx * seg_len
            y1, y2 = seg.y - cam_y, nxt.y - cam_y
            x, dx = x2, dx + seg.curve * seg_len  # bend the road for the next segment
            if z2 <= near:
                continue
            if z1 < near:  # cut the segment at the near plane
                u = (near - z1) / (z2 - z1)
                x1, y1, z1 = x1 + (x2 - x1) * u, y1 + (y2 - y1) * u, near
            p1 = (cx + x1 * f / z1, hz - y1 * f / z1, track.road_half * f / z1, z1)
            p2 = (cx + x2 * f / z2, hz - y2 * f / z2, track.road_half * f / z2, z2)
            visible = p2[1] < max_y  # far edge pokes above everything nearer
            max_y = min(max_y, p2[1]) if visible else max_y
            out.append((i, p1, p2, visible, seg.sprites))
        return out

    def _fog(self, color: Color, z: float) -> Color:
        start = self.cfg["fog_start"]
        end = self.cfg["draw_distance"] * self.track.segment_length
        t = min(max((z - start) / (end - start), 0.0), 1.0)
        return lerp_color(color, FOG, t ** 0.8)

    def _draw_segment(self, screen: pygame.Surface, i: int, p1: Projected, p2: Projected,
                      visible: bool, sprites: Sequence[Tuple[float, str]]) -> None:
        if visible and p1[1] > p2[1]:
            z = p1[3]
            stripe = (i // 3) % 2
            top, bottom = int(p2[1]), int(p1[1]) + 1
            pygame.draw.rect(screen, self._fog(GRASS[stripe], z), (0, top, self.w, bottom - top))
            rumble = self.track.rumble / self.track.road_half
            self._quad(screen, self._fog(RUMBLE[stripe], z), p1, p2, 1 + rumble)
            self._quad(screen, self._fog(ROAD[stripe], z), p1, p2, 1.0)
            if z < self.cfg["line_distance"]:
                line = self._fog(LINE, z)
                self._strip(screen, line, p1, p2, 0.93, 0.97)
                self._strip(screen, line, p1, p2, -0.97, -0.93)
                if i % 4 < 2:  # dashed center line
                    self._strip(screen, line, p1, p2, -0.02, 0.02)
            if i == 0:
                self._start_line(screen, p1, p2)
        for offset, kind in sprites:
            self._draw_sprite(screen, p1, offset, kind)

    def _start_line(self, screen: pygame.Surface, p1: Projected, p2: Projected) -> None:
        """Checkered start/finish line: two rows of squares across the road."""
        cells = self.cfg["start_line_cells"]
        rows = [(0.0, 0.2), (0.2, 0.4)]  # fractions of the segment length
        for r, (u0, u1) in enumerate(rows):
            a = tuple(p1[k] + (p2[k] - p1[k]) * u0 for k in range(4))
            b = tuple(p1[k] + (p2[k] - p1[k]) * u1 for k in range(4))
            for c in range(cells):
                col = (240, 240, 240) if (c + r) % 2 else (20, 20, 20)
                f0, f1 = -1 + 2 * c / cells, -1 + 2 * (c + 1) / cells
                self._strip(screen, col, a, b, f0, f1)  # type: ignore[arg-type]

    @staticmethod
    def _quad(screen: pygame.Surface, color: Color, p1: Projected, p2: Projected, scale: float) -> None:
        pygame.draw.polygon(screen, color, [(p1[0] - p1[2] * scale, p1[1]), (p1[0] + p1[2] * scale, p1[1]),
                                            (p2[0] + p2[2] * scale, p2[1]), (p2[0] - p2[2] * scale, p2[1])])

    @staticmethod
    def _strip(screen: pygame.Surface, color: Color, p1: Projected, p2: Projected,
               a: float, b: float) -> None:
        """Band between fractions a..b of the half road width (-1 = left edge)."""
        pygame.draw.polygon(screen, color, [(p1[0] + p1[2] * a, p1[1]), (p1[0] + p1[2] * b, p1[1]),
                                            (p2[0] + p2[2] * b, p2[1]), (p2[0] + p2[2] * a, p2[1])])

    def _draw_sprite(self, screen: pygame.Surface, p: Projected, offset: float, kind: str) -> None:
        z = p[3]
        scale = self.focal / z
        x = p[0] + offset * scale
        y = p[1]
        if scale < 0.6 or x < -400 or x > self.w + 400:
            return
        if kind == "pine":
            h, w = 9 * scale, 2.6 * scale
            pygame.draw.rect(screen, self._fog((80, 56, 36), z), (x - w * 0.08, y - h * 0.25, w * 0.16, h * 0.25))
            for k, col in enumerate(((34, 82, 44), (40, 94, 50), (48, 108, 58))):
                base_y, top_y, half = y - h * (0.15 + 0.25 * k), y - h * (0.6 + 0.2 * k), w * (0.5 - 0.1 * k)
                pygame.draw.polygon(screen, self._fog(col, z), [(x - half, base_y), (x + half, base_y), (x, top_y)])
        elif kind == "tree":
            h = 7 * scale
            pygame.draw.rect(screen, self._fog((90, 64, 40), z), (x - 0.2 * scale, y - h * 0.45, 0.4 * scale, h * 0.45))
            pygame.draw.circle(screen, self._fog((58, 112, 52), z), (x, y - h * 0.62), 2.2 * scale)
            pygame.draw.circle(screen, self._fog((70, 128, 60), z), (x - 0.6 * scale, y - h * 0.7), 1.4 * scale)
        elif kind == "bush":
            pygame.draw.ellipse(screen, self._fog((60, 104, 48), z), (x - 1.4 * scale, y - 1.4 * scale, 2.8 * scale, 1.6 * scale))
        elif kind == "post":
            pygame.draw.rect(screen, self._fog((240, 240, 240), z), (x - 0.06 * scale, y - 1.1 * scale, 0.12 * scale, 1.1 * scale))
            pygame.draw.rect(screen, self._fog((30, 30, 30), z), (x - 0.06 * scale, y - 0.9 * scale, 0.12 * scale, 0.15 * scale))
        elif kind.startswith("chevron"):
            w, h = 1.6 * scale, 1.0 * scale
            top = y - 1.9 * scale
            pygame.draw.rect(screen, self._fog((60, 60, 60), z), (x - 0.05 * scale, top + h, 0.1 * scale, 0.9 * scale))
            pygame.draw.rect(screen, self._fog((240, 200, 0), z), (x - w / 2, top, w, h))
            d = 1 if kind == "chevron_r" else -1  # arrows point the way the road turns
            for k in (-0.35, 0.05):
                ax = x + d * k * w
                pygame.draw.polygon(screen, self._fog((20, 20, 20), z), [
                    (ax - d * 0.15 * w, top + 0.15 * h), (ax + d * 0.15 * w, top + 0.5 * h),
                    (ax - d * 0.15 * w, top + 0.85 * h)])

    def _draw_car(self, screen: pygame.Surface, car: CarState, cam_psi: float,
                  braking: bool, ground_y: float, dt: float) -> None:
        cfg = self.cfg
        n = cfg["car_yaw_buckets"]
        # what the camera sees: the car's heading against the lagging camera
        # plus some of the slide angle, so drifts look like drifts
        yaw = car.psi - cam_psi + math.atan2(car.vy, max(car.vx, 1.0)) * 0.5
        k = round((max(-1.0, min(1.0, yaw / cfg["car_max_yaw"])) + 1) / 2 * (n - 1))
        sprite = self.car_sprites[(k, braking)]
        shake = self.rng.uniform(-2, 2) if car.on_grass and car.vx > 2 else 0.0
        dive = max(-8.0, min(8.0, -car.ax * 0.8))  # nose dips under braking, squats when accelerating
        x = self.w / 2 - sprite.get_width() / 2
        y = ground_y - sprite.get_height() + 6 + shake + dive * 0.3
        self._update_particles(car, x, y + sprite.get_height(), sprite.get_width(), dt)
        screen.blit(sprite, (x, y))
        self._draw_particles(screen)

    def _update_particles(self, car: CarState, x: float, bottom: float, width: float, dt: float) -> None:
        sliding = abs(car.slip_rear) > self.cfg["smoke_slip"] or (
            abs(car.slip_front) > self.cfg["smoke_slip"] * 1.5 and car.vx > 5)
        kind = "dust" if car.on_grass else "smoke"
        if car.vx > 3 and (sliding or car.on_grass):
            for wheel_x in (x + width * 0.15, x + width * 0.85):
                self.particles.append([wheel_x + self.rng.uniform(-8, 8), bottom - 10,
                                       self.rng.uniform(-40, 40), self.rng.uniform(-60, -20),
                                       0.0, self.rng.uniform(0.4, 0.8), kind])
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[4] += dt
        self.particles = [p for p in self.particles if p[4] < p[5]][-self.cfg["max_particles"]:]

    def _draw_particles(self, screen: pygame.Surface) -> None:
        for x, y, _, _, age, life, kind in self.particles:
            frames = self.puffs[kind]
            puff = frames[min(int(age / life * len(frames)), len(frames) - 1)]  # grows as it ages
            screen.blit(puff, (x - puff.get_width() / 2, y - puff.get_height() / 2))

    def draw_minimap(self, screen: pygame.Surface, car: CarState, pos: Tuple[int, int]) -> None:
        screen.blit(self.minimap, pos)
        mx, my = self._map_point(self.track.map_position(car.s, car.d))
        pygame.draw.circle(screen, (0, 0, 0), (pos[0] + mx, pos[1] + my), 7)
        pygame.draw.circle(screen, CAR_BODY, (pos[0] + mx, pos[1] + my), 5)
