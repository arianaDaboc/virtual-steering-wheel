"""Car physics: a dynamic "bicycle model" with real tire behaviour.

Why this model: it is what vehicle-dynamics courses use as the first
realistic car model. The two wheels of each axle are merged into one, and
each tire produces sideways force from its *slip angle* (the angle between
where the wheel points and where it is actually moving). That is enough to
get understeer, oversteer, drifting, weight transfer under braking and
grip loss on grass, while staying a few dozen lines of math.

Coordinates: the car frame has x forward and y to the right; yaw rate `r`
is positive when turning right. The car's place on the circuit is kept in
track coordinates (Frenet frame): `s` = distance along the centerline,
`d` = meters right of the centerline, `psi` = heading relative to the road.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict

from game.track import Track, wrap_angle

G = 9.81
STOP_SPEED = 0.05  # m/s; slower than this with the brake on counts as stopped


@dataclass
class CarParams:
    mass: float               # kg
    yaw_inertia: float        # kg m^2
    cg_to_front: float        # m, center of gravity to front axle
    cg_to_rear: float         # m
    cg_height: float          # m, sets how much weight shifts under braking
    power: float              # W, engine power at the rear wheels
    max_drive_force: float    # N, low-speed traction/torque limit
    brake_force: float        # N, total at full brake
    brake_front_share: float  # 0..1
    drag: float               # N per (m/s)^2, aerodynamic
    rolling: float            # N per m/s
    mu_road: float            # tire friction coefficient on asphalt
    mu_grass: float
    grass_drag: float         # extra N per m/s on grass
    tire_b_front: float       # Pacejka stiffness factor, front tires
    tire_b_rear: float        # stiffer rear = stable car that understeers a little,
                              # like road cars are tuned from the factory
    tire_c: float             # Pacejka shape factor (>1 = grip drops past the peak)
    max_steer_deg: float      # wheel angle at full lock, standing still
    steer_speed_ref: float    # m/s; lock shrinks with (v/ref)^2 like a real driver's hands
    steer_rate_deg: float     # how fast the wheels can turn, deg/s
    traction_control: float   # share of rear grip the engine may use (1 = off)
    steer_assist: float       # 0 = off; else max share of grip the steering may ask for
    steer_assist_slip_deg: float  # extra wheel angle the tires need to build that grip
    esc_gain: float           # N m per rad/s of excess yaw rate (0 = ESC off)
    esc_max_moment: float     # N m, what braking single wheels can give
    esc_tolerance: float      # rad/s of excess yaw rate before ESC steps in
    substep: float            # s, integration step
    low_speed: float          # m/s, below this blend into a simple kinematic model

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CarParams":
        return cls(**cfg)

    @property
    def wheelbase(self) -> float:
        return self.cg_to_front + self.cg_to_rear


@dataclass
class CarState:
    s: float = 0.0       # m along the track
    d: float = 0.0       # m right of the centerline
    psi: float = 0.0     # rad, heading relative to the road (+ = pointing right)
    vx: float = 0.0      # m/s forward
    vy: float = 0.0      # m/s sideways (+ = sliding right)
    r: float = 0.0       # rad/s yaw rate (+ = turning right)
    delta: float = 0.0   # rad, front wheel angle
    ax: float = 0.0      # m/s^2 last longitudinal acceleration (for weight transfer)
    # read-outs for the renderer and HUD
    slip_front: float = 0.0
    slip_rear: float = 0.0
    on_grass: bool = False
    hit_barrier: bool = False

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def tire_force(alpha: float, peak: float, b: float, c: float) -> float:
    """Simplified Pacejka "magic formula": sideways force for slip angle `alpha`.

    Grows almost linearly for small slip, peaks around 5-8 degrees, then
    falls off a little: past the peak the tire slides and you lose grip.
    """
    return -peak * math.sin(c * math.atan(b * alpha))


def _axle_lateral_peak(mu: float, fz: float, fx: float) -> float:
    """Friction circle: grip used for braking/accelerating is not available for turning."""
    total = mu * fz
    return math.sqrt(max(total * total - fx * fx, (0.1 * total) ** 2))


def step(car: CarState, p: CarParams, track: Track,
         steer: float, throttle: float, brake: float, dt: float) -> None:
    """Advance the car by `dt` seconds with fixed small substeps for stability."""
    n = max(1, math.ceil(dt / p.substep))
    car.hit_barrier = False
    for _ in range(n):
        _substep(car, p, track, steer, throttle, brake, dt / n)


def _substep(car: CarState, p: CarParams, track: Track,
             steer: float, throttle: float, brake: float, h: float) -> None:
    a, b, L = p.cg_to_front, p.cg_to_rear, p.wheelbase
    m = p.mass
    vx = max(car.vx, 0.0)

    # steering: less lock at speed, and the wheels turn at a finite rate
    edge = track.road_half + track.rumble
    car.on_grass = abs(car.d) > edge
    mu = p.mu_grass if car.on_grass else p.mu_road

    target = steer * math.radians(p.max_steer_deg) / (1 + (vx / p.steer_speed_ref) ** 2)
    if p.steer_assist > 0 and vx > p.low_speed:
        # steering assist: never ask the tires for more sideways grip than
        # they have. A turn of radius R at speed v needs v^2/R of grip, and
        # for a wheel angle delta, R ~ wheelbase / tan(delta).
        # Tires also need some slip angle to make that force, so allow a
        # little extra wheel angle on top (`steer_assist_slip`).
        limit = math.atan(L * p.steer_assist * mu * G / (vx * vx)) + math.radians(p.steer_assist_slip_deg)
        target = max(-limit, min(limit, target))
    max_move = math.radians(p.steer_rate_deg) * h
    car.delta += max(-max_move, min(max_move, target - car.delta))
    delta = car.delta

    # weight transfer: braking (ax < 0) loads the front, accelerating the rear
    shift = m * p.cg_height * car.ax / L
    fz_front = max(m * G * b / L - shift, 0.1 * m * G)
    fz_rear = max(m * G * a / L + shift, 0.1 * m * G)

    # longitudinal forces: rear-wheel drive, brakes on both axles
    drive = throttle * min(p.max_drive_force, p.power / max(vx, 1.0))
    drive = min(drive, p.traction_control * mu * fz_rear)

    # ESC (electronic stability control): compare the real yaw rate with
    # the one the driver asked for. Spinning faster than that = the rear is
    # sliding out, so brake single wheels (a yaw moment) and cut the engine.
    esc_moment = 0.0
    if p.esc_gain > 0 and vx > p.low_speed:
        grip_limit = mu * G / vx
        wanted = max(-grip_limit, min(grip_limit, vx * math.tan(delta) / L))
        excess = car.r - wanted
        if abs(car.r) > abs(wanted) and abs(excess) > p.esc_tolerance:
            esc_moment = max(-p.esc_max_moment, min(p.esc_max_moment, -p.esc_gain * excess))
            drive = 0.0
    moving = 1.0 if vx > STOP_SPEED else 0.0  # brakes never push the car backwards
    fx_front = -brake * p.brake_force * p.brake_front_share * moving
    fx_rear = drive - brake * p.brake_force * (1 - p.brake_front_share) * moving
    fx_front = max(fx_front, -mu * fz_front)
    fx_rear = max(min(fx_rear, mu * fz_rear), -mu * fz_rear)

    # lateral tire forces from slip angles (speed floored to avoid dividing by ~0)
    v_safe = max(vx, p.low_speed)
    car.slip_front = math.atan2(car.vy + a * car.r, v_safe) - delta
    car.slip_rear = math.atan2(car.vy - b * car.r, v_safe)
    fy_front = tire_force(car.slip_front, _axle_lateral_peak(mu, fz_front, fx_front), p.tire_b_front, p.tire_c)
    fy_rear = tire_force(car.slip_rear, _axle_lateral_peak(mu, fz_rear, fx_rear), p.tire_b_rear, p.tire_c)

    resist = p.drag * vx * vx + p.rolling * vx + (p.grass_drag * vx if car.on_grass else 0.0)
    resist += m * G * track.slope(car.s)  # uphill slows you, downhill speeds you up

    cos_d, sin_d = math.cos(delta), math.sin(delta)
    ax = (fx_rear + fx_front * cos_d - fy_front * sin_d - resist) / m
    ay = (fy_rear + fy_front * cos_d + fx_front * sin_d) / m
    r_dot = (a * (fy_front * cos_d + fx_front * sin_d) - b * fy_rear + esc_moment) / p.yaw_inertia

    car.ax = ax
    car.vx = max(car.vx + (ax + car.vy * car.r) * h, 0.0)  # no reverse gear
    car.vy += (ay - vx * car.r) * h
    car.r += r_dot * h
    if brake > 0 and car.vx < STOP_SPEED:
        car.vx = 0.0  # brakes hold a stopped car instead of letting it creep

    if vx < p.low_speed:
        # at walking pace tire slip is meaningless: roll like a kinematic car
        w = vx / p.low_speed
        car.r = w * car.r + (1 - w) * car.vx * math.tan(delta) / L
        car.vy *= w

    # move along the track (Frenet frame)
    kappa = track.curvature(car.s)
    cos_p, sin_p = math.cos(car.psi), math.sin(car.psi)
    s_dot = (car.vx * cos_p - car.vy * sin_p) / max(1 - car.d * kappa, 0.1)
    car.s = (car.s + s_dot * h) % track.length
    car.d += (car.vx * sin_p + car.vy * cos_p) * h
    car.psi = wrap_angle(car.psi + (car.r - kappa * s_dot) * h)

    wall = edge + track.runoff
    if abs(car.d) > wall:
        # barrier: lose most of the speed and stop rotating
        car.d = math.copysign(wall, car.d)
        car.vx *= 0.3
        car.vy = 0.0
        car.r = 0.0
        car.psi *= 0.3
        car.hit_barrier = True


def is_stuck(car: CarState, max_angle_deg: float) -> bool:
    """Facing sideways or backwards. Speed doesn't matter: with the automatic
    throttle a spun car keeps rolling, so it would never count as stopped."""
    return abs(car.psi) > math.radians(max_angle_deg)


def recover(car: CarState, track: Track) -> None:
    """Put the car back on the road at the same place, pointing the right way."""
    car.d = max(-track.road_half / 2, min(track.road_half / 2, car.d))
    car.psi = car.vx = car.vy = car.r = car.delta = car.ax = 0.0
