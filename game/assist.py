"""Corner speed assist: slows the car down for corners, like a braking assist.

Why: with hand tracking the throttle is automatic and the brake is an
on/off fist, so judging braking points is hard. Without help the car
reaches corners faster than the tires can turn and slides straight on.
The assist does what a driver does: look down the road, work out the
fastest speed each corner allows, and brake early enough to get there.
The player only steers; the fist still brakes harder on top.
"""
import math
from typing import Any, Dict, Tuple

from game.track import Track

G = 9.81


def corner_speed_limit(track: Track, s: float, cfg: Dict[str, Any]) -> float:
    """Fastest speed (m/s) at `s` that still allows braking for every corner ahead.

    A corner of curvature k can be taken at v = sqrt(a_lat / k). To reach that
    speed after `d` meters of braking at a_brake, you may go at most
    sqrt(v^2 + 2 * a_brake * d) now (from v_end^2 = v_start^2 - 2*a*d).
    """
    a_lat = cfg["lateral_g"] * G
    a_brake = cfg["braking_g"] * G
    limit = math.inf
    d = 0.0
    while d <= cfg["lookahead_m"]:
        k = abs(track.curvature(s + d))
        if k > 1e-6:
            limit = min(limit, math.sqrt(a_lat / k + 2 * a_brake * d))
        d += cfg["step_m"]
    return limit


def apply_corner_assist(throttle: float, brake: float, speed: float, limit: float,
                        cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Return (throttle, brake) after the assist; it only ever slows the car."""
    if speed > limit:
        assist_brake = min(1.0, (speed - limit) * cfg["brake_gain"])
        return 0.0, max(brake, assist_brake)
    if speed > limit - cfg["coast_margin"]:
        return 0.0, brake  # close to the limit: stop accelerating, just roll
    return throttle, brake
