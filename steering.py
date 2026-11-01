"""Steering math: hand-line angle, calibration, One Euro filter and deadzone.

Everything here is pure logic, so it can be tested without a camera.
"""
import math
from typing import Optional, Sequence


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., 2012).

    Smooths heavily when the signal moves slowly (kills jitter) and lightly
    when it moves fast (keeps latency low).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Forget the history; the next sample passes through unchanged."""
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        """Filter sample `x` taken at time `t` (seconds)."""
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def steering_angle_deg(left_pt: Sequence[float], right_pt: Sequence[float]) -> float:
    """Angle of the left->right hand line against the horizontal, in degrees.

    Image y grows downwards, so positive = right hand lower = turning right.
    """
    dx = right_pt[0] - left_pt[0]
    dy = right_pt[1] - left_pt[1]
    return math.degrees(math.atan2(dy, dx))


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero out |value| < deadzone and rescale the rest back to the full range.

    The rescale avoids a jump from 0 to `deadzone` at the edge of the zone.
    """
    if abs(value) < deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1 - deadzone), value)


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Limit `value` to [low, high]."""
    return max(low, min(high, value))


class SteeringProcessor:
    """Turns two palm positions into a steering value in [-1, 1]."""

    def __init__(self, max_angle_deg: float, deadzone: float,
                 min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.max_angle_deg = max_angle_deg
        self.deadzone = deadzone
        self.filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.neutral_offset = 0.0
        self.last_raw_angle: Optional[float] = None

    def update(self, left_pt: Sequence[float], right_pt: Sequence[float], t: float) -> float:
        """Process one frame and return the steering value."""
        self.last_raw_angle = steering_angle_deg(left_pt, right_pt)
        value = clamp((self.last_raw_angle - self.neutral_offset) / self.max_angle_deg)
        value = self.filter(value, t)
        return apply_deadzone(value, self.deadzone)

    def calibrate(self, raw_angle: Optional[float] = None) -> bool:
        """Treat `raw_angle` (default: the last measured angle) as straight ahead.

        Returns False if there is no angle to calibrate from yet.
        """
        angle = self.last_raw_angle if raw_angle is None else raw_angle
        if angle is None:
            return False
        self.neutral_offset = angle
        self.filter.reset()  # old samples were relative to the previous offset
        return True
