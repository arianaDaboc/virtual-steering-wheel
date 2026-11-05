import math

import pytest

from steering import (OneEuroFilter, SteeringProcessor, apply_deadzone, clamp,
                      steering_angle_deg)


# ---- angle ----
def test_level_hands_give_zero_angle():
    assert steering_angle_deg((100, 200), (300, 200)) == 0.0


def test_right_hand_lower_is_positive():
    # image y grows downwards, so a larger y means lower on screen
    assert steering_angle_deg((0, 0), (10, 10)) == pytest.approx(45.0)


def test_right_hand_higher_is_negative():
    assert steering_angle_deg((0, 10), (10, 0)) == pytest.approx(-45.0)


# ---- deadzone ----
def test_inside_deadzone_is_zero():
    assert apply_deadzone(0.04, 0.05) == 0.0
    assert apply_deadzone(-0.04, 0.05) == 0.0


def test_deadzone_has_no_jump_at_edge():
    assert apply_deadzone(0.05, 0.05) == pytest.approx(0.0)
    assert apply_deadzone(0.0501, 0.05) == pytest.approx(0.0, abs=1e-3)


def test_deadzone_keeps_full_range_and_sign():
    assert apply_deadzone(1.0, 0.05) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.05) == pytest.approx(-1.0)


def test_clamp():
    assert clamp(2.0) == 1.0
    assert clamp(-3.0) == -1.0
    assert clamp(0.3) == 0.3


# ---- One Euro filter ----
def test_filter_passes_first_sample():
    f = OneEuroFilter(1.0, 0.02)
    assert f(0.7, 0.0) == 0.7


def test_filter_constant_input_stays_constant():
    f = OneEuroFilter(1.0, 0.02)
    out = [f(0.5, i / 30) for i in range(30)]
    assert all(v == pytest.approx(0.5) for v in out)


def test_filter_smooths_a_step_then_converges():
    f = OneEuroFilter(1.0, 0.02)
    f(0.0, 0.0)
    first = f(1.0, 1 / 30)
    assert 0.0 < first < 1.0  # smoothed, not an instant jump
    v = first
    for i in range(2, 120):
        v = f(1.0, i / 30)
    assert v == pytest.approx(1.0, abs=1e-3)


def test_filter_reduces_jitter():
    f = OneEuroFilter(1.0, 0.02)
    noisy = [0.01 * (1 if i % 2 else -1) for i in range(60)]  # +/-0.01 zigzag
    out = [f(x, i / 30) for i, x in enumerate(noisy)]
    assert max(abs(v) for v in out[10:]) < 0.01


def test_filter_reset_forgets_history():
    f = OneEuroFilter(1.0, 0.02)
    f(0.0, 0.0)
    f.reset()
    assert f(1.0, 1.0) == 1.0


# ---- SteeringProcessor ----
def make_processor(**kw):
    params = dict(max_angle_deg=45.0, deadzone=0.0, min_cutoff=1.0, beta=0.02)
    params.update(kw)
    return SteeringProcessor(**params)


def test_processor_full_lock_and_clamp():
    p = make_processor()
    assert p.update((0, 0), (10, 10), 0.0) == pytest.approx(1.0)   # 45 deg
    p.filter.reset()
    assert p.update((0, 0), (10, 20), 1.0) == pytest.approx(1.0)   # >45 deg, clamped


def test_calibration_offsets_neutral():
    p = make_processor()
    tilted = ((0, 0), (100, 100 * math.tan(math.radians(10))))  # 10 deg tilt
    p.update(*tilted, 0.0)
    assert p.calibrate()
    assert p.update(*tilted, 0.1) == pytest.approx(0.0, abs=1e-9)


def test_calibrate_without_measurement_fails():
    assert make_processor().calibrate() is False


# ---- response curve ----
from steering import response_curve  # noqa: E402


def test_response_curve_keeps_ends_and_sign():
    assert response_curve(1.0, 1.5) == 1.0
    assert response_curve(-1.0, 1.5) == -1.0
    assert response_curve(0.0, 1.5) == 0.0


def test_response_curve_is_gentler_near_center():
    assert response_curve(0.5, 1.5) == pytest.approx(0.354, abs=1e-3)
    assert response_curve(-0.5, 1.5) == pytest.approx(-0.354, abs=1e-3)
    assert response_curve(0.5, 1.0) == 0.5


def test_gain_reaches_full_lock_earlier():
    p = make_processor(gain=1.5)
    # 30 deg is 2/3 of the 45 deg range: with gain 1.5 that is already full lock
    tilt = ((0, 0), (100, 100 * math.tan(math.radians(30))))
    assert p.update(*tilt, 0.0) == pytest.approx(1.0)
