import json

import pytest

from calibration import (Calibration, CalibrationSession, load_calibration,
                         save_calibration)
from steering import SteeringProcessor

FPS = 30


def make_session(include_range=True):
    return CalibrationSession(hold_seconds=1.0, include_range=include_range,
                              min_span_deg=15.0, max_spread_deg=4.0,
                              max_jump_deg=8.0, motion_frames=3)


def hold(session, angle, t0, seconds=1.1):
    """Feed a steady angle for `seconds`; return the time after the last frame."""
    n = int(seconds * FPS)
    for i in range(n + 1):
        session.update(angle, t0 + i / FPS)
    return t0 + (n + 1) / FPS


# ---- session flow ----
def test_full_session_measures_neutral_and_spans():
    s = make_session()
    t = hold(s, 5.0, 0.0)          # straight (camera slightly tilted)
    t = hold(s, -30.0, t)          # full left: right hand higher
    hold(s, 45.0, t)               # full right
    assert s.done
    assert s.result() == Calibration(5.0, 35.0, 40.0)


def test_center_only_session():
    s = make_session(include_range=False)
    hold(s, -3.0, 0.0)
    assert s.done
    assert s.result() == Calibration(-3.0, None, None)


def test_progress_grows_while_holding():
    s = make_session()
    s.update(0.0, 0.0)
    s.update(0.0, 0.5)
    assert s.progress == pytest.approx(0.5)


def test_losing_hands_restarts_the_step():
    s = make_session()
    hold(s, 0.0, 0.0, seconds=0.8)
    s.update(None, 0.85)
    assert s.progress == 0.0
    assert s.index == 0


def test_turning_restarts_the_step():
    s = make_session()
    hold(s, 0.0, 0.0, seconds=0.8)
    for i in range(3):               # hands really move to a new pose
        s.update(-30.0, 0.9 + i / FPS)
    assert s.index == 0
    assert s.progress < 0.1          # counting starts again from the new pose


def test_slow_drift_is_rejected():
    s = make_session()
    for i in range(40):              # creeps 0.5 deg per frame: never a big jump
        s.update(i * 0.5, i / FPS)
    assert s.index == 0
    assert s.message == "Hold still"


def test_too_small_turn_is_rejected():
    s = make_session()
    t = hold(s, 0.0, 0.0)
    hold(s, -5.0, t)               # only 5 deg left, need 15
    assert s.step.key == "left"
    assert "further left" in s.message


def test_single_glitch_frame_is_ignored():
    s = make_session(include_range=False)
    for i in range(40):
        s.update(80.0 if i == 10 else 2.0, i / FPS)
    assert s.result().neutral_deg == 2.0


def test_skip_before_center_cancels():
    s = make_session()
    s.skip()
    assert s.done and s.result() is None


def test_skip_after_center_keeps_neutral():
    s = make_session()
    hold(s, 4.0, 0.0)
    s.skip()
    assert s.done
    assert s.result() == Calibration(4.0, None, None)


# ---- save / load ----
def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    cal = Calibration(2.5, 30.0, None)
    save_calibration(cal, path)
    assert load_calibration(path) == cal


def test_load_missing_file(tmp_path):
    assert load_calibration(tmp_path / "nope.json") is None


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps({"left_span_deg": 3})])
def test_load_broken_file(tmp_path, content):
    path = tmp_path / "calibration.json"
    path.write_text(content)
    assert load_calibration(path) is None


# ---- steering with a calibrated range ----
def test_asymmetric_range_reaches_full_lock_on_both_sides():
    p = SteeringProcessor(max_angle_deg=45.0, deadzone=0.0, min_cutoff=1.0, beta=0.02)
    p.apply_calibration(Calibration(0.0, 20.0, 40.0))
    # a 20 deg left turn is already full lock, a 20 deg right turn only half
    assert p.update((0, 0), (100, -100 * 0.36397), 0.0) == pytest.approx(-1.0, abs=1e-3)
    p.filter.reset()
    assert p.update((0, 0), (100, 100 * 0.36397), 1.0) == pytest.approx(0.5, abs=1e-3)


def test_quick_recenter_keeps_the_range():
    p = SteeringProcessor(max_angle_deg=45.0, deadzone=0.0, min_cutoff=1.0, beta=0.02)
    p.apply_calibration(Calibration(0.0, 20.0, 40.0))
    p.update((0, 0), (100, 10), 0.0)
    p.calibrate()
    cal = p.current_calibration()
    assert cal.left_span_deg == 20.0 and cal.right_span_deg == 40.0
    assert cal.neutral_deg != 0.0
