"""Tests for the 3D test track: geometry, lap timing and car physics (no window needed)."""
import math
from pathlib import Path

import pytest
import yaml

from game.physics import CarParams, CarState, is_stuck, recover, step, tire_force
from game.track import LapTimer, Track, fmt_time, wrap_angle

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())["game"]


@pytest.fixture(scope="module")
def track():
    return Track(CFG["track"])


@pytest.fixture(scope="module")
def params():
    return CarParams.from_config(CFG["car"])


class FlatRoad:
    """Endless flat straight with a huge road, to test the car on its own."""
    road_half, rumble, runoff, length = 1e6, 0.0, 1e6, 1e9

    def curvature(self, s):
        return 0.0

    def slope(self, s):
        return 0.0


def run(car, params, road, steer, throttle, brake, seconds, dt=1 / 60):
    for _ in range(int(seconds / dt)):
        step(car, params, road, steer, throttle, brake, dt)
    return car


# ---- track ----
def test_track_is_a_closed_loop(track):
    turn = sum(seg.curve for seg in track.segments) * track.segment_length
    assert turn == pytest.approx(2 * math.pi, abs=1e-6)  # exactly one full turn


def test_track_corners_are_drivable(track):
    tightest_radius = 1 / max(abs(seg.curve) for seg in track.segments)
    assert tightest_radius > 40


def test_elevation_closes_and_wraps(track):
    assert track.elevation(0.0) == pytest.approx(track.elevation(track.length))
    assert track.index(track.length + 1.0) == 0


def test_scenery_never_on_the_road(track):
    edge = track.road_half + track.rumble
    assert all(abs(off) > edge for seg in track.segments for off, _ in seg.sprites)


def test_wrap_angle():
    assert abs(wrap_angle(3 * math.pi)) == pytest.approx(math.pi)
    assert wrap_angle(-0.5) == -0.5


# ---- lap timer ----
def run_lap(timer, t0, n=100, on_road=True):
    t = t0
    for idx in list(range(n)) + [0]:
        timer.update(idx, t, on_road, moving=True)
        t += 0.1


def test_lap_counts_with_its_time():
    timer = LapTimer(100)
    run_lap(timer, 0.0)
    assert timer.laps == 1 and timer.last == pytest.approx(10.0)


def test_reversing_over_the_line_is_not_a_lap():
    timer = LapTimer(100)
    for idx in (0, 99, 98, 99, 0, 1):
        timer.update(idx, 0.0, True, moving=True)
    assert timer.laps == 0


def test_off_road_lap_cannot_be_best():
    timer = LapTimer(100)
    run_lap(timer, 0.0, on_road=False)
    assert not timer.last_clean and timer.best is None


def test_fmt_time():
    assert fmt_time(None) == "--:--.--"
    assert fmt_time(75.5) == "01:15.50"


# ---- tires ----
def test_tire_force_opposes_slip_and_peaks():
    assert tire_force(0.05, 1000, 10, 1.5) < 0 < tire_force(-0.05, 1000, 10, 1.5)
    forces = [abs(tire_force(math.radians(a), 1000, 10, 1.5)) for a in range(0, 30)]
    assert max(forces) == pytest.approx(1000, rel=1e-3)  # peak = available grip
    assert forces[-1] < max(forces)                       # sliding past the peak loses grip


# ---- car: numbers a real road car would have ----
def test_acceleration_0_100_is_realistic(params):
    car, t = CarState(), 0.0
    while car.vx < 100 / 3.6 and t < 30:
        step(car, params, FlatRoad(), 0, 1, 0, 1 / 60)
        t += 1 / 60
    assert 5 < t < 12


def test_braking_distance_is_realistic(params):
    car = CarState(vx=100 / 3.6)
    run(car, params, FlatRoad(), 0, 0, 1, 5)
    assert car.vx == 0.0
    assert 33 < car.s < 50  # real cars: about 36-45 m from 100 km/h


def test_brakes_never_push_backwards(params):
    car = run(CarState(), params, FlatRoad(), 0, 0, 1, 2)
    assert car.vx == 0.0 and car.s == 0.0


def test_positive_steer_turns_right(params):
    car = run(CarState(vx=15), params, FlatRoad(), 0.3, 0.2, 0, 1)
    assert car.r > 0 and car.d > 0


def test_cornering_grip_is_about_one_g(params):
    car = CarState(vx=20)
    for _ in range(300):  # hold 20 m/s on a steady circle
        need = params.drag * car.vx ** 2 + params.rolling * car.vx + (20 - car.vx) * 3000
        thr = max(0.0, min(1.0, need / min(params.max_drive_force, params.power / car.vx)))
        step(car, params, FlatRoad(), 0.6, thr, 0, 1 / 60)
    assert 0.7 < car.vx * car.r / 9.81 < 1.1


def test_car_is_stable_after_a_steering_jab(params):
    car = run(CarState(vx=30), params, FlatRoad(), 1.0, 0.5, 0, 0.5)
    run(car, params, FlatRoad(), 0.0, 0.5, 0, 5)
    assert abs(car.r) < 0.01            # settled, not spinning
    assert car.vx > 25                  # and did not scrub off all speed


def test_grass_has_less_grip(params, track):
    on_road = run(CarState(vx=25), params, track, 0.0, 0.0, 1, 1.0)
    on_grass = run(CarState(vx=25, d=track.road_half + 5), params, track, 0.0, 0.0, 1, 1.0)
    assert on_grass.on_grass
    assert on_grass.s > on_road.s       # brakes need more distance on grass


def test_standstill_full_lock_stays_finite(params):
    car = run(CarState(), params, FlatRoad(), 1.0, 0.0, 0, 2)
    assert all(math.isfinite(v) for v in (car.vx, car.vy, car.r, car.psi))


def test_barrier_stops_the_car(params, track):
    car = CarState(vx=30, psi=math.radians(60))
    run(car, params, track, 0.0, 0.6, 0, 3)
    wall = track.road_half + track.rumble + track.runoff
    assert abs(car.d) <= wall + 1e-9


def test_recover_after_spin(track):
    car = CarState(s=100, d=20, psi=math.radians(170), vx=0.5)
    assert is_stuck(car, 100)
    recover(car, track)
    assert car.psi == 0 and abs(car.d) <= track.road_half


def test_spun_car_counts_as_stuck_even_while_rolling():
    # with the automatic throttle a spun car keeps moving; it must still be recovered
    assert is_stuck(CarState(psi=math.radians(160), vx=8.0), 100)
    assert not is_stuck(CarState(psi=math.radians(30), vx=8.0), 100)


def full_lock_yank(params, speed, seconds=0.6):
    car = run(CarState(vx=speed), params, FlatRoad(), 1.0, 0.6, 0, seconds)
    run(car, params, FlatRoad(), 0.0, 0.6, 0, 4)
    return car


def test_assists_stop_a_full_lock_yank_from_spinning(params):
    car = full_lock_yank(params, 36)          # 130 km/h, hands thrown to full lock
    assert abs(car.r) < 0.05 and car.vx > 20  # back under control, still moving


def test_steer_assist_limits_the_wheel_angle_and_can_be_switched_off(params):
    from dataclasses import replace
    raw = replace(params, steer_assist=0.0, esc_gain=0.0)
    helped = run(CarState(vx=20), params, FlatRoad(), 1.0, 0.6, 0, 0.3)
    unhelped = run(CarState(vx=20), raw, FlatRoad(), 1.0, 0.6, 0, 0.3)
    assert helped.delta < unhelped.delta  # the assist asked the tires for less


# ---- the whole track with the player's controls ----
def test_track_is_drivable_with_auto_throttle_and_fist_brake(track, params):
    """A simple driver with only steer, fixed 0.6 throttle and on/off brake finishes a clean lap."""
    car, timer, t, dt = CarState(), LapTimer(len(track.segments)), 0.0, 1 / 30
    while timer.laps == 0 and t < 200:
        ahead = max(abs(track.curvature(car.s + k)) for k in range(0, int(30 + car.vx * 3), 4))
        look = 8 + car.vx * 0.6
        err = -car.psi + sum(track.curvature(car.s + k) for k in range(0, int(look), 2)) * 2
        steer = max(-1.0, min(1.0, 1.6 * err - 0.08 * car.d - 0.3 * car.r))
        brake = 1.0 if car.vx > math.sqrt(0.75 * 9.81 / max(ahead, 1e-4)) else 0.0
        step(car, params, track, steer, 0.0 if brake else 0.6, brake, dt)
        t += dt
        timer.update(track.index(car.s), t, not car.on_grass, car.vx > 0.5)
    assert timer.laps == 1 and timer.last_clean
