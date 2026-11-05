import math
from pathlib import Path

import pytest
import yaml

from game.assist import apply_corner_assist, corner_speed_limit
from game.track import Track

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())["game"]
ASSIST = CFG["corner_assist"]


class OneCorner:
    """Straight road with one corner of radius 100 m starting 100 m ahead."""
    def curvature(self, s):
        return 0.01 if 100 <= s <= 160 else 0.0


def test_no_corner_means_no_limit():
    class Straight:
        def curvature(self, s):
            return 0.0
    assert corner_speed_limit(Straight(), 0.0, ASSIST) == math.inf


def test_limit_inside_the_corner_is_the_grip_speed():
    v_corner = math.sqrt(ASSIST["lateral_g"] * 9.81 / 0.01)
    assert corner_speed_limit(OneCorner(), 120.0, ASSIST) == pytest.approx(v_corner, rel=0.01)


def test_limit_grows_with_braking_distance():
    near = corner_speed_limit(OneCorner(), 80.0, ASSIST)
    far = corner_speed_limit(OneCorner(), 0.0, ASSIST)
    assert far > near > corner_speed_limit(OneCorner(), 120.0, ASSIST)


def test_assist_only_slows_down():
    assert apply_corner_assist(0.6, 0.0, 20.0, 30.0, ASSIST) == (0.6, 0.0)   # well under
    assert apply_corner_assist(0.6, 0.0, 29.5, 30.0, ASSIST) == (0.0, 0.0)   # close: coast
    thr, brk = apply_corner_assist(0.6, 0.0, 32.0, 30.0, ASSIST)             # over: brake
    assert thr == 0.0 and brk > 0.0


def test_player_fist_brake_is_kept():
    assert apply_corner_assist(0.0, 1.0, 20.0, 30.0, ASSIST) == (0.0, 1.0)


def test_real_track_limits_are_sensible():
    track = Track(CFG["track"])
    limits = [corner_speed_limit(track, s, ASSIST) for s in range(0, int(track.length), 20)]
    finite = [v for v in limits if v != math.inf]  # long straights may see no corner yet
    assert len(finite) > len(limits) // 2
    assert min(finite) > 15                        # every corner can be taken above 54 km/h
    assert min(finite) < 40                        # but the corners do slow the car down
