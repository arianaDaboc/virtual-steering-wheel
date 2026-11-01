from types import SimpleNamespace

from gestures import count_extended_fingers, is_fist


def make_hand(extended):
    """Fake 21-landmark hand: wrist at the origin, fingers pointing up (-y).

    Each finger in `extended` gets its tip beyond the PIP joint;
    the others get the tip curled back toward the wrist.
    """
    lm = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    for finger, (tip, pip_) in enumerate(zip((8, 12, 16, 20), (6, 10, 14, 18))):
        x = 0.05 * finger
        lm[pip_] = SimpleNamespace(x=x, y=-0.2)
        lm[tip] = SimpleNamespace(x=x, y=-0.3 if finger in extended else -0.1)
    return SimpleNamespace(landmark=lm)


def test_open_hand_counts_four():
    assert count_extended_fingers(make_hand({0, 1, 2, 3})) == 4
    assert not is_fist(make_hand({0, 1, 2, 3}))


def test_closed_hand_is_fist():
    assert count_extended_fingers(make_hand(set())) == 0
    assert is_fist(make_hand(set()))


def test_one_finger_still_fist_by_default():
    assert is_fist(make_hand({0}))
    assert not is_fist(make_hand({0, 1}))


def test_threshold_is_configurable():
    assert not is_fist(make_hand({0}), max_extended=0)
