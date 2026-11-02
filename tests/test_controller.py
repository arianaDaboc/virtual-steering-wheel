from controller import Controller, NullController


class RecordingController(Controller):
    def __init__(self):
        self.sent = []

    def send(self, steer, throttle, brake):
        self.sent.append((steer, throttle, brake))


def test_neutral_sends_zeros():
    c = RecordingController()
    c.send(0.8, 0.6, 0.0)
    c.neutral()
    assert c.sent[-1] == (0.0, 0.0, 0.0)


def test_close_goes_neutral():
    c = RecordingController()
    c.send(-1.0, 0.0, 1.0)
    c.close()
    assert c.sent[-1] == (0.0, 0.0, 0.0)


def test_null_controller_is_unavailable_and_safe():
    c = NullController()
    assert not c.available
    c.send(1.0, 1.0, 1.0)
    c.close()


# ---- keyboard fallback ----
from controller import KeyboardController, keys_for  # noqa: E402

KEYS = {"left": "a", "right": "d", "throttle": "w", "brake": "s"}


def test_keys_for_thresholds():
    assert keys_for(0.1, 0.0, 0.0, KEYS, 0.25, 0.5) == set()
    assert keys_for(0.3, 0.6, 0.0, KEYS, 0.25, 0.5) == {"d", "w"}
    assert keys_for(-0.9, 0.0, 0.0, KEYS, 0.25, 0.5) == {"a"}


def test_brake_wins_over_throttle():
    assert keys_for(0.0, 1.0, 1.0, KEYS, 0.25, 0.5) == {"s"}


def make_keyboard():
    events = []
    kb = KeyboardController(KEYS, 0.25, 0.5,
                            key_down=lambda k: events.append(("down", k)),
                            key_up=lambda k: events.append(("up", k)))
    return kb, events


def test_keyboard_sends_only_changes():
    kb, events = make_keyboard()
    kb.send(0.5, 0.6, 0.0)
    kb.send(0.6, 0.6, 0.0)   # same keys: nothing new is sent
    assert sorted(events) == [("down", "d"), ("down", "w")]


def test_keyboard_switches_direction():
    kb, events = make_keyboard()
    kb.send(0.5, 0.0, 0.0)
    events.clear()
    kb.send(-0.5, 0.0, 0.0)
    assert events == [("up", "d"), ("down", "a")]


def test_keyboard_close_releases_everything():
    kb, events = make_keyboard()
    kb.send(-0.8, 0.0, 1.0)
    kb.close()
    assert kb.held == set()
    assert sorted(e for e in events if e[0] == "up") == [("up", "a"), ("up", "s")]


def test_keyboard_starts_disabled():
    assert KeyboardController.starts_enabled is False
