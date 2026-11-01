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
