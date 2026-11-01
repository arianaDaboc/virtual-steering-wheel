"""Controller outputs. Every output takes the same (steer, throttle, brake) values."""
from abc import ABC, abstractmethod

try:
    import vgamepad as vg
    HAS_VGAMEPAD = True
except Exception:  # not installed, or its bundled client DLL failed to load
    HAS_VGAMEPAD = False

VIGEMBUS_URL = "https://github.com/nefarius/ViGEmBus/releases"


class Controller(ABC):
    """Common interface: steer in [-1, 1], throttle and brake in [0, 1]."""

    name = "controller"
    available = True

    @abstractmethod
    def send(self, steer: float, throttle: float, brake: float) -> None:
        """Push one set of inputs to the output."""

    def neutral(self) -> None:
        """Straight wheel, no throttle, no brake."""
        self.send(0.0, 0.0, 0.0)

    def close(self) -> None:
        """Leave the output in a safe state."""
        self.neutral()


class GamepadController(Controller):
    """Virtual Xbox 360 pad via vgamepad: left stick X = steer, RT = throttle, LT = brake."""

    name = "GAMEPAD"

    def __init__(self) -> None:
        self.pad = vg.VX360Gamepad()

    def send(self, steer: float, throttle: float, brake: float) -> None:
        self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=throttle)
        self.pad.left_trigger_float(value_float=brake)
        self.pad.update()

    def neutral(self) -> None:
        self.pad.reset()  # zeroes every button and axis, not just the ones we use
        self.pad.update()


class NullController(Controller):
    """Does nothing; used when no real output is available."""

    name = "NO OUTPUT"
    available = False

    def send(self, steer: float, throttle: float, brake: float) -> None:
        pass


def create_controller() -> Controller:
    """Pick the best available output and explain on the console if one is missing."""
    if not HAS_VGAMEPAD:
        print("[!] vgamepad is not installed: pip install vgamepad")
        return NullController()
    try:
        return GamepadController()
    except Exception as e:  # usually the ViGEmBus driver is missing
        print(f"[!] Could not create the virtual gamepad ({e}).")
        print(f"    Install the ViGEmBus driver: {VIGEMBUS_URL}")
        return NullController()
