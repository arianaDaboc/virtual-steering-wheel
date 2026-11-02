"""Controller outputs. Every output takes the same (steer, throttle, brake) values."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, FrozenSet, Optional, Set

try:
    import vgamepad as vg
    HAS_VGAMEPAD = True
except Exception:  # not installed, or its bundled client DLL failed to load
    HAS_VGAMEPAD = False

try:
    import pydirectinput
    # pydirectinput sleeps 0.1 s after every call by default, which would cap
    # the whole loop at a few FPS. We call it at most a few times per frame.
    pydirectinput.PAUSE = 0
    # The "mouse in a screen corner" failsafe would raise mid-loop and again
    # while releasing keys on exit; our own neutral-on-exit does that job.
    pydirectinput.FAILSAFE = False
    HAS_PYDIRECTINPUT = True
except Exception:  # not installed, or not on Windows
    HAS_PYDIRECTINPUT = False

VIGEMBUS_URL = "https://github.com/nefarius/ViGEmBus/releases"


class Controller(ABC):
    """Common interface: steer in [-1, 1], throttle and brake in [0, 1]."""

    name = "controller"
    available = True
    starts_enabled = True  # send inputs right away, or wait for the `g` key

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


def keys_for(steer: float, throttle: float, brake: float, keys: Dict[str, str],
             steer_threshold: float, pedal_threshold: float) -> FrozenSet[str]:
    """Which keys should be held down for these inputs.

    Keys are on/off, so the analog values are cut at a threshold. Brake wins
    over throttle so the car can never get both at once.
    """
    wanted: Set[str] = set()
    if steer >= steer_threshold:
        wanted.add(keys["right"])
    elif steer <= -steer_threshold:
        wanted.add(keys["left"])
    if brake >= pedal_threshold:
        wanted.add(keys["brake"])
    elif throttle >= pedal_threshold:
        wanted.add(keys["throttle"])
    return frozenset(wanted)


class KeyboardController(Controller):
    """Fallback output: holds keys (WASD by default) with pydirectinput.

    Only key changes are sent (press when a key becomes wanted, release when
    it stops being wanted), never a press per frame, so keys don't "stutter".
    """

    name = "KEYBOARD"
    # Keys go to whichever window has focus, so start OFF: the user focuses
    # the game first, then presses `g` in the camera window.
    starts_enabled = False

    def __init__(self, keys: Dict[str, str], steer_threshold: float, pedal_threshold: float,
                 key_down: Optional[Callable[[str], Any]] = None,
                 key_up: Optional[Callable[[str], Any]] = None) -> None:
        self.keys = keys
        self.steer_threshold = steer_threshold
        self.pedal_threshold = pedal_threshold
        # injectable so tests can record key events without a real keyboard
        self._down = key_down or pydirectinput.keyDown
        self._up = key_up or pydirectinput.keyUp
        self.held: Set[str] = set()

    def send(self, steer: float, throttle: float, brake: float) -> None:
        wanted = keys_for(steer, throttle, brake, self.keys,
                          self.steer_threshold, self.pedal_threshold)
        for key in self.held - wanted:
            self._up(key)
        for key in wanted - self.held:
            self._down(key)
        self.held = set(wanted)


class NullController(Controller):
    """Does nothing; used when no real output is available."""

    name = "NO OUTPUT"
    available = False

    def send(self, steer: float, throttle: float, brake: float) -> None:
        pass


def _try_gamepad() -> Optional[Controller]:
    if not HAS_VGAMEPAD:
        print("[!] Gamepad output: vgamepad is not installed (pip install vgamepad).")
        return None
    try:
        return GamepadController()
    except Exception as e:  # usually the ViGEmBus driver is missing
        print(f"[!] Gamepad output: could not create the virtual pad ({e}).")
        print(f"    Install the ViGEmBus driver: {VIGEMBUS_URL}")
        return None


def _try_keyboard(cfg: Dict[str, Any]) -> Optional[Controller]:
    if not HAS_PYDIRECTINPUT:
        print("[!] Keyboard output: pydirectinput is not installed "
              "(pip install pydirectinput, Windows only).")
        return None
    keys = {k: cfg[k] for k in ("left", "right", "throttle", "brake")}
    return KeyboardController(keys, cfg["steer_threshold"], cfg["pedal_threshold"])


def create_controller(cfg: Dict[str, Any]) -> Controller:
    """Pick the output from `cfg["output"]`: auto, gamepad, keyboard or none.

    `auto` tries the gamepad first (analog steering), then the keyboard.
    Whatever is missing is explained on the console.
    """
    choice = cfg.get("output", "auto")
    controller: Optional[Controller] = None
    if choice in ("auto", "gamepad"):
        controller = _try_gamepad()
    if controller is None and choice in ("auto", "keyboard"):
        if choice == "auto":
            print("[i] Falling back to keyboard output (on/off steering, no gradations).")
        controller = _try_keyboard(cfg["keyboard"])
    if controller is None:
        if choice != "none":
            print("[!] No controller output available: the wheel is only shown on screen.")
        return NullController()
    hint = "" if controller.starts_enabled else " (focus the game, then press g)"
    print(f"[i] Output: {controller.name}{hint}")
    return controller
