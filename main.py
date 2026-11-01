"""
Virtual Steering Wheel
Drive a racing game with two hands in front of the webcam.

Keys:  c = calibrate (hold your hands in the "wheel straight" position)
       g = toggle sending inputs to the controller
       q = quit
"""
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml

from controller import Controller, create_controller
from gestures import is_fist
from hand_tracker import HandTracker
from metrics import FrameTimer
from overlay import draw_hands, draw_hud, draw_message
from steering import SteeringProcessor

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Read the YAML config into a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_camera(cfg: Dict[str, Any]) -> cv2.VideoCapture:
    """Open the webcam through DirectShow with the configured resolution."""
    cap = cv2.VideoCapture(cfg["index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg["buffer_size"])
    return cap


def main() -> None:
    cfg = load_config()
    cap = open_camera(cfg["camera"])
    if not cap.isOpened():
        print(f"[!] Could not open camera {cfg['camera']['index']}. Try another "
              "camera.index in config.yaml or close apps that use the camera.")
        return

    controller = create_controller()
    try:
        run_loop(cap, controller, cfg)
    except KeyboardInterrupt:
        pass  # Ctrl+C: exit cleanly, the finally block does the rest
    finally:
        # every exit path (q, Ctrl+C, exception) leaves the controller neutral
        controller.close()
        cap.release()
        cv2.destroyAllWindows()


def run_loop(cap: cv2.VideoCapture, controller: Controller, cfg: Dict[str, Any]) -> None:
    """Capture -> track -> steer -> send -> draw, until the user quits."""
    cam_cfg, ui_cfg = cfg["camera"], cfg["overlay"]
    steering = SteeringProcessor(**cfg["steering"])
    max_extended = cfg["gestures"]["max_extended_fingers_for_fist"]
    auto_throttle = cfg["controller"]["auto_throttle"]
    window = ui_cfg["window_name"]
    timer = FrameTimer(ui_cfg["fps_smoothing"])
    sending = controller.available
    read_failures = 0

    with HandTracker(**cfg["tracking"]) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                # some cameras return a few empty frames at startup (DirectShow)
                read_failures += 1
                if read_failures >= cam_cfg["max_read_failures"]:
                    print("[!] The camera stopped sending frames.")
                    break
                continue
            read_failures = 0
            t0 = timer.start()
            if cam_cfg["mirror"]:
                frame = cv2.flip(frame, 1)  # mirror view feels natural

            hands = tracker.find_two_hands(frame)
            if hands is not None:
                draw_hands(frame, hands)
                steer = steering.update(hands.left_palm, hands.right_palm, t0)
                braking = is_fist(hands.left, max_extended) or is_fist(hands.right, max_extended)
                brake = 1.0 if braking else 0.0
                throttle = 0.0 if braking else auto_throttle
            else:
                # hands not visible: go neutral so nothing stays stuck
                steer, throttle, brake = 0.0, 0.0, 0.0
                draw_message(frame, "Show both hands")

            if sending:
                controller.send(steer, throttle, brake)

            timer.stop()
            if not controller.available:
                status = controller.name
            else:
                status = f"{controller.name} {'ON' if sending else 'OFF'}"
            draw_hud(frame, steer, throttle, brake, timer.fps, timer.proc_ms, status,
                     steering.max_angle_deg, ui_cfg)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # window X button: otherwise imshow would just reopen the window
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("g") and controller.available:
                sending = not sending
                if not sending:
                    controller.neutral()  # don't leave the last input stuck in the game
            if key == ord("c") and hands is not None:
                steering.calibrate()


if __name__ == "__main__":
    main()
