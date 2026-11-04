"""
Virtual Steering Wheel
Drive a racing game with two hands in front of the webcam.

Keys:  k     = guided calibration (straight, full left, full right)
       c     = quick re-center (hold your hands in the "wheel straight" position)
       g     = toggle sending inputs to the controller
       SPACE = skip the rest of the guided calibration
       q     = quit

Options:
       --calibrate                 force the guided calibration on start
       --output {auto,gamepad,keyboard,none,game}   override controller.output
                                   (game = built-in PyGame track, no external game needed)
"""
import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from calibration import CalibrationSession, load_calibration, save_calibration
from camera_worker import CameraWorker
from controller import Controller, create_controller
from display import QUIT_KEY, CvWindow, Display
from gestures import is_fist
from hand_tracker import HandTracker, TwoHands
from metrics import FrameTimer
from overlay import draw_calibration, draw_hands, draw_hud, draw_message
from steering import SteeringProcessor, steering_angle_deg

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
KEY_SPACE = 32


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Read the YAML config into a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam hand-tracking steering wheel")
    p.add_argument("--calibrate", action="store_true",
                   help="run the guided calibration even if a saved one exists")
    p.add_argument("--output", choices=["auto", "gamepad", "keyboard", "none", "game"],
                   help="controller output (overrides config.yaml)")
    return p.parse_args()


def open_camera(cfg: Dict[str, Any]) -> cv2.VideoCapture:
    """Open the webcam through DirectShow with the configured resolution."""
    cap = cv2.VideoCapture(cfg["index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg["buffer_size"])
    return cap


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.output:
        cfg["controller"]["output"] = args.output
    cap = open_camera(cfg["camera"])
    if not cap.isOpened():
        print(f"[!] Could not open camera {cfg['camera']['index']}. Try another "
              "camera.index in config.yaml or close apps that use the camera.")
        return

    controller: Controller
    display: Display
    if cfg["controller"]["output"] == "game":
        from game.app import RacingGame  # imported here: pygame is only needed in this mode
        game = RacingGame(cfg["game"])
        controller, display = game, game  # the game is both the output and the window
    else:
        controller = create_controller(cfg["controller"])
        display = CvWindow(cfg["overlay"]["window_name"])
    try:
        run_loop(cap, controller, display, cfg, args.calibrate)
    except KeyboardInterrupt:
        pass  # Ctrl+C: exit cleanly, the finally block does the rest
    finally:
        # every exit path (q, Ctrl+C, exception) leaves the controller neutral
        controller.close()
        display.close()
        cap.release()


def new_session(cfg: Dict[str, Any]) -> CalibrationSession:
    return CalibrationSession(cfg["hold_seconds"], cfg["include_range"], cfg["min_span_deg"],
                              cfg["max_spread_deg"], cfg["max_jump_deg"], cfg["motion_frames"])


def finish_session(session: CalibrationSession, steering: SteeringProcessor, path: Path) -> None:
    """Apply and save the result; a cancelled session keeps the old calibration."""
    cal = session.result()
    if cal is None:
        print("[i] Calibration cancelled, keeping the previous one.")
        return
    steering.apply_calibration(cal)
    save_calibration(cal, path)
    spans = ("default range" if cal.left_span_deg is None else
             f"left {cal.left_span_deg:.1f} deg, right {cal.right_span_deg:.1f} deg")
    print(f"[i] Calibrated: neutral {cal.neutral_deg:+.1f} deg, {spans}. Saved to {path.name}")


def run_loop(cap: cv2.VideoCapture, controller: Controller, display: Display,
             cfg: Dict[str, Any], force_calibration: bool = False) -> None:
    """Take tracked frames from the camera thread -> steer -> send -> draw, until quit.

    The display is refreshed at its own rate (`display.refresh_interval`):
    the game redraws smoothly at 60 FPS even when hand tracking is slower,
    and new steering values are used as soon as they arrive.
    """
    cam_cfg, ui_cfg, cal_cfg = cfg["camera"], cfg["overlay"], cfg["calibration"]
    steering = SteeringProcessor(**cfg["steering"])
    max_extended = cfg["gestures"]["max_extended_fingers_for_fist"]
    auto_throttle = cfg["controller"]["auto_throttle"]
    timer = FrameTimer(ui_cfg["fps_smoothing"])
    sending = controller.available and controller.starts_enabled

    # reuse the last calibration if there is one; otherwise guide the user
    cal_path = PROJECT_DIR / cal_cfg["file"]
    saved = load_calibration(cal_path)
    if saved is not None:
        steering.apply_calibration(saved)
        print(f"[i] Loaded calibration from {cal_path.name} (press k to redo it).")
    session: Optional[CalibrationSession] = None
    if force_calibration or saved is None:
        session = new_session(cal_cfg)

    # what is on screen until the first camera frame arrives
    shown = np.zeros((cam_cfg["height"], cam_cfg["width"], 3), np.uint8)
    draw_message(shown, "Starting camera...")
    hands: Optional[TwoHands] = None
    refresh = display.refresh_interval or ui_cfg["camera_wait"]
    last_show = time.perf_counter()

    with HandTracker(**cfg["tracking"]) as tracker:
        worker = CameraWorker(cap, tracker, cam_cfg["mirror"], cam_cfg["max_read_failures"])
        worker.start()
        try:
            while True:
                # wait for a new camera result, but never past the next redraw
                wait = max(0.0, refresh - (time.perf_counter() - last_show))
                tracked = worker.get(wait)
                if worker.failed:
                    print("[!] The camera stopped sending frames.")
                    break
                if tracked is not None:
                    frame, hands = tracked.frame, tracked.hands
                    t0 = timer.start(tracked.t_capture)
                    steer, throttle, brake = 0.0, 0.0, 0.0  # neutral unless driving
                    if hands is not None:
                        draw_hands(frame, hands)

                    if session is not None:
                        # calibrating: measure the raw angle, never drive the car
                        angle = (steering_angle_deg(hands.left_palm, hands.right_palm)
                                 if hands is not None else None)
                        session.update(angle, t0)
                    elif hands is not None:
                        steer = steering.update(hands.left_palm, hands.right_palm, t0)
                        braking = (is_fist(hands.left, max_extended)
                                   or is_fist(hands.right, max_extended))
                        brake = 1.0 if braking else 0.0
                        throttle = 0.0 if braking else auto_throttle

                    if sending:
                        controller.send(steer, throttle, brake)
                    timer.stop()

                    if not controller.available:
                        status = controller.name
                    else:
                        status = f"{controller.name} {'ON' if sending else 'OFF'}"
                    if session is not None:
                        draw_calibration(frame, session.index + 1, len(session.steps),
                                         session.step.instruction, session.progress,
                                         session.message, ui_cfg)
                        if hands is None:
                            draw_message(frame, "Show both hands", y=ui_cfg["panel_height"] + 30)
                    else:
                        draw_hud(frame, steer, throttle, brake, timer.fps, timer.latency_ms,
                                 status, steering.max_angle_deg, ui_cfg)
                        if hands is None:
                            draw_message(frame, "Show both hands")
                    shown = frame

                last_show = time.perf_counter()
                key = display.show(shown, large=session is not None)
                if key in (QUIT_KEY, ord("q")):
                    break
                if key == ord("g") and controller.available:
                    sending = not sending
                    if not sending:
                        controller.neutral()  # don't leave the last input stuck in the game
                if session is not None:
                    if key == KEY_SPACE:
                        session.skip()
                    if session.done:
                        finish_session(session, steering, cal_path)
                        session = None
                elif key == ord("k"):
                    session = new_session(cal_cfg)
                elif key == ord("c") and hands is not None and steering.calibrate():
                    save_calibration(steering.current_calibration(), cal_path)  # keeps the range
        finally:
            worker.stop()  # before the tracker closes: the thread may still be using it


if __name__ == "__main__":
    main()
