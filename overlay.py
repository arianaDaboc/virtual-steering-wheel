"""All OpenCV drawing for the camera window."""
import math
from typing import Any, Dict

import cv2
import mediapipe as mp
import numpy as np

from hand_tracker import TwoHands, palm_points

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 220)
HUD_GREEN = (0, 255, 0)
WARN_RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_mp_draw = mp.solutions.drawing_utils
_connections = mp.solutions.hands.HAND_CONNECTIONS


def draw_hands(frame: np.ndarray, hands: TwoHands) -> None:
    """Draw both hand skeletons and the steering line between the palms."""
    lp, rp = palm_points(hands)
    cv2.line(frame, lp, rp, YELLOW, 3)
    for hand in (hands.left, hands.right):
        _mp_draw.draw_landmarks(frame, hand, _connections)


def draw_message(frame: np.ndarray, text: str) -> None:
    """Show a warning line under the HUD."""
    cv2.putText(frame, text, (10, 55), FONT, 0.7, WARN_RED, 2)


def _draw_bar(frame: np.ndarray, x: int, bottom: int, value: float,
              color: tuple, cfg: Dict[str, Any]) -> None:
    w, h = cfg["bar_width"], cfg["bar_height"]
    cv2.rectangle(frame, (x, bottom - h), (x + w, bottom), WHITE, 1)
    cv2.rectangle(frame, (x, int(bottom - h * value)), (x + w, bottom), color, -1)


def draw_hud(frame: np.ndarray, steer: float, throttle: float, brake: float,
             fps: float, proc_ms: float, status: str,
             max_angle_deg: float, cfg: Dict[str, Any]) -> None:
    """Draw the wheel indicator, throttle/brake bars and the status line."""
    h, w = frame.shape[:2]
    r = cfg["wheel_radius"]
    cx, cy = w // 2, h - cfg["wheel_bottom_margin"]
    cv2.circle(frame, (cx, cy), r, WHITE, 3)
    # -90 so the needle points straight up when steer = 0
    ang = math.radians(steer * max_angle_deg * cfg["wheel_rotation_scale"] - 90)
    cv2.line(frame, (cx, cy), (int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))),
             YELLOW, 4)

    margin = cfg["bar_margin"]
    bottom = h - margin
    _draw_bar(frame, margin, bottom, throttle, GREEN, cfg)
    _draw_bar(frame, w - margin - cfg["bar_width"], bottom, brake, RED, cfg)

    # "proc" is processing time only; true end-to-end latency comes with Task 6
    cv2.putText(frame, f"FPS {fps:4.1f} | proc {proc_ms:4.1f} ms | steer {steer:+.2f} | {status}",
                (10, 25), FONT, 0.55, HUD_GREEN, 2)
