"""MediaPipe Hands wrapper: finds the two hands and their palm centers."""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


@dataclass
class TwoHands:
    """Both hands, ordered by their position on screen."""
    left: Any                 # MediaPipe NormalizedLandmarkList
    right: Any
    left_palm: np.ndarray     # palm center in pixels (x, y)
    right_palm: np.ndarray


class HandTracker:
    """Runs MediaPipe Hands on BGR frames. Use as a context manager."""

    def __init__(self, max_num_hands: int, model_complexity: int,
                 min_detection_confidence: float, min_tracking_confidence: float,
                 palm_landmark: int) -> None:
        self.palm_landmark = palm_landmark
        self._hands = mp_hands.Hands(max_num_hands=max_num_hands,
                                     model_complexity=model_complexity,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def palm_center(self, hand: Any, w: int, h: int) -> np.ndarray:
        """Palm landmark of `hand` in pixel coordinates."""
        lm = hand.landmark[self.palm_landmark]
        return np.array([lm.x * w, lm.y * h])

    def find_two_hands(self, frame_bgr: np.ndarray) -> Optional[TwoHands]:
        """Return both hands, or None unless exactly two are visible."""
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        found = res.multi_hand_landmarks
        if not found or len(found) != 2:
            return None
        # sort by screen x instead of trusting MediaPipe's handedness label,
        # which can flip when hands are close together
        left, right = sorted(found, key=lambda hand: hand.landmark[self.palm_landmark].x)
        h, w = frame_bgr.shape[:2]
        return TwoHands(left, right, self.palm_center(left, w, h), self.palm_center(right, w, h))


def palm_points(hands: TwoHands) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Integer pixel coordinates of both palms, ready for OpenCV drawing."""
    return tuple(hands.left_palm.astype(int)), tuple(hands.right_palm.astype(int))
