"""Hand gesture detection from MediaPipe landmarks."""
import math
from typing import Any

WRIST = 0
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky (thumb ignored)
FINGER_PIPS = (6, 10, 14, 18)  # the middle joint of each of those fingers


def _dist(a: Any, b: Any) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def count_extended_fingers(hand: Any) -> int:
    """Count extended fingers (thumb excluded).

    A finger counts as extended when its tip is further from the wrist than
    its PIP joint. `hand` is a MediaPipe NormalizedLandmarkList (or anything
    with `.landmark[i].x/.y`).
    """
    lm = hand.landmark
    wrist = lm[WRIST]
    return sum(1 for tip, pip_ in zip(FINGER_TIPS, FINGER_PIPS)
               if _dist(lm[tip], wrist) > _dist(lm[pip_], wrist))


def is_fist(hand: Any, max_extended: int = 1) -> bool:
    """True if at most `max_extended` fingers are extended."""
    return count_extended_fingers(hand) <= max_extended
