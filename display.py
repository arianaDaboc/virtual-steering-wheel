"""Where the camera frame is shown: an OpenCV window, or the game window."""
from typing import Optional, Protocol

import cv2
import numpy as np

QUIT_KEY = -2  # returned by show() when the user closes the window


class Display(Protocol):
    # seconds between redraws, or None = redraw only when a new camera frame arrives
    refresh_interval: Optional[float]

    def show(self, frame: np.ndarray, large: bool = False) -> int:
        """Show the frame and return the pressed key code (or QUIT_KEY)."""

    def close(self) -> None:
        """Close the window."""


class CvWindow:
    """Plain OpenCV window; `large` is ignored because the frame is always full size."""

    refresh_interval: Optional[float] = None  # nothing changes between camera frames

    def __init__(self, name: str) -> None:
        self.name = name

    def show(self, frame: np.ndarray, large: bool = False) -> int:
        cv2.imshow(self.name, frame)
        key = cv2.waitKey(1) & 0xFF
        # window X button: otherwise imshow would just reopen the window
        if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            return QUIT_KEY
        return key

    def close(self) -> None:
        cv2.destroyAllWindows()
