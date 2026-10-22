from __future__ import annotations

import cv2
import numpy as np


def change_fraction(prev: np.ndarray, cur: np.ndarray) -> float:
    """Share of pixels that changed noticeably between two BGR frames (compared at low resolution)."""

    def small_gray(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height = max(1, round(gray.shape[0] * 160 / gray.shape[1]))
        return cv2.GaussianBlur(cv2.resize(gray, (160, height), interpolation=cv2.INTER_AREA), (5, 5), 0)

    diff = cv2.absdiff(small_gray(prev), small_gray(cur))
    return float(np.count_nonzero(diff > 25)) / diff.size


class SceneMonitor:
    """Tells whether a PTZ camera is holding one view.

    The feeds tour between presets, so the whole picture shifts for a moment (and the first
    frames after a move are soft). A skier covers a tiny share of pixels; a pan covers many.
    A frame that changes more than `change_fraction` from the previous one marks the view as
    unstable until `settle_seconds` later.
    """

    def __init__(self, change_fraction: float, settle_seconds: float) -> None:
        self._threshold = change_fraction
        self._settle = settle_seconds
        self._prev: np.ndarray | None = None
        self._unstable_until = float("-inf")

    def update(self, now: float, frame: np.ndarray) -> bool:
        """Feed each sampled frame. Returns True if the view is stable (safe to detect on)."""
        if self._prev is not None and change_fraction(self._prev, frame) >= self._threshold:
            self._unstable_until = now + self._settle
        self._prev = frame
        return now >= self._unstable_until
