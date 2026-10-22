from __future__ import annotations

from collections import deque

import numpy as np


class ClipBuffer:
    """Rolling pre-roll buffer that keeps collecting frames as post-roll after a trigger."""

    def __init__(self, pre_seconds: float, post_seconds: float) -> None:
        self._pre = pre_seconds
        self._post = post_seconds
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._clip: list[np.ndarray] | None = None
        self._trigger_ts = 0.0

    def trigger(self, now: float) -> None:
        """Start a clip (or extend the one in progress) from the pre-roll already buffered."""
        if self._clip is None:
            self._clip = [frame for _, frame in self._history]
        self._trigger_ts = now

    def push(self, now: float, frame: np.ndarray) -> list[np.ndarray] | None:
        """Add a frame. Returns the finished clip once post-roll is complete, else None."""
        self._history.append((now, frame))
        while self._history and self._history[0][0] < now - self._pre:
            self._history.popleft()

        if self._clip is None:
            return None
        self._clip.append(frame)
        if now - self._trigger_ts >= self._post:
            done, self._clip = self._clip, None
            return done
        return None
