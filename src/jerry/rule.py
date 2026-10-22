from __future__ import annotations

from jerry.detect import Person


class CrashRule:
    """Fires when the primary skier's box turns horizontal and stays that way.

    Placeholder heuristic: the largest person in frame is the skier, and "down"
    means width/height >= horizontal_ratio for hold_seconds. Tune on real footage.
    """

    def __init__(self, horizontal_ratio: float, hold_seconds: float, cooldown_seconds: float) -> None:
        self._ratio = horizontal_ratio
        self._hold = hold_seconds
        self._cooldown = cooldown_seconds
        self._down_since: float | None = None
        self._last_fired = float("-inf")

    def update(self, people: list[Person], now: float) -> bool:
        """Feed one frame's detections. Returns True when a crash fires."""
        subject = max(people, key=lambda p: p.width * p.height, default=None)
        if subject is None or subject.height <= 0 or subject.width / subject.height < self._ratio:
            self._down_since = None
            return False

        if self._down_since is None:
            self._down_since = now
        if now - self._down_since < self._hold or now - self._last_fired < self._cooldown:
            return False

        self._last_fired = now
        self._down_since = None
        return True
