from __future__ import annotations

import random
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from jerry.config import ScheduleConfig


class OperatingHours:
    """Daily open/close window in a fixed timezone. close <= open means it wraps past midnight."""

    def __init__(self, cfg: ScheduleConfig) -> None:
        self._tz = ZoneInfo(cfg.timezone)
        self._open = time.fromisoformat(cfg.open)
        self._close = time.fromisoformat(cfg.close)

    def now(self) -> datetime:
        return datetime.now(self._tz)

    def is_open(self, at: datetime) -> bool:
        t = at.astimezone(self._tz).time()
        if self._open == self._close:
            return True  # a 24h window
        if self._open < self._close:
            return self._open <= t < self._close
        return t >= self._open or t < self._close

    def seconds_until_open(self, at: datetime) -> float:
        if self.is_open(at):
            return 0.0
        local = at.astimezone(self._tz)
        opens = datetime.combine(local.date(), self._open, tzinfo=self._tz)
        if opens <= local:
            opens += timedelta(days=1)
        return (opens - local).total_seconds()


def next_spot_check_delay(cfg: ScheduleConfig, rng: random.Random) -> float:
    """Seconds until the next off-hours spot check, uniformly random so it can't be predicted."""
    return rng.uniform(cfg.spot_check_min_minutes, cfg.spot_check_max_minutes) * 60.0
