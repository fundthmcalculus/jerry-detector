import random
from datetime import datetime
from zoneinfo import ZoneInfo

from jerry.config import ScheduleConfig
from jerry.schedule import OperatingHours, next_spot_check_delay

NY = ZoneInfo("America/New_York")


def at(h: int, m: int = 0, day: int = 15, month: int = 1) -> datetime:
    return datetime(2027, month, day, h, m, tzinfo=NY)


def test_open_window_edges():
    hours = OperatingHours(ScheduleConfig())
    assert not hours.is_open(at(9, 29))
    assert hours.is_open(at(9, 30))
    assert hours.is_open(at(21, 29))
    assert not hours.is_open(at(21, 30))
    assert not hours.is_open(at(3))


def test_other_timezones_are_converted():
    hours = OperatingHours(ScheduleConfig())
    assert hours.is_open(datetime(2027, 1, 15, 15, 0, tzinfo=ZoneInfo("UTC")))  # 10:00 EST
    assert not hours.is_open(datetime(2027, 1, 15, 14, 0, tzinfo=ZoneInfo("UTC")))  # 09:00 EST


def test_follows_dst():
    hours = OperatingHours(ScheduleConfig())
    # 13:45 UTC is 09:45 EDT in summer (open) but 08:45 EST in winter (closed)
    assert hours.is_open(datetime(2027, 7, 15, 13, 45, tzinfo=ZoneInfo("UTC")))
    assert not hours.is_open(datetime(2027, 1, 15, 13, 45, tzinfo=ZoneInfo("UTC")))


def test_seconds_until_open():
    hours = OperatingHours(ScheduleConfig())
    assert hours.seconds_until_open(at(12)) == 0
    assert hours.seconds_until_open(at(9, 0)) == 30 * 60
    assert hours.seconds_until_open(at(22, 30)) == 11 * 3600  # rolls to tomorrow


def test_window_wrapping_midnight():
    hours = OperatingHours(ScheduleConfig(open="18:00", close="02:00"))
    assert hours.is_open(at(23))
    assert hours.is_open(at(1))
    assert not hours.is_open(at(12))


def test_equal_open_close_is_always_open():
    hours = OperatingHours(ScheduleConfig(open="09:30", close="09:30"))
    assert hours.is_open(at(3)) and hours.is_open(at(12))


def test_spot_check_delay_is_random_within_bounds():
    cfg = ScheduleConfig(spot_check_min_minutes=20, spot_check_max_minutes=60)
    rng = random.Random(1)
    delays = [next_spot_check_delay(cfg, rng) for _ in range(200)]
    assert all(20 * 60 <= d <= 60 * 60 for d in delays)
    assert len(set(delays)) > 100
