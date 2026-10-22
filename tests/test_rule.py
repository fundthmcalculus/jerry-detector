from jerry.detect import Person
from jerry.rule import CrashRule

UPRIGHT = Person(box=(0, 0, 50, 100), score=1.0)  # w/h = 0.5
FLAT = Person(box=(0, 0, 150, 100), score=1.0)  # w/h = 1.5


def test_no_people_never_fires():
    rule = CrashRule(1.3, 1.0, 10.0)
    assert not rule.update([], 0.0)


def test_upright_skier_never_fires():
    rule = CrashRule(1.3, 1.0, 10.0)
    assert not any(rule.update([UPRIGHT], t) for t in range(20))


def test_fires_after_hold_then_respects_cooldown():
    rule = CrashRule(1.3, 1.0, 10.0)
    assert not rule.update([FLAT], 1.0)  # starts the hold
    assert not rule.update([FLAT], 1.5)
    assert rule.update([FLAT], 2.0)  # held for 1s
    assert not rule.update([FLAT], 2.5)  # hold restarts after firing
    assert not rule.update([FLAT], 3.5)  # held, but inside cooldown
    assert rule.update([FLAT], 12.5)  # cooldown over, still down


def test_getting_up_resets_hold():
    rule = CrashRule(1.3, 1.0, 10.0)
    rule.update([FLAT], 0.0)
    rule.update([UPRIGHT], 0.5)
    assert not rule.update([FLAT], 1.2)  # hold started again at 1.2
    assert rule.update([FLAT], 2.2)
