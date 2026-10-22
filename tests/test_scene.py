import numpy as np

from jerry.scene import SceneMonitor, change_fraction

VIEW_A = np.zeros((90, 160, 3), np.uint8)
VIEW_B = np.full((90, 160, 3), 200, np.uint8)  # a different preset


def with_skier(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[40:50, 80:84] = 255  # a few pixels, like a distant skier
    return out


def test_change_fraction():
    assert change_fraction(VIEW_A, VIEW_A) == 0
    assert change_fraction(VIEW_A, VIEW_B) == 1.0
    assert change_fraction(VIEW_A, with_skier(VIEW_A)) < 0.02


def test_skier_moving_in_a_held_view_stays_stable():
    scene = SceneMonitor(0.03, 1.5)
    assert all(scene.update(t * 0.2, with_skier(VIEW_A) if t % 2 else VIEW_A) for t in range(20))


def test_view_change_is_unstable_until_settled():
    scene = SceneMonitor(0.03, 1.5)
    assert scene.update(0.0, VIEW_A)
    assert not scene.update(0.2, VIEW_B)  # moved
    assert not scene.update(1.0, VIEW_B)  # still settling
    assert not scene.update(1.6, VIEW_B)
    assert scene.update(1.8, VIEW_B)  # 1.5s after the move


def test_each_move_restarts_settling():
    scene = SceneMonitor(0.03, 1.0)
    scene.update(0.0, VIEW_A)
    scene.update(0.2, VIEW_B)
    assert not scene.update(1.0, VIEW_A)  # second move at 1.0 extends the wait
    assert not scene.update(1.8, VIEW_A)
    assert scene.update(2.0, VIEW_A)
