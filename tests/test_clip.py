from jerry.clip import ClipBuffer


def test_clip_has_pre_and_post_roll():
    buf = ClipBuffer(pre_seconds=2, post_seconds=2)
    for t in range(6):  # 1 fps, pre-roll keeps t=3,4,5
        assert buf.push(float(t), f"f{t}") is None

    buf.trigger(6.0)
    assert buf.push(6.0, "f6") is None
    assert buf.push(7.0, "f7") is None
    assert buf.push(8.0, "f8") == ["f3", "f4", "f5", "f6", "f7", "f8"]


def test_retrigger_extends_post_roll():
    buf = ClipBuffer(pre_seconds=0, post_seconds=2)
    buf.trigger(0.0)
    buf.push(0.0, "a")
    buf.trigger(1.5)  # second event during recording extends the clip
    assert buf.push(2.0, "b") is None
    assert buf.push(3.5, "c") == ["a", "b", "c"]


def test_no_trigger_no_clip():
    buf = ClipBuffer(pre_seconds=1, post_seconds=1)
    assert all(buf.push(float(t), t) is None for t in range(10))
