
from jerry.stream import _redact, _scaled_size, is_http_stream


def test_http_sources_use_ffmpeg_others_do_not():
    assert is_http_stream("https://b15.hdrelay.com/camera/x/relay/playlist.m3u8")
    assert not is_http_stream("rtsp://u:p@10.0.0.1/stream")
    assert not is_http_stream("clips/test.mp4")
    assert not is_http_stream(r"C:\video\test.mp4")


def test_scaled_size_keeps_aspect_and_is_even():
    assert _scaled_size((1920, 1080), 960) == (960, 540)
    assert _scaled_size((1280, 720), 480) == (480, 270)
    assert _scaled_size((640, 481), 960) == (640, 480)  # never upscales


def test_redact_drops_credentials():
    assert _redact("rtsp://user:pw@10.0.0.1:554/s") == "rtsp://10.0.0.1:554/s"
    assert _redact("https://host/a.m3u8") == "https://host/a.m3u8"
