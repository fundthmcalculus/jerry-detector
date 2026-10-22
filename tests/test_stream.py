import os

import cv2
import numpy as np

from jerry.config import StreamConfig
from jerry.stream import CvReader, _redact, _set_capture_options, open_reader


def write_video(path, frames=30, size=(320, 180)):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), i * 8 % 256, np.uint8))
    out.release()


def test_reader_downscales_and_samples(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=30)
    reader = open_reader(str(video), fps=1e9, cfg=StreamConfig(frame_width=160))
    first = reader.read()[1]
    assert first.shape == (90, 160, 3)
    n = 1
    while reader.read() is not None:
        n += 1
    assert n == 30  # fps high enough that no frame is skipped
    reader.close()


def test_reader_drops_frames_above_sampling_rate(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=30)
    reader = CvReader(str(video), fps=1.0, cfg=StreamConfig())
    assert reader.read() is not None
    assert reader.read() is None  # 1fps of a 30-frame file at 30fps: only frame 0 is sampled
    reader.close()


def test_unopenable_source_raises(tmp_path):
    try:
        open_reader(str(tmp_path / "missing.avi"), 5, StreamConfig())
    except OSError:
        return
    raise AssertionError("expected OSError")


def test_capture_options_set_user_agent():
    _set_capture_options("Test/1.0")
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "user_agent;Test/1.0"


def test_redact_drops_credentials():
    assert _redact("rtsp://user:pw@10.0.0.1:554/s") == "rtsp://10.0.0.1:554/s"
    assert _redact("https://host/a.m3u8") == "https://host/a.m3u8"


def test_samples_by_frame_count_not_wall_clock(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=30)  # written at 30fps
    reader = CvReader(str(video), fps=10, cfg=StreamConfig())
    n = 0
    while reader.read() is not None:
        n += 1
    reader.close()
    assert n == 10  # every 3rd frame, regardless of how fast the file decodes


def test_frames_carry_stream_time(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=30)
    reader = CvReader(str(video), fps=10, cfg=StreamConfig())
    times = []
    while (item := reader.read()) is not None:
        times.append(item[0])
    reader.close()
    assert times == [i / 10 for i in range(10)]  # 0.1s apart however fast they were decoded
