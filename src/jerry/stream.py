from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from typing import Protocol
from urllib.parse import urlparse

import cv2
import numpy as np

from jerry.config import StreamConfig

log = logging.getLogger(__name__)


class FrameReader(Protocol):
    def read(self) -> tuple[float, np.ndarray] | None:
        """Next (stream_seconds, BGR frame) at the sampling rate, or None when the stream ended or dropped.

        Time is the frame's position in the stream, not arrival time: HLS arrives in bursts."""
        ...

    def close(self) -> None: ...


def _set_capture_options(user_agent: str) -> None:
    """Make OpenCV's FFmpeg send `user_agent`; the camera relay answers its default `Lavf` one with 405.

    OpenCV reads OPENCV_FFMPEG_CAPTURE_OPTIONS from the C runtime's environment. On Windows the
    FFmpeg plugin (a MinGW build linked to msvcrt.dll) has its own copy that `os.environ` does not
    update, so also call msvcrt._putenv.
    """
    value = f"user_agent;{user_agent}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = value
    if sys.platform == "win32":
        ctypes.CDLL("msvcrt")._putenv(f"OPENCV_FFMPEG_CAPTURE_OPTIONS={value}".encode())


class CvReader:
    """OpenCV capture (HLS/HTTP, RTSP or a file), throttled to the sampling rate and downscaled."""

    def __init__(self, source: str, fps: float, cfg: StreamConfig) -> None:
        _set_capture_options(cfg.user_agent)
        timeout_ms = int(cfg.timeout_seconds * 1000)
        self._cap = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
        )
        if not self._cap.isOpened():
            self._cap.release()
            raise OSError(f"could not open {_redact(source)}")
        self._width = cfg.frame_width
        # Sample by counting frames: HLS arrives a segment at a time in bursts, so a wall-clock
        # throttle would keep one frame per burst. Fall back to wall-clock if the rate is unknown.
        source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = source_fps
        self._step = max(1, round(source_fps / fps)) if source_fps >= 1 else None
        self._interval = 1.0 / fps
        self._next_at = 0.0
        self._count = 0

    def read(self) -> tuple[float, np.ndarray] | None:
        # Every frame is decoded to stay live; only sampled ones are returned.
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return None
            index, self._count = self._count, self._count + 1
            if self._step is not None:
                if index % self._step == 0:
                    return index / self._source_fps, self._downscale(bgr)
                continue
            now = time.monotonic()
            if now >= self._next_at:
                self._next_at = now + self._interval
                return now, self._downscale(bgr)

    def close(self) -> None:
        self._cap.release()

    def _downscale(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if w <= self._width:
            return bgr
        return cv2.resize(bgr, (self._width, round(h * self._width / w)), interpolation=cv2.INTER_AREA)


def open_reader(source: str, fps: float, cfg: StreamConfig) -> FrameReader:
    """Raises OSError if the source can't be opened."""
    return CvReader(source, fps, cfg)


def _redact(source: str) -> str:
    """Drop credentials from a URL before logging it."""
    p = urlparse(source)
    return p._replace(netloc=p.netloc.rpartition("@")[2]).geturl() if "@" in p.netloc else source
