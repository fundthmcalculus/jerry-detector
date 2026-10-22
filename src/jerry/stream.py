from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Protocol
from urllib.parse import urlparse

import cv2
import numpy as np

from jerry.config import StreamConfig

log = logging.getLogger(__name__)


class FrameReader(Protocol):
    def read(self) -> np.ndarray | None:
        """Next BGR frame at the sampling rate, or None when the stream ended or dropped."""
        ...

    def close(self) -> None: ...


def is_http_stream(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


class FfmpegReader:
    """Reads an HTTP/HLS stream through an ffmpeg subprocess, already sampled and downscaled.

    OpenCV's bundled FFmpeg can't set the User-Agent, and the camera relay answers its
    default `Lavf` one with 405. Going through ffmpeg also gives reconnects, a read timeout,
    and keeps only the video stream (the feeds carry an extra timed-ID3 data stream).
    """

    def __init__(self, source: str, fps: float, cfg: StreamConfig) -> None:
        width, height = _scaled_size(_probe_size(source, cfg), cfg.frame_width)
        self._shape = (height, width, 3)
        self._frame_bytes = height * width * 3
        timeout_us = str(int(cfg.timeout_seconds * 1_000_000))
        self._proc = subprocess.Popen(
            [
                cfg.ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
                "-user_agent", cfg.user_agent,
                "-rw_timeout", timeout_us,
                "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                "-i", source,
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", f"fps={fps},scale={width}:{height}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._log_stderr, args=(source,), daemon=True).start()

    def read(self) -> np.ndarray | None:
        data = self._proc.stdout.read(self._frame_bytes)
        if len(data) < self._frame_bytes:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(self._shape)

    def close(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self._proc.stdout.close()

    def _log_stderr(self, source: str) -> None:
        for line in self._proc.stderr:
            if b"deprecated pixel format" in line:  # harmless: feeds are full-range yuvj420p
                continue
            log.warning("ffmpeg[%s]: %s", _redact(source), line.decode(errors="replace").rstrip())


class CvReader:
    """OpenCV capture for RTSP and files, throttled to the sampling rate."""

    def __init__(self, source: str, fps: float) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise OSError(f"could not open {_redact(source)}")
        self._interval = 1.0 / fps
        self._next_at = 0.0

    def read(self) -> np.ndarray | None:
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return None
            now = time.monotonic()
            if now >= self._next_at:
                self._next_at = now + self._interval
                return bgr

    def close(self) -> None:
        self._cap.release()


def open_reader(source: str, fps: float, cfg: StreamConfig) -> FrameReader:
    """Raises OSError if the source can't be opened."""
    if is_http_stream(source):
        return FfmpegReader(source, fps, cfg)
    return CvReader(source, fps)


def _probe_size(source: str, cfg: StreamConfig) -> tuple[int, int]:
    try:
        out = subprocess.run(
            [
                cfg.ffprobe, "-v", "error", "-user_agent", cfg.user_agent,
                "-rw_timeout", str(int(cfg.timeout_seconds * 1_000_000)),
                "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x", source,
            ],
            capture_output=True, text=True, timeout=cfg.timeout_seconds * 2, check=True,
        ).stdout.strip()
        w, h = out.splitlines()[0].split("x")
        return int(w), int(h)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        raise OSError(f"could not probe {_redact(source)}: {e}") from e


def _scaled_size(size: tuple[int, int], width: int) -> tuple[int, int]:
    w, h = size
    if w <= width:
        return w - w % 2, h - h % 2
    height = round(h * width / w)
    return width, height - height % 2


def _redact(source: str) -> str:
    """Drop credentials from a URL before logging it."""
    p = urlparse(source)
    return p._replace(netloc=p.netloc.rpartition("@")[2]).geturl() if "@" in p.netloc else source
