from __future__ import annotations

import os

import cv2

# FFmpeg read timeout (microseconds) so a stalled feed fails instead of hanging.
# Only takes effect if set before the first FFmpeg-backed capture is opened.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rw_timeout;10000000")


def open_capture(source: str) -> cv2.VideoCapture:
    """Open an RTSP, HTTP/HLS (.m3u8), MJPEG, or video-file source."""
    return cv2.VideoCapture(source)
