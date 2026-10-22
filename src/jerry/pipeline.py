from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from jerry.clip import ClipBuffer
from jerry.config import CameraConfig, Config
from jerry.detect import PersonDetector
from jerry.rule import CrashRule
from jerry.store import ClipStore

log = logging.getLogger(__name__)


class CameraWorker(threading.Thread):
    """Reads one feed, samples it at clip.fps, detects, and saves clips on crashes."""

    RECONNECT_SECONDS = 5.0

    def __init__(
        self,
        camera: CameraConfig,
        cfg: Config,
        detector: PersonDetector,
        store: ClipStore,
        stop: threading.Event,
    ) -> None:
        super().__init__(name=f"camera-{camera.name}", daemon=True)
        self.camera = camera
        self._cfg = cfg
        self._detector = detector
        self._store = store
        self._stop = stop
        self._rule = CrashRule(cfg.rule.horizontal_ratio, cfg.rule.hold_seconds, cfg.rule.cooldown_seconds)
        self._buffer = ClipBuffer(cfg.clip.pre_seconds, cfg.clip.post_seconds)
        self._interval = 1.0 / cfg.clip.fps

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.camera.source)
            if cap.isOpened():
                log.info("%s: connected", self.camera.name)
                try:
                    self._read_loop(cap)
                finally:
                    cap.release()
            else:
                log.warning("%s: could not open source", self.camera.name)
            self._stop.wait(self.RECONNECT_SECONDS)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        next_at = 0.0
        while not self._stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                log.warning("%s: stream ended or dropped", self.camera.name)
                return
            now = time.monotonic()
            if now < next_at:
                continue
            next_at = now + self._interval
            self._process(now, bgr)

    def _process(self, now: float, bgr: np.ndarray) -> None:
        people = self._detector.detect(bgr)
        if self._rule.update(people, now):
            log.info("%s: crash detected", self.camera.name)
            self._buffer.trigger(now)

        clip = self._buffer.push(now, self._for_gif(bgr))
        if clip is not None:
            meta = self._store.save(self.camera.name, datetime.now(timezone.utc), clip, self._cfg.clip.fps)
            log.info("saved clip %s (%d frames)", meta["id"], meta["frames"])

    def _for_gif(self, bgr: np.ndarray) -> np.ndarray:
        """Shrink and convert to RGB for GIF encoding."""
        h, w = bgr.shape[:2]
        width = self._cfg.clip.gif_width
        if w > width:
            bgr = cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
