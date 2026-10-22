from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from jerry.clip import ClipBuffer
from jerry.config import CameraConfig, Config
from jerry.detect import PersonDetector
from jerry.rule import CrashRule
from jerry.schedule import OperatingHours, next_spot_check_delay
from jerry.store import ClipStore
from jerry.stream import FrameReader, open_reader

log = logging.getLogger(__name__)


class CameraWorker(threading.Thread):
    """One camera. Detects and saves clips during operating hours; outside them it sits idle
    and only takes short, randomly timed spot checks that log unexpected activity."""

    RECONNECT_SECONDS = 5.0
    IDLE_POLL_SECONDS = 60.0

    def __init__(
        self,
        camera: CameraConfig,
        cfg: Config,
        detector: PersonDetector,
        store: ClipStore,
        stop: threading.Event,
        hours: OperatingHours | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(name=f"camera-{camera.name}", daemon=True)
        self.camera = camera
        self._cfg = cfg
        self._detector = detector
        self._store = store
        self._stop_event = stop  # not `_stop`: that would shadow threading.Thread._stop
        self._hours = hours or OperatingHours(cfg.schedule)
        self._rng = rng or random.Random()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._hours.is_open(self._hours.now()):
                self._watch()
                self._stop_event.wait(self.RECONNECT_SECONDS)
            else:
                self._idle()

    # --- operating hours -------------------------------------------------

    def _open_reader(self) -> FrameReader | None:
        try:
            reader = open_reader(self.camera.source, self._cfg.clip.fps, self._cfg.stream)
        except OSError as e:
            log.warning("%s: could not open source: %s", self.camera.name, e)
            return None
        log.info("%s: connected", self.camera.name)
        return reader

    def _watch(self) -> None:
        reader = self._open_reader()
        if reader is None:
            return
        rule = CrashRule(self._cfg.rule.horizontal_ratio, self._cfg.rule.hold_seconds, self._cfg.rule.cooldown_seconds)
        buffer = ClipBuffer(self._cfg.clip.pre_seconds, self._cfg.clip.post_seconds)
        try:
            while not self._stop_event.is_set():
                item = reader.read()
                if item is None:
                    log.warning("%s: stream ended or dropped", self.camera.name)
                    return
                if not self._hours.is_open(self._hours.now()):
                    log.info("%s: closing time, pausing", self.camera.name)
                    return
                now, bgr = item
                self._process(now, bgr, rule, buffer)
        finally:
            reader.close()

    def _process(self, now: float, bgr: np.ndarray, rule: CrashRule, buffer: ClipBuffer) -> None:
        people = self._detector.detect(bgr)
        if rule.update(people, now):
            log.info("%s: crash detected", self.camera.name)
            buffer.trigger(now)

        clip = buffer.push(now, self._for_gif(bgr))
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

    # --- off hours -------------------------------------------------------

    def _idle(self) -> None:
        """Wait for opening time, taking a spot check at random intervals. Never saves clips."""
        until_open = self._hours.seconds_until_open(self._hours.now())
        log.info("%s: closed, next open in %.1f h", self.camera.name, until_open / 3600)
        check_at = time.monotonic() + next_spot_check_delay(self._cfg.schedule, self._rng)
        while not self._stop_event.is_set() and not self._hours.is_open(self._hours.now()):
            if time.monotonic() >= check_at:
                self._spot_check()
                check_at = time.monotonic() + next_spot_check_delay(self._cfg.schedule, self._rng)
            self._stop_event.wait(min(self.IDLE_POLL_SECONDS, max(0.0, check_at - time.monotonic())))

    def _spot_check(self) -> None:
        reader = self._open_reader()
        if reader is None:
            return
        sched = self._cfg.schedule
        deadline = time.monotonic() + sched.spot_check_seconds
        frames = max_people = 0
        try:
            while time.monotonic() < deadline and not self._stop_event.is_set():
                item = reader.read()
                if item is None:
                    break
                frames += 1
                max_people = max(max_people, len(self._detector.detect(item[1])))
        finally:
            reader.close()

        # People only: the feeds are PTZ cameras that tour between presets, so frame
        # differencing can't tell a pan from activity.
        if max_people:
            record = {
                "camera": self.camera.name,
                "checked": self._hours.now().isoformat(),
                "frames": frames,
                "people": max_people,
            }
            log.warning("%s: activity outside operating hours: %s", self.camera.name, record)
            self._store.log_off_hours(record)
        else:
            log.info("%s: off-hours spot check quiet (%d frames)", self.camera.name, frames)
