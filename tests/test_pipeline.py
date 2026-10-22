import json
import threading
from datetime import datetime, timezone

import numpy as np

import jerry.pipeline as pipeline
from jerry.config import CameraConfig, ClipConfig, Config, ScheduleConfig
from jerry.detect import Person
from jerry.store import ClipStore

FRAME = np.zeros((90, 160, 3), np.uint8)


class FakeReader:
    def __init__(self, frames: int) -> None:
        self._left = frames
        self.closed = False

    def read(self):
        self._left -= 1
        return (0.0, FRAME) if self._left >= 0 else None

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self, people: int) -> None:
        self._people = [Person(box=(0, 0, 10, 10), score=1.0)] * people

    def detect(self, frame):
        return self._people


class StubHours:
    def __init__(self, open_: bool) -> None:
        self._open = open_

    def now(self):
        return datetime(2027, 1, 15, 3, 0, tzinfo=timezone.utc)

    def is_open(self, at) -> bool:
        return self._open


def make_worker(tmp_path, monkeypatch, people: int, open_: bool = False):
    reader = FakeReader(frames=10)
    monkeypatch.setattr(pipeline, "open_reader", lambda *a: reader)
    cfg = Config(
        cameras=[CameraConfig("cam", "https://example.test/x.m3u8")],
        schedule=ScheduleConfig(spot_check_seconds=5),
        clip=ClipConfig(output_dir=tmp_path),
    )
    store = ClipStore(tmp_path)
    worker = pipeline.CameraWorker(
        cfg.cameras[0], cfg, FakeDetector(people), store, threading.Event(), hours=StubHours(open_)
    )
    return worker, reader


def test_spot_check_logs_person_seen_off_hours(tmp_path, monkeypatch):
    worker, reader = make_worker(tmp_path, monkeypatch, people=2)
    worker._spot_check()
    record = json.loads((tmp_path / "off_hours.jsonl").read_text())
    assert record["camera"] == "cam" and record["people"] == 2 and record["frames"] == 10
    assert reader.closed
    assert list(tmp_path.glob("*.gif")) == []  # never a clip off hours


def test_quiet_spot_check_logs_nothing(tmp_path, monkeypatch):
    worker, _ = make_worker(tmp_path, monkeypatch, people=0)
    worker._spot_check()
    assert not (tmp_path / "off_hours.jsonl").exists()


def test_watch_stops_at_closing_time(tmp_path, monkeypatch):
    worker, reader = make_worker(tmp_path, monkeypatch, people=0, open_=False)
    worker._watch()  # closed from the first frame: returns without processing
    assert reader.closed
    assert list(tmp_path.glob("*")) == []


def test_worker_can_be_joined(tmp_path, monkeypatch):
    """Regression: an attribute named `_stop` shadows Thread._stop and breaks join()."""
    worker, _ = make_worker(tmp_path, monkeypatch, people=0, open_=False)
    worker._stop_event.set()
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
