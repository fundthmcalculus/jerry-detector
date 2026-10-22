from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from jerry.config import DetectorConfig

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Person:
    box: Box
    score: float

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Person]:
        """`frame` is a BGR uint8 image, the OpenCV convention."""
        ...


class YoloPoseDetector:
    """Ultralytics YOLO (COCO class 0 = person).

    On Jetson, export once and point `model` at the result:
        yolo export model=yolo11n-pose.pt format=engine half=True device=0
    """

    def __init__(self, model: str, device: str, confidence: float, imgsz: int) -> None:
        from ultralytics import YOLO  # optional dependency: pip install -e ".[ml]"

        self._model = YOLO(model)
        self._device = device
        self._confidence = confidence
        self._imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Person]:
        result = self._model.predict(
            frame,
            classes=[0],
            conf=self._confidence,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        return [
            Person(box=tuple(xyxy), score=float(conf))
            for xyxy, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
        ]


class NullDetector:
    """Detects nothing: for exercising stream, schedule and motion checks without a model."""

    def detect(self, frame: np.ndarray) -> list[Person]:
        return []


def make_detector(cfg: DetectorConfig) -> PersonDetector:
    if cfg.backend == "none":
        return NullDetector()
    if cfg.backend == "yolo_pose":
        return YoloPoseDetector(cfg.model, cfg.device, cfg.confidence, cfg.imgsz)
    raise ValueError(f"unknown detector backend: {cfg.backend!r}")
