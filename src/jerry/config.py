from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraConfig:
    name: str
    source: str
    enabled: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "yolo_pose"
    model: str = "yolo11n-pose.pt"
    device: str = "cpu"
    confidence: float = 0.4


@dataclass(frozen=True)
class RuleConfig:
    horizontal_ratio: float = 1.3
    hold_seconds: float = 1.0
    cooldown_seconds: float = 60.0


@dataclass(frozen=True)
class ClipConfig:
    fps: float = 5.0
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    gif_width: int = 480
    output_dir: Path = Path("clips")


@dataclass(frozen=True)
class Config:
    cameras: list[CameraConfig] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    clip = dict(raw.get("clip", {}))
    if "output_dir" in clip:
        clip["output_dir"] = Path(clip["output_dir"])
    return Config(
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
        detector=DetectorConfig(**raw.get("detector", {})),
        rule=RuleConfig(**raw.get("rule", {})),
        clip=ClipConfig(**clip),
    )
