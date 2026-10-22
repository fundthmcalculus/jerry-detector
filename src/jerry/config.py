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
class StreamConfig:
    """How HTTP(S)/HLS sources are read (through an ffmpeg subprocess)."""

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
    frame_width: int = 960  # frames are downscaled to this width before detection
    timeout_seconds: float = 15.0  # network read stall limit


@dataclass(frozen=True)
class ScheduleConfig:
    """Operating hours, in `timezone` wall-clock. Outside them no clips are ever saved."""

    timezone: str = "America/New_York"
    open: str = "09:30"
    close: str = "21:30"
    spot_check_min_minutes: float = 20.0  # random gap between off-hours spot checks
    spot_check_max_minutes: float = 60.0
    spot_check_seconds: float = 15.0


@dataclass(frozen=True)
class Config:
    cameras: list[CameraConfig] = field(default_factory=list)
    stream: StreamConfig = field(default_factory=StreamConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
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
        stream=StreamConfig(**raw.get("stream", {})),
        schedule=ScheduleConfig(**raw.get("schedule", {})),
        detector=DetectorConfig(**raw.get("detector", {})),
        rule=RuleConfig(**raw.get("rule", {})),
        clip=ClipConfig(**clip),
    )
