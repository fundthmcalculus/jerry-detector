from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """`frames` are RGB uint8 arrays."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
    )


class ClipStore:
    """Writes each clip as <id>.gif plus <id>.json metadata in one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, camera: str, when: datetime, frames: list[np.ndarray], fps: float) -> dict:
        clip_id = f"{camera}_{when:%Y%m%d-%H%M%S}"
        write_gif(frames, self.root / f"{clip_id}.gif", fps)
        meta = {
            "id": clip_id,
            "camera": camera,
            "created": when.isoformat(),
            "frames": len(frames),
            "fps": fps,
        }
        (self.root / f"{clip_id}.json").write_text(json.dumps(meta))
        return meta

    def log_off_hours(self, record: dict) -> None:
        """Append activity seen outside operating hours to off_hours.jsonl (not served as a clip)."""
        with (self.root / "off_hours.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def list(self) -> list[dict]:
        metas = [json.loads(p.read_text()) for p in self.root.glob("*.json")]
        return sorted(metas, key=lambda m: m["created"], reverse=True)

    def gif_path(self, clip_id: str) -> Path | None:
        if not _SAFE_ID.fullmatch(clip_id):
            return None
        path = self.root / f"{clip_id}.gif"
        return path if path.is_file() else None
