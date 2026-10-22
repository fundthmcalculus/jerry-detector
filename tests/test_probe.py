import cv2
import numpy as np

from jerry.cli import probe


def _write_video(path, frames=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()


def test_probe_reads_local_video(tmp_path, capsys):
    video = tmp_path / "cam.avi"
    _write_video(video)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"cameras:\n  - name: test-cam\n    source: {video}\n"
        f"  - name: off-cam\n    source: {video}\n    enabled: false\n"
    )

    assert probe(config, frames=5) is True
    out = capsys.readouterr().out
    assert "OK    test-cam: 64x48" in out
    assert "off-cam" not in out


def test_probe_reports_missing_source(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(f"cameras:\n  - name: gone\n    source: {tmp_path / 'nope.avi'}\n")

    assert probe(config, frames=5) is False
    assert "FAIL  gone" in capsys.readouterr().out
