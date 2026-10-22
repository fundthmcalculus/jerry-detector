from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

from jerry.capture import open_capture
from jerry.config import load_config
from jerry.detect import make_detector
from jerry.pipeline import CameraWorker
from jerry.store import ClipStore

log = logging.getLogger(__name__)


def run_detection(config_path: Path) -> None:
    cfg = load_config(config_path)
    detector = make_detector(cfg.detector)
    store = ClipStore(cfg.clip.output_dir)
    stop = threading.Event()

    workers = [
        CameraWorker(cam, cfg, detector, store, stop)
        for cam in cfg.cameras
        if cam.enabled
    ]
    if not workers:
        raise SystemExit("no enabled cameras in config")

    for w in workers:
        w.start()
    try:
        for w in workers:
            w.join()
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()


def probe(config_path: Path, frames: int) -> bool:
    """Open each enabled camera, read a few frames, and report what came back."""
    cfg = load_config(config_path)
    all_ok = True
    for cam in cfg.cameras:
        if not cam.enabled:
            continue
        cap = open_capture(cam.source)
        if not cap.isOpened():
            print(f"FAIL  {cam.name}: could not open source")
            all_ok = False
            cap.release()
            continue
        start = time.monotonic()
        got = 0
        shape = None
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            got += 1
            shape = frame.shape
        elapsed = time.monotonic() - start
        cap.release()
        if got == 0:
            print(f"FAIL  {cam.name}: opened but no frames")
            all_ok = False
        else:
            h, w = shape[:2]
            fps = got / elapsed if elapsed > 0 else float("nan")
            print(f"OK    {cam.name}: {w}x{h}, read {got}/{frames} frames at {fps:.1f} fps")
    return all_ok


def serve_web(config_path: Path, host: str, port: int) -> None:
    import uvicorn

    from jerry.web.app import create_app

    cfg = load_config(config_path)
    uvicorn.run(create_app(cfg.clip.output_dir), host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jerry", description="Skier crash detector")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="watch cameras and save crash clips")

    check = sub.add_parser("probe", help="check each camera opens and delivers frames")
    check.add_argument("--frames", type=int, default=30)

    serve = sub.add_parser("serve", help="serve the clip website")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "run":
        run_detection(args.config)
    elif args.cmd == "probe":
        raise SystemExit(0 if probe(args.config, args.frames) else 1)
    else:
        serve_web(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
