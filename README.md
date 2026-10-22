# Jerry Detector

Watches north-facing webcam feeds for crashing skiers. When it sees one, it saves a
pre/post-roll GIF and serves it on a small website. Subscriptions and notifications are
planned, not built yet.

Target: `jerrydetector.com`, eventually routed to a Jetson Orin Nano at home. Hosting is not set up.

## Layout

| Path | Purpose |
|---|---|
| `src/jerry/config.py` | YAML config loading |
| `src/jerry/detect.py` | Person detector interface + YOLO pose backend |
| `src/jerry/rule.py` | Crash heuristic (placeholder, to tune) |
| `src/jerry/clip.py` | Pre/post-roll frame buffer |
| `src/jerry/pipeline.py` | One thread per camera: read, detect, clip |
| `src/jerry/store.py` | GIF + JSON metadata writer and reader |
| `src/jerry/web/` | FastAPI app and static page |
| `src/jerry/cli.py` | `jerry run` and `jerry serve` |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",ml" for the YOLO detector
cp config.example.yaml config.yaml # set camera sources
jerry run                          # watch cameras, write clips/*.gif
jerry serve --port 8000            # website at http://127.0.0.1:8000
pytest
```

A video file works as a camera `source` for testing. It replays when it ends.

## Jetson Orin Nano

- Use JetPack's PyTorch and OpenCV builds. The pip `opencv-python-headless` wheel has no CUDA or GStreamer.
- Export the model to TensorRT once, then point `detector.model` at the `.engine` file:
  `yolo export model=yolo11n-pose.pt format=engine half=True device=0`
- Set `detector.device: 0`.

## Domain (later)

`jerrydetector.com` is at GoDaddy. When hosting is ready, the usual route for a device behind
home NAT is a Cloudflare Tunnel, or a DNS record to your public IP with port forwarding.

## Next

- Tune the crash rule on real footage. It uses the largest person's box shape only.
- Add a pose-based check (hip and shoulder keypoints) if box shape is too noisy.
- Subscriptions and email/push notifications with the GIF attached.
