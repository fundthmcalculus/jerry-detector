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
| `src/jerry/pipeline.py` | One thread per camera: read, detect, clip; idle and spot-check off hours |
| `src/jerry/stream.py` | OpenCV frame reader: sampling, downscaling, stream timestamps |
| `src/jerry/schedule.py` | Operating hours and random spot-check timing |
| `src/jerry/scene.py` | PTZ scene-change gate in front of the crash rule |
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
  `yolo export model=yolo11n-pose.pt format=engine half=True device=0 imgsz=1920`
- Set `detector.device: 0`.

## Domain (later)

`jerrydetector.com` is at GoDaddy. When hosting is ready, the usual route for a device behind
home NAT is a Cloudflare Tunnel, or a DNS record to your public IP with port forwarding.

## Next

- Tune the crash rule on real footage. It uses the largest person's box shape only.
- Add a pose-based check (hip and shoulder keypoints) if box shape is too noisy.
- Subscriptions and email/push notifications with the GIF attached.

## Perfect North streams

`config.example.yaml` has the Area Cam and Red Cam HLS feeds. Notes from probing them:

- The relay answers the default FFmpeg/curl User-Agent with `405`. OpenCV reads the feeds fine once `OPENCV_FFMPEG_CAPTURE_OPTIONS=user_agent;...` is set. On Windows that must go through `msvcrt._putenv`, because OpenCV's FFmpeg plugin ignores `os.environ`; `stream.py` handles it.
- Both are 1080p H.264 at ~30fps with discontinuity markers. OpenCV delivers HLS in segment-sized bursts, so frames are sampled by counting against the stream fps and timestamped by stream position, not arrival time.
- Both cameras are PTZ and tour between presets every few seconds. `scene.py` watches how much of the picture changes between sampled frames; past `rule.scene_change_fraction` the view counts as moving, detection is skipped for `rule.settle_seconds`, and the rule's "down" timer resets so a skier in one preset cannot complete a crash in the next.

## Operating hours

`schedule:` sets open/close in `America/New_York` (EST/EDT). Outside the window no clip is ever saved and no stream is held open. At random 20-60 minute intervals each camera takes a 15 s spot check; if the person detector sees anyone, it logs a warning and appends to `clips/off_hours.jsonl`. Motion differencing was tried and dropped because camera tours trigger it.
