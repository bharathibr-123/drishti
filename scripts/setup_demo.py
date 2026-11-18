"""Download model weights and the CC BY 4.0 demo videos used in demo/cameras.json.

python scripts/setup_demo.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
WEIGHTS = {
    "yolo11s-pose.pt": ULTRALYTICS + "yolo11s-pose.pt",
    # ~830 MB, the most accurate CLIP model in our benchmark (see eval/RESULTS.md)
    "clip_vit_b16plus240_laion400m.pt": "https://github.com/mlfoundations/open_clip/releases/"
    "download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt",
}
# Intel IoT DevKit sample videos, licensed CC BY 4.0
VIDEO_BASE = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
VIDEOS = [
    "people-detection.mp4",
    "face-demographics-walking.mp4",
    "one-by-one-person-detection.mp4",
    "store-aisle-detection.mp4",
    "worker-zone-detection.mp4",
    "classroom.mp4",
    "face-demographics-walking-and-pause.mp4",
    "person-bicycle-car-detection.mp4",
    "head-pose-face-detection-female-and-male.mp4",
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok   {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {dest.relative_to(ROOT)}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as exc:  # keep going, report at the end
        tmp.unlink(missing_ok=True)
        print(f"  FAILED {url}: {exc}", file=sys.stderr)
        raise


def main() -> None:
    print("Model weights:")
    for name, url in WEIGHTS.items():
        fetch(url, ROOT / "weights" / name)
    print("Demo videos (CC BY 4.0, Intel IoT DevKit):")
    for v in VIDEOS:
        fetch(VIDEO_BASE + v, ROOT / "demo" / "videos" / v)
    print("Done. Next: python -m drishti.ingest demo/cameras.json")


if __name__ == "__main__":
    main()
