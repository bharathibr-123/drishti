"""Settings. Every value can be overridden with a ``DRISHTI_<NAME>`` environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DRISHTI_{name}", default)


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0") == "1"


# Locally downloaded checkpoints (scripts/setup_demo.py), by OpenCLIP architecture name.
_LOCAL_CLIP = {
    "ViT-B-16-plus-240": "clip_vit_b16plus240_laion400m.pt",
    "ViT-B-16": "clip_vit_b16_laion400m.pt",
    "ViT-B-32-quickgelu": "clip_vit_b32_laion400m.pt",
}


@dataclass(frozen=True)
class Settings:
    # where things live
    index_dir: Path
    audit_log: Path
    cameras_file: Path
    # person detector + tracker (Ultralytics YOLO); weights download automatically if missing
    detector_weights: str
    tracker: str
    det_conf: float
    det_imgsz: int
    # vision-language model (OpenCLIP)
    clip_arch: str
    clip_pretrained: str
    # ingestion
    sample_fps: float
    min_track_hits: int
    crops_per_track: int
    min_box_height: int
    mask_background: bool
    # privacy
    privacy_blur: bool
    # live monitoring
    live_fps: float

    @classmethod
    def from_env(cls) -> Settings:
        mask = _flag("MASK_BACKGROUND", False)
        arch = _env("CLIP_ARCH", "ViT-B-16-plus-240")
        local_clip = ROOT / "weights" / _LOCAL_CLIP.get(arch, "")
        # the pose model finds people AND their eyes/nose/ears, used to blur faces precisely
        default_det = "yolo11s-seg.pt" if mask else "yolo11s-pose.pt"
        return cls(
            index_dir=Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))),
            audit_log=Path(_env("AUDIT_LOG", str(ROOT / "data" / "audit_log.jsonl"))),
            cameras_file=Path(_env("CAMERAS", str(ROOT / "demo" / "cameras.json"))),
            detector_weights=_env("DETECTOR_WEIGHTS", str(ROOT / "weights" / default_det)),
            tracker=_env("TRACKER", "bytetrack.yaml"),
            det_conf=float(_env("DET_CONF", "0.35")),
            det_imgsz=int(_env("DET_IMGSZ", "640")),
            # ViT-B/16+ won our benchmark (eval/RESULTS.md); its checkpoint is GitHub-hosted, so it
            # works where Hugging Face is blocked. On a slow laptop use
            # DRISHTI_CLIP_ARCH=ViT-B-32-quickgelu (about 3x faster, a little less accurate).
            clip_arch=arch,
            clip_pretrained=_env(
                "CLIP_PRETRAINED",
                str(local_clip) if local_clip.is_file() else "laion400m_e32",
            ),
            sample_fps=float(_env("SAMPLE_FPS", "4")),
            min_track_hits=int(_env("MIN_TRACK_HITS", "3")),
            crops_per_track=int(_env("CROPS_PER_TRACK", "6")),
            min_box_height=int(_env("MIN_BOX_HEIGHT", "40")),
            # Experimental: grey out everything that is not the person before embedding. In our
            # pilot it fixed some queries and broke others, so it is off by default.
            mask_background=mask,
            privacy_blur=_flag("PRIVACY_BLUR", True),
            live_fps=float(_env("LIVE_FPS", "3")),
        )


settings = Settings.from_env()
