"""Turn raw CCTV video into a searchable index.

For every camera:
  1. sample frames (default 4 fps); detect and track people with YOLO11 + ByteTrack
  2. keep each person's best few crops, spread across their time on screen
  3. embed every crop with CLIP so it can be matched against a text or photo query

Usage:
    python -m drishti.ingest demo/cameras.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import settings
from .models import get_encoder, new_detector
from .privacy import head_box
from .records import Box, Camera, Crop, Track
from .store import Store


@dataclass
class _Candidate:
    t: float
    frame: int
    box: Box
    quality: float
    head: list[float]  # head box as fractions of the crop
    img: np.ndarray  # raw crop (shown to officers)
    emb_img: np.ndarray  # crop used for the embedding (optionally background-masked)


EDGE_MARGIN = 0.015  # a box this close to the frame border (fraction of size) is cut off


def _quality(conf: float, box: Box, w: int, h: int) -> float:
    """Prefer large, confident, fully visible views. Boxes cut off by the frame edge are often
    half a person, or two people blended, so they are only used when nothing better exists."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    mx, my = max(4.0, EDGE_MARGIN * w), max(4.0, EDGE_MARGIN * h)
    touches_edge = x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my
    return conf * float(np.sqrt(area)) * (0.2 if touches_edge else 1.0)


def _pick_spread(cands: list[_Candidate], k: int) -> list[_Candidate]:
    """The best-quality candidate inside each of k equal time bins."""
    if len(cands) <= k:
        return cands
    edges = np.linspace(cands[0].t, cands[-1].t + 1e-6, k + 1)
    out = []
    for a, b in pairwise(edges):
        in_bin = [c for c in cands if a <= c.t < b]
        if in_bin:
            out.append(max(in_bin, key=lambda c: c.quality))
    return out


def _person_only(crop: np.ndarray, poly: np.ndarray | None, x0: int, y0: int) -> np.ndarray:
    """Grey out everything in the box that is not this person (segmentation mask)."""
    if poly is None or len(poly) < 3:
        return crop.copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [(poly - [x0, y0]).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    if mask.mean() < 25:  # segmentation failed - keep the raw crop
        return crop.copy()
    out = np.full_like(crop, 128)
    out[mask > 0] = crop[mask > 0]
    return out


def _detect_and_track(
    video: Path,
) -> tuple[dict[int, list[_Candidate]], dict[int, list[list[float]]], float, int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / settings.sample_fps))
    detector = new_detector()
    cands: dict[int, list[_Candidate]] = defaultdict(list)
    frame_heads: dict[int, list[list[float]]] = {}
    idx, size = 0, (0, 0)
    while cap.grab():
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            size = (w, h)
            r = detector.track(
                frame,
                persist=True,
                classes=[0],
                conf=settings.det_conf,
                imgsz=settings.det_imgsz,
                tracker=settings.tracker,
                verbose=False,
            )[0]
            if r.boxes is not None and r.boxes.id is not None:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
                heads = [
                    head_box(box, kpts[i] if kpts is not None else None)
                    for i, box in enumerate(xyxy.tolist())
                ]
                frame_heads[idx] = [[round(v, 1) for v in hb] for hb in heads]
                polys = r.masks.xy if r.masks is not None else [None] * len(xyxy)
                ids = r.boxes.id.int().tolist()
                for box, tid, conf, poly, hb in zip(
                    xyxy, ids, r.boxes.conf.tolist(), polys, heads, strict=True
                ):
                    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                    if y2 - y1 < settings.min_box_height:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    emb_img = _person_only(crop, poly, x1, y1) if settings.mask_background else crop
                    cw, ch = x2 - x1, y2 - y1
                    head = [
                        (hb[0] - x1) / cw,
                        (hb[1] - y1) / ch,
                        (hb[2] - x1) / cw,
                        (hb[3] - y1) / ch,
                    ]
                    scale = min(1.0, 256 / crop.shape[0])
                    if scale < 1:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale)
                        emb_img = cv2.resize(emb_img, None, fx=scale, fy=scale)
                    lst = cands[tid]
                    b: Box = [x1, y1, x2, y2]
                    lst.append(
                        _Candidate(
                            t=idx / fps,
                            frame=idx,
                            box=b,
                            quality=_quality(conf, b, w, h),
                            head=[round(v, 4) for v in head],
                            img=crop.copy(),
                            emb_img=emb_img.copy(),
                        )
                    )
                    if len(lst) > 80:  # bound memory on very long tracks
                        del lst[1::2]
        idx += 1
    cap.release()
    return cands, frame_heads, fps, idx, size


def _read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    i = 0
    while wanted - frames.keys() and cap.grab():
        if i in wanted:
            _, f = cap.retrieve()
            frames[i] = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])))
        i += 1
    cap.release()
    return frames


def ingest_camera(
    camera: Camera, store: Store, base_dir: Path, log: Callable[[str], None] = print
) -> int:
    started = time.time()
    video = (base_dir / camera["video"]).resolve()
    cands, frame_heads, fps, n_frames, size = _detect_and_track(video)
    cam = camera["id"]

    chosen = {
        tid: _pick_spread(lst, settings.crops_per_track)
        for tid, lst in cands.items()
        if len(lst) >= settings.min_track_hits
    }
    frames = _read_frames(video, {c.frame for lst in chosen.values() for c in lst})

    tracks: list[Track] = []
    crops: list[Crop] = []
    images: list[Image.Image] = []
    for tid, lst in chosen.items():
        key, everything = f"{cam}:{tid}", cands[tid]
        best = max(range(len(lst)), key=lambda j: lst[j].quality)
        for j, c in enumerate(lst):
            name = f"{cam}_{tid}_{j}"
            cv2.imwrite(str(store.root / "thumbs" / f"{name}.jpg"), c.img)
            if c.frame in frames:
                cv2.imwrite(str(store.root / "frames" / f"{name}.jpg"), frames[c.frame])
            crops.append(
                Crop(
                    track=key,
                    camera_id=cam,
                    t=round(c.t, 2),
                    box=c.box,
                    frame_size=list(size),
                    thumb=name,
                    head=c.head,
                    heads=frame_heads.get(c.frame, []),
                )
            )
            images.append(Image.fromarray(cv2.cvtColor(c.emb_img, cv2.COLOR_BGR2RGB)))
        step = max(1, len(everything) // 40)
        path = [
            [
                round(c.t, 2),
                round((c.box[0] + c.box[2]) / 2 / size[0], 3),
                round((c.box[1] + c.box[3]) / 2 / size[1], 3),
            ]
            for c in everything[::step]
        ]
        tracks.append(
            Track(
                key=key,
                camera_id=cam,
                t_start=round(everything[0].t, 2),
                t_end=round(everything[-1].t, 2),
                hits=len(everything),
                thumb=f"{cam}_{tid}_{best}",
                path=path,
            )
        )

    emb = get_encoder().embed_images(images)
    store.add(
        {**camera, "video": str(video), "duration": round(n_frames / fps, 1)}, tracks, crops, emb
    )
    log(
        f"[{cam}] {n_frames / fps:.0f}s of video -> {len(tracks)} people, "
        f"{len(crops)} crops in {time.time() - started:.0f}s"
    )
    return len(tracks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Index CCTV footage for Drishti search")
    ap.add_argument("cameras", help="JSON file listing cameras (see demo/cameras.json)")
    ap.add_argument("--only", nargs="*", help="only (re)index these camera ids")
    args = ap.parse_args()
    path = Path(args.cameras)
    cams: list[Camera] = json.loads(path.read_text())
    store = Store()
    for cam in cams:
        if cam.get("live") or (args.only and cam["id"] not in args.only):
            continue
        ingest_camera(cam, store, path.parent)
        store.save()
    print(
        f"Index ready: {len(store.tracks)} people across {len(store.cameras)} cameras "
        f"-> {store.root}"
    )


if __name__ == "__main__":
    main()
