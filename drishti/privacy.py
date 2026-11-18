"""Privacy helpers: find each person's head and blur it before anything is shown.

Heads are located from pose keypoints (nose, eyes, ears) when the detector provides them, so a
close-up face is blurred as well as a distant one. Without keypoints (e.g. a person seen from
behind) the top band of the person box is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

HEAD_FRACTION = 0.22  # fallback: top part of a person box that holds the head
KEYPOINT_CONF = 0.5
FACE_KEYPOINTS = 5  # COCO order: nose, left eye, right eye, left ear, right ear


def head_box(person: Sequence[float], keypoints: Sequence[Sequence[float]] | None) -> list[float]:
    """Head region [x1, y1, x2, y2] for a person box, from face keypoints when available."""
    x1, y1, x2, y2 = (float(v) for v in person)
    height = y2 - y1
    pts = [
        (float(k[0]), float(k[1]))
        for k in (keypoints if keypoints is not None else [])[:FACE_KEYPOINTS]
        if len(k) > 2 and k[2] >= KEYPOINT_CONF
    ]
    if len(pts) >= 2:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.09 * height, 8.0)
        half = 1.1 * span
        # the skull extends further above the eyes than the chin does below them
        return [cx - half, cy - 1.5 * half, cx + half, cy + 1.25 * half]
    return [x1, y1, x2, y1 + max(4.0, HEAD_FRACTION * height)]


def blur_regions(img: np.ndarray, regions: Iterable[Sequence[float]]) -> np.ndarray:
    """Blur each [x1, y1, x2, y2] region in place (clipped to the image)."""
    h, w = img.shape[:2]
    for rx1, ry1, rx2, ry2 in regions:
        x1, y1 = max(0, int(rx1)), max(0, int(ry1))
        x2, y2 = min(w, int(rx2) + 1), min(h, int(ry2) + 1)
        roi = img[y1:y2, x1:x2]
        if roi.size:
            k = max(9, (max(roi.shape[:2]) // 2) | 1)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


def blur_crop(img: np.ndarray, head: Sequence[float] | None) -> np.ndarray:
    """Blur a person crop. ``head`` is the head box as fractions of the crop (from ingest)."""
    hgt, wid = img.shape[:2]
    if head is None:
        region = [0.0, 0.0, float(wid), max(4.0, HEAD_FRACTION * hgt)]
    else:
        region = [head[0] * wid, head[1] * hgt, head[2] * wid, head[3] * hgt]
    return blur_regions(img, [region])


def to_jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode JPEG")
    return bytes(buf.tobytes())
