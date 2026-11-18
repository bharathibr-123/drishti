"""Live monitoring: watch a camera stream and raise an alert when a registered description appears.

An officer registers a watch ("a man in a dark blue t-shirt"). Each live camera runs in a
background thread: detect + track people, embed each person about once a second, score them with
exactly the same scoring as archive search, and judge the score against how that same query scores
across the whole archive (calibration). Two confident readings in a row raise an alert.

If the archive search already found the person (a strong #1 match), the watch also remembers what
they look like. A live person then also alerts on a moderate description match plus a same-person
appearance match (camera-normalised re-identification). Live close-ups score lower on descriptions
than full-body archive views, and this second signal recovers them without loosening the first.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import settings
from .models import new_detector
from .privacy import blur_regions, head_box, to_jpeg
from .query import ParsedQuery, parse
from .records import Camera
from .scoring import STRONG_Z, ScoringOptions, calibrate
from .search import SearchEngine
from .trail import SAME_PERSON_SIM

CONFIRMATIONS = 2  # consecutive strong readings needed before alerting (avoids one-frame flukes)
MIN_ALERT_Z = 1.8  # never alert below this, however vague the description
ARCHIVE_FRACTION = 0.75  # alert when a live person matches nearly as well as the best archive match
REID_MIN_Z = 1.0  # with an appearance match, a moderate description match is enough
MIN_CAMERA_SAMPLES = 4  # live crops needed before the camera's average look is usable
IDLE_AFTER_S = 10.0  # with no watches and nobody viewing the feed, cameras stop analysing
EMBED_EVERY_S = 1.0  # re-embed each tracked person at most this often


@dataclass
class Watch:
    id: int
    query: str
    parsed: ParsedQuery
    centre: float
    spread: float
    threshold: float  # z needed on the live feed (see alert_threshold)
    look: np.ndarray | None  # appearance of the archive's strong #1 match, if there was one
    officer: str
    reason: str
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Alert:
    id: int
    watch_id: int
    query: str
    camera_id: str
    camera: str
    time: str
    z: float
    signal: str  # "description" or "description + appearance"
    snapshot: bytes = field(repr=False)  # privacy-blurred crop
    frame: bytes = field(repr=False)  # privacy-blurred frame with the person boxed

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "watch_id": self.watch_id,
            "query": self.query,
            "camera_id": self.camera_id,
            "camera": self.camera,
            "time": self.time,
            "z": round(self.z, 2),
            "signal": self.signal,
        }


def should_alert(z: float, look_sim: float | None, threshold: float) -> str | None:
    """Which signal (if any) says this live reading is the watched person."""
    if z >= threshold:
        return "description"
    if look_sim is not None and look_sim >= SAME_PERSON_SIM and z >= REID_MIN_Z:
        return "description + appearance"
    return None


def same_query_watch(watches: Iterable[Watch], query: str) -> Watch | None:
    """An active watch for the same description (ignoring case and extra spaces), if any."""
    key = " ".join(query.lower().split())
    return next((w for w in watches if " ".join(w.query.lower().split()) == key), None)


def alert_threshold(best_archive_z: float) -> float:
    """z a live sighting needs before it alerts.

    A fixed bar fails for common descriptions: when many archive people half-match "blue
    t-shirt", even the right person stands out less. So the bar is "about as convincing as the
    best match in the archive", clamped to [MIN_ALERT_Z, STRONG_Z].
    """
    return float(min(STRONG_Z, max(MIN_ALERT_Z, ARCHIVE_FRACTION * best_archive_z)))


class LiveMonitor:
    """Owns the watches, the alert log and one worker thread per live camera."""

    def __init__(self, engine: SearchEngine, cameras: list[Camera], base_dir: Path) -> None:
        self.engine = engine
        self.cameras = [c for c in cameras if c.get("live")]
        self.base_dir = base_dir
        self.opts = ScoringOptions()
        self.watches: dict[int, Watch] = {}
        self.alerts: deque[Alert] = deque(maxlen=200)
        self.frames: dict[str, bytes] = {}  # latest annotated frame per camera (privacy-blurred)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self.last_viewed: dict[str, float] = {}  # camera id -> monotonic time of last viewer
        self._cam_sum: dict[str, np.ndarray] = {}  # running sum of live embeddings per camera
        self._cam_n: dict[str, int] = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------------ watches
    def add_watch(self, query: str, officer: str, reason: str) -> Watch:
        """Register a watch. The same description twice returns the existing watch."""
        with self._lock:
            existing = same_query_watch(self.watches.values(), query)
        if existing is not None:
            self.start()
            return existing
        q = parse(query)
        archive = self.engine.rank(self.engine.score_crops(q, self.opts), self.opts)
        centre, spread = calibrate(np.array([r.score for r in archive]))
        threshold = alert_threshold(archive[0].z if archive else STRONG_Z)
        look = None
        if archive and archive[0].label == "strong":
            look = self.engine.store.looks()[archive[0].key]
        w = Watch(next(self._ids), query, q, centre, spread, threshold, look, officer, reason)
        with self._lock:
            self.watches[w.id] = w
        self.start()
        return w

    def remove_watch(self, watch_id: int) -> None:
        with self._lock:
            self.watches.pop(watch_id, None)

    def alerts_after(self, alert_id: int) -> list[Alert]:
        with self._lock:
            return [a for a in self.alerts if a.id > alert_id]

    def alert(self, alert_id: int) -> Alert | None:
        with self._lock:
            return next((a for a in self.alerts if a.id == alert_id), None)

    # ------------------------------------------------------------------ threads
    def start(self) -> None:
        for cam in self.cameras:
            t = self._threads.get(cam["id"])
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._run, args=(cam,), daemon=True, name=cam["id"])
                self._threads[cam["id"]] = t
                t.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads.values())

    def _run(self, cam: Camera) -> None:
        video = str((self.base_dir / cam["video"]).resolve())
        detector = new_detector()
        last_embed: dict[int, float] = {}
        streak: dict[tuple[int, int], int] = {}  # (watch id, track id) -> consecutive strong reads
        alerted: set[tuple[int, int]] = set()
        while not self._stop.is_set():
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stride = max(1, round(fps / settings.live_fps))
            idx, t0 = 0, time.monotonic()
            while not self._stop.is_set() and cap.grab():
                idx += 1
                if idx % stride:
                    continue
                if self._idle(cam["id"]):
                    # nothing to do: hold the "camera" here instead of using the CPU
                    paused = time.monotonic()
                    while self._idle(cam["id"]) and not self._stop.is_set():
                        time.sleep(0.5)
                    t0 += time.monotonic() - paused
                # play back at real-time speed, as a live camera would deliver frames
                delay = idx / fps - (time.monotonic() - t0)
                if delay > 0:
                    time.sleep(delay)
                ok, frame = cap.retrieve()
                if not ok:
                    break
                self._process(cam, frame, detector, last_embed, streak, alerted)
            cap.release()
            detector = new_detector()  # the clip loops: start tracking afresh
            last_embed.clear()
            streak.clear()
            alerted.clear()

    def _process(
        self,
        cam: Camera,
        frame: np.ndarray,
        detector: Any,
        last_embed: dict[int, float],
        streak: dict[tuple[int, int], int],
        alerted: set[tuple[int, int]],
    ) -> None:
        r = detector.track(
            frame,
            persist=True,
            classes=[0],
            conf=settings.det_conf,
            imgsz=settings.det_imgsz,
            tracker=settings.tracker,
            verbose=False,
        )[0]
        boxes: list[list[int]] = []
        tids: list[int] = []
        heads: list[list[float]] = []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
            tids = r.boxes.id.int().tolist()
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            heads = [
                head_box(b, kpts[i] if kpts is not None else None) for i, b in enumerate(boxes)
            ]

        with self._lock:
            watches = list(self.watches.values())
        now = time.monotonic()
        due = [
            i
            for i, tid in enumerate(tids)
            if watches
            and now - last_embed.get(tid, 0.0) >= EMBED_EVERY_S
            and boxes[i][3] - boxes[i][1] >= settings.min_box_height
        ]
        hits: set[int] = set()
        if due:
            crops = [
                frame[max(0, boxes[i][1]) : boxes[i][3], max(0, boxes[i][0]) : boxes[i][2]]
                for i in due
            ]
            emb = self.engine.encoder.embed_images(
                [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
            )
            looks = self._camera_normalised(cam["id"], emb)
            for j, i in enumerate(due):
                last_embed[tids[i]] = now
                for w in watches:
                    s = float(self.engine.score_against(emb[j : j + 1], w.parsed, self.opts)[0])
                    z = (s - w.centre) / w.spread
                    look_sim = (
                        float(looks[j] @ w.look)
                        if looks is not None and w.look is not None
                        else None
                    )
                    signal = should_alert(z, look_sim, w.threshold)
                    key = (w.id, tids[i])
                    streak[key] = streak.get(key, 0) + 1 if signal else 0
                    if signal and streak[key] >= CONFIRMATIONS:
                        hits.add(i)
                        if key not in alerted:
                            alerted.add(key)
                            self._raise(w, cam, frame, boxes, heads, i, z, signal)

        self.frames[cam["id"]] = to_jpeg(self._annotate(frame, boxes, heads, hits), quality=75)

    def _idle(self, cam_id: str) -> bool:
        with self._lock:
            watching = bool(self.watches)
        viewed = time.monotonic() - self.last_viewed.get(cam_id, -1e9) < IDLE_AFTER_S
        return not watching and not viewed

    def _camera_normalised(self, cam_id: str, emb: np.ndarray) -> np.ndarray | None:
        """Live crops minus this camera's running average look (as store.camera_normalise)."""
        total = self._cam_sum.get(cam_id, np.zeros(emb.shape[1], dtype=np.float32)) + emb.sum(0)
        n = self._cam_n.get(cam_id, 0) + len(emb)
        self._cam_sum[cam_id], self._cam_n[cam_id] = total, n
        if n < MIN_CAMERA_SAMPLES:
            return None
        out = emb - total / n
        return np.asarray(out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9))

    def _annotate(
        self, frame: np.ndarray, boxes: list[list[int]], heads: list[list[float]], hits: set[int]
    ) -> np.ndarray:
        img = frame.copy()
        if settings.privacy_blur:
            blur_regions(img, heads)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            colour = (60, 106, 255) if i in hits else (180, 180, 180)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3 if i in hits else 1)
        scale = 640 / img.shape[1]
        return cv2.resize(img, (640, int(img.shape[0] * scale)))

    def _raise(
        self,
        w: Watch,
        cam: Camera,
        frame: np.ndarray,
        boxes: list[list[int]],
        heads: list[list[float]],
        i: int,
        z: float,
        signal: str,
    ) -> None:
        blurred = frame.copy()
        if settings.privacy_blur:
            blur_regions(blurred, heads)
        x1, y1, x2, y2 = boxes[i]
        snap = blurred[max(0, y1) : y2, max(0, x1) : x2].copy()
        full = self._annotate(frame, boxes, heads, {i})
        alert = Alert(
            id=next(self._ids),
            watch_id=w.id,
            query=w.query,
            camera_id=cam["id"],
            camera=cam.get("name", cam["id"]),
            time=datetime.now().isoformat(timespec="seconds"),
            z=z,
            signal=signal,
            snapshot=to_jpeg(snap),
            frame=to_jpeg(full),
        )
        with self._lock:
            self.alerts.append(alert)