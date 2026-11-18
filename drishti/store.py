"""On-disk index: one embedding per person crop, plus camera/track/crop metadata."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .records import Camera, Crop, Track

faiss: Any
try:
    faiss = importlib.import_module("faiss")
except ImportError:  # numpy fallback keeps the project usable without faiss
    faiss = None


class Store:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.index_dir)
        (self.root / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        self.cameras: dict[str, Camera] = self._load("cameras.json", {})
        self.tracks: list[Track] = self._load("tracks.json", [])
        self.crops: list[Crop] = self._load("crops.json", [])
        emb_path = self.root / "embeddings.npy"
        self.emb: np.ndarray = (
            np.load(emb_path) if emb_path.exists() else np.zeros((0, 0), dtype=np.float32)
        )
        self._reset_caches()

    def _reset_caches(self) -> None:
        self._index: Any = None
        self._looks: dict[str, np.ndarray] | None = None
        self._crop_by_thumb: dict[str, Crop] | None = None
        self._track_by_key: dict[str, Track] | None = None

    # ------------------------------------------------------------------ persistence
    def _load(self, name: str, default: Any) -> Any:
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def save(self) -> None:
        (self.root / "cameras.json").write_text(json.dumps(self.cameras, indent=2))
        (self.root / "tracks.json").write_text(json.dumps(self.tracks))
        (self.root / "crops.json").write_text(json.dumps(self.crops))
        np.save(self.root / "embeddings.npy", self.emb.astype(np.float32))
        self._reset_caches()

    # ------------------------------------------------------------------ mutation
    def remove_camera(self, camera_id: str) -> None:
        keep = [i for i, c in enumerate(self.crops) if c["camera_id"] != camera_id]
        self.crops = [self.crops[i] for i in keep]
        if len(self.emb):
            self.emb = self.emb[keep]
        self.tracks = [t for t in self.tracks if t["camera_id"] != camera_id]
        self.cameras.pop(camera_id, None)
        self._reset_caches()

    def add(self, camera: Camera, tracks: list[Track], crops: list[Crop], emb: np.ndarray) -> None:
        self.remove_camera(camera["id"])
        self.cameras[camera["id"]] = camera
        self.tracks.extend(tracks)
        self.crops.extend(crops)
        self.emb = emb if self.emb.size == 0 else np.concatenate([self.emb, emb])
        self._reset_caches()

    # ------------------------------------------------------------------ lookups
    @property
    def crop_tracks(self) -> list[str]:
        return [c["track"] for c in self.crops]

    def crop(self, thumb: str) -> Crop | None:
        if self._crop_by_thumb is None:
            self._crop_by_thumb = {c["thumb"]: c for c in self.crops}
        return self._crop_by_thumb.get(thumb)

    def track(self, key: str) -> Track:
        if self._track_by_key is None:
            self._track_by_key = {t["key"]: t for t in self.tracks}
        return self._track_by_key[key]

    def looks(self) -> dict[str, np.ndarray]:
        """One camera-normalised appearance vector per person, for re-identification."""
        if self._looks is None:
            self._looks = track_looks(
                self.emb, self.crop_tracks, [c["camera_id"] for c in self.crops]
            )
        return self._looks

    # ------------------------------------------------------------------ search
    def similarity(self, queries: np.ndarray) -> np.ndarray:
        """Cosine similarity of each query vector against every stored crop -> (q, n)."""
        if not len(self.emb):
            return np.zeros((len(queries), 0), dtype=np.float32)
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if faiss is None:
            return np.asarray(q @ self.emb.T)
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.emb.shape[1])
            self._index.add(np.ascontiguousarray(self.emb, dtype=np.float32))
        sims, ids = self._index.search(q, len(self.emb))
        out = np.empty_like(sims)
        np.put_along_axis(out, ids, sims, axis=1)
        return out


def camera_normalise(emb: np.ndarray, crop_cameras: list[str]) -> np.ndarray:
    """Subtract each camera's mean crop embedding, then re-normalise.

    Raw CLIP vectors of two different people on the same camera are ~0.85 similar because they
    share background and lighting. Removing the camera's average leaves what differs between
    people. In our pilot this lifted cross-camera re-identification from 6/7 to 7/7.
    """
    out = emb.astype(np.float32, copy=True)
    by_cam: dict[str, list[int]] = {}
    for i, cam in enumerate(crop_cameras):
        by_cam.setdefault(cam, []).append(i)
    for idx in by_cam.values():
        out[idx] -= out[idx].mean(0)
    return np.asarray(out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9))


def track_looks(
    emb: np.ndarray, crop_tracks: list[str], crop_cameras: list[str]
) -> dict[str, np.ndarray]:
    normed = camera_normalise(emb, crop_cameras)
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(crop_tracks):
        groups.setdefault(key, []).append(i)
    looks = {}
    for key, idx in groups.items():
        v = normed[idx].mean(0)
        looks[key] = v / (np.linalg.norm(v) + 1e-9)
    return looks
