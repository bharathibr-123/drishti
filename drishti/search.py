"""Natural-language and photo search over indexed people.

Pipeline: parse the query (Kannada -> English, split attributes) -> embed it -> score every
crop (scoring.combine) -> aggregate per person (scoring.rank_tracks) -> add the same-person check
and the movement trail (trail.build_trail).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from .models import Encoder, get_encoder
from .query import ParsedQuery, parse
from .records import Camera, Hit, SearchResult
from .scoring import RankedTrack, ScoringOptions, combine, rank_tracks
from .store import Store
from .trail import build_trail

TEMPLATES = (
    "a photo of {}.",
    "a CCTV photo of {}.",
    "a low resolution surveillance photo of {}.",
)
GENERIC = "a person"


class SearchEngine:
    def __init__(self, store: Store, encoder: Encoder | None = None) -> None:
        self.store = store
        self._encoder = encoder
        self._text_cache: dict[tuple[str, bool], np.ndarray] = {}

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = get_encoder()
        return self._encoder

    # ------------------------------------------------------------------ embeddings
    def text_vector(self, phrase: str, ensemble: bool = True) -> np.ndarray:
        key = (phrase, ensemble)
        if key not in self._text_cache:
            prompts = [t.format(phrase) for t in TEMPLATES] if ensemble else [phrase]
            v = self.encoder.embed_texts(prompts).mean(0)
            self._text_cache[key] = v / np.linalg.norm(v)
        return self._text_cache[key]

    def query_vectors(self, q: ParsedQuery, opts: ScoringOptions) -> np.ndarray:
        """Rows: full query, generic prompt, then one row per attribute."""
        phrases = [q.english, GENERIC, *q.attributes]
        return np.stack([self.text_vector(p, opts.prompt_ensemble) for p in phrases])

    # ------------------------------------------------------------------ scoring
    def score_crops(self, q: ParsedQuery, opts: ScoringOptions) -> np.ndarray:
        sims = self.store.similarity(self.query_vectors(q, opts))
        return combine(sims[0], sims[1], sims[2:] if q.attributes else None, opts)

    def score_against(
        self, crop_emb: np.ndarray, q: ParsedQuery, opts: ScoringOptions
    ) -> np.ndarray:
        """Same scoring for crops that are not in the index (live camera)."""
        sims = self.query_vectors(q, opts) @ crop_emb.T
        return combine(sims[0], sims[1], sims[2:] if q.attributes else None, opts)

    def rank(self, scores: np.ndarray, opts: ScoringOptions | None = None) -> list[RankedTrack]:
        k = (opts or ScoringOptions()).track_top_k
        return rank_tracks(scores, self.store.crop_tracks, k)

    # ------------------------------------------------------------------ public API
    def search(
        self,
        query: str = "",
        image: Image.Image | None = None,
        top_k: int = 12,
        opts: ScoringOptions | None = None,
    ) -> SearchResult:
        opts = opts or ScoringOptions()
        q = parse(query)
        if image is not None:
            scores = self.store.similarity(self.encoder.embed_images([image]))[0]
            if query.strip():
                scores = scores + 0.5 * self.score_crops(q, opts)
            attributes: list[str] = []
        else:
            scores = self.score_crops(q, opts)
            attributes = list(q.attributes) if opts.attribute_and else []
        hits = self._hits(self.rank(scores, opts)[:top_k])
        trail, last_seen = build_trail(hits)
        return SearchResult(
            query=query,
            understood_as=q.english,
            language=q.language,
            attributes=attributes,
            results=hits,
            trail=trail,
            last_seen=last_seen,
        )

    def similar(self, track_key: str, top_k: int = 6) -> list[tuple[str, float]]:
        """People on OTHER cameras who look like this one (camera-normalised re-identification)."""
        looks = self.store.looks()
        cam = self.store.track(track_key)["camera_id"]
        me = looks[track_key]
        others = [
            (k, float(me @ v)) for k, v in looks.items() if self.store.track(k)["camera_id"] != cam
        ]
        others.sort(key=lambda kv: -kv[1])
        return others[:top_k]

    def _hits(self, ranked: list[RankedTrack]) -> list[Hit]:
        if not ranked:
            return []
        looks = self.store.looks()
        top = looks[ranked[0].key]
        hits: list[Hit] = []
        for r in ranked:
            t = self.store.track(r.key)
            c = self.store.crops[r.best_crop]
            cam = self.store.cameras[t["camera_id"]]
            hits.append(
                Hit(
                    track=r.key,
                    camera_id=t["camera_id"],
                    camera=cam.get("name", t["camera_id"]),
                    score=round(r.score, 4),
                    z=round(r.z, 2),
                    label=r.label,
                    same_look=round(float(top @ looks[r.key]), 3),
                    on_trail=False,
                    thumb=c["thumb"],
                    t=c["t"],
                    t_start=t["t_start"],
                    t_end=t["t_end"],
                    clock=clock(cam, c["t"]),
                    clock_start=clock(cam, t["t_start"]),
                    clock_end=clock(cam, t["t_end"]),
                )
            )
        return hits


_STARTED = datetime.now().replace(microsecond=0)


def recording_start(start_time: str | None) -> datetime | None:
    """A camera's recording start: an ISO time, or "-HH:MM:SS" meaning that long before the app
    started (used by the demo so archive sightings are minutes old, like a real incident)."""
    if not start_time:
        return None
    if start_time.startswith("-"):
        h, m, s = (int(v) for v in start_time[1:].split(":"))
        return _STARTED - timedelta(hours=h, minutes=m, seconds=s)
    return datetime.fromisoformat(start_time)


def clock(cam: Camera, t: float) -> str | None:
    start = recording_start(cam.get("start_time"))
    return None if start is None else (start + timedelta(seconds=t)).isoformat()
