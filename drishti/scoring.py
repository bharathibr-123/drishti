"""Pure-numpy scoring: turns raw CLIP similarities into ranked, labelled people.

Kept free of model code so it is unit-tested in CI and reused by search, live alerts and the
evaluation ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .records import Label

STRONG_Z = 2.5  # stands out clearly from everyone else in the footage
POSSIBLE_Z = 1.5
POSSIBLE_REL = 0.5  # a "possible" must also score at least half as high as the best match
SOFTMIN_TEMPERATURE = 50.0

FloatArray = np.ndarray


@dataclass(frozen=True)
class ScoringOptions:
    """Each switch is one idea; the evaluation ablates them one at a time."""

    prompt_ensemble: bool = True  # embed the query in several CCTV-style phrasings
    background_norm: bool = True  # subtract similarity to a generic "a person" prompt
    # Experimental: every attribute in the description must match. Did not help on our
    # benchmark (eval/RESULTS.md), so it is off by default.
    attribute_and: bool = False
    track_top_k: int = 2  # score a person by the mean of their best k views
    # Tuned on eval/benchmark.json (python -m drishti.evaluate): subtracting half the generic
    # similarity, with k=2 views, gave the best Recall@1, MRR and trail precision.
    background_weight: float = 0.5  # how much of the generic-prompt similarity to subtract
    attribute_weight: float = 0.25  # blend between whole-query and all-attributes scores

    @property
    def name(self) -> str:
        on = [
            n
            for n, flag in (
                ("ensemble", self.prompt_ensemble),
                ("bg-norm", self.background_norm),
                ("attr-AND", self.attribute_and),
            )
            if flag
        ]
        return "+".join(on) or "plain CLIP"


def softmin(x: FloatArray, temperature: float = SOFTMIN_TEMPERATURE, axis: int = 0) -> FloatArray:
    """Smooth minimum: close to min(x) but differentiable and tolerant of one noisy value."""
    return np.asarray(-np.log(np.exp(-x * temperature).mean(axis=axis)) / temperature)


def combine(
    full: FloatArray,
    generic: FloatArray,
    attributes: FloatArray | None,
    opts: ScoringOptions,
) -> FloatArray:
    """Per-crop score from similarities to the full query, the generic prompt and each attribute.

    full, generic: (n_crops,)   attributes: (n_attrs, n_crops) or None
    """
    base = opts.background_weight * generic if opts.background_norm else np.zeros_like(full)
    score = full - base
    if opts.attribute_and and attributes is not None and len(attributes):
        w = opts.attribute_weight
        score = (1 - w) * score + w * softmin(attributes - base)
    return np.asarray(score)


@dataclass(frozen=True)
class RankedTrack:
    key: str
    score: float
    z: float
    best_crop: int
    label: Label


def label_for(z: float, top_z: float) -> Label:
    if z >= STRONG_Z:
        return "strong"
    if z >= POSSIBLE_Z and z >= POSSIBLE_REL * top_z:
        return "possible"
    return "weak"


def rank_tracks(
    crop_scores: FloatArray, crop_tracks: Sequence[str], top_k_views: int = 2
) -> list[RankedTrack]:
    """Aggregate crop scores per person and label each by how far they stand out (z-score)."""
    if len(crop_scores) == 0:
        return []
    by_track: dict[str, list[tuple[float, int]]] = {}
    for i, (s, key) in enumerate(zip(crop_scores.tolist(), crop_tracks, strict=True)):
        by_track.setdefault(key, []).append((float(s), i))

    agg: list[tuple[float, str, int]] = []
    for key, views in by_track.items():
        views.sort(reverse=True)
        agg.append((float(np.mean([s for s, _ in views[:top_k_views]])), key, views[0][1]))
    agg.sort(reverse=True)

    vals = np.array([a[0] for a in agg])
    centre, spread = float(np.median(vals)), float(vals.std() + 1e-6)
    zs = [(s - centre) / spread for s, _, _ in agg]
    top_z = zs[0]
    return [
        RankedTrack(key=key, score=s, z=z, best_crop=crop, label=label_for(z, top_z))
        for (s, key, crop), z in zip(agg, zs, strict=True)
    ]


def calibrate(track_scores: FloatArray) -> tuple[float, float]:
    """Median and spread of a query's scores over the archive, used to judge live sightings."""
    return float(np.median(track_scores)), float(track_scores.std() + 1e-6)
