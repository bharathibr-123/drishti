"""Reconstruct where the #1 match went, across cameras, in time order."""

from __future__ import annotations

from .records import Hit, TrailStop

# Camera-normalised appearance similarity to the #1 match needed to join its trail. On our
# benchmark (ViT-B/16+), the 7 people seen on two cameras scored 0.35-0.68 against themselves,
# while different people scored at most 0.25 across cameras and 0.11 on the same camera.
SAME_PERSON_SIM = 0.30


def build_trail(results: list[Hit]) -> tuple[list[TrailStop], TrailStop | None]:
    """A sighting joins the trail only if it is a confident match AND looks like the same person
    as the #1 result, so a stranger who matches part of the description never becomes LAST SEEN.

    Marks ``on_trail`` on the hits it uses. Returns (trail, last_seen).
    """
    for r in results:
        r["on_trail"] = False
    if not results or results[0]["label"] == "weak":
        return [], None

    seen = [
        r
        for i, r in enumerate(results)
        if r["clock_start"]
        and (i == 0 or (r["label"] != "weak" and r["same_look"] >= SAME_PERSON_SIM))
    ]
    for r in seen:
        r["on_trail"] = True
    seen.sort(key=lambda r: r["clock_start"] or "")

    trail: list[TrailStop] = []
    for r in seen:
        start, end = r["clock_start"] or "", r["clock_end"] or ""
        if trail and trail[-1]["camera_id"] == r["camera_id"]:
            trail[-1]["clock_end"] = max(trail[-1]["clock_end"], end)
            continue
        trail.append(
            TrailStop(
                camera_id=r["camera_id"],
                camera=r["camera"],
                clock_start=start,
                clock_end=end,
                thumb=r["thumb"],
                track=r["track"],
            )
        )
    return trail, (trail[-1] if trail else None)
