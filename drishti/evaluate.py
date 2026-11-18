"""Benchmark and ablation.

    python -m drishti.evaluate                      # eval/benchmark.json -> eval/RESULTS.md

Ground truth marks each person who fits a description with an anchor: (camera, time, x, y), the
centre of that person in the frame at that moment (x, y as fractions of the frame). A result is
relevant when its tracked path passes through the anchor. This identifies people, not time windows,
so someone standing next to the right person does not count, and it does not depend on tracker IDs.
Reported per configuration:

  * Recall@1 / Recall@5 - a relevant person is ranked first / in the top five
  * MRR                 - mean reciprocal rank of the first relevant person
  * Trail precision     - share of trail stops that really are the described person
  * Last-seen accuracy  - LAST SEEN points at the right person (queries that produce a trail)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .config import ROOT
from .query import ParsedQuery, parse
from .records import Hit
from .scoring import ScoringOptions
from .search import SearchEngine
from .store import Store
from .trail import build_trail

Anchor = tuple[str, float, float, float]  # camera, t (s), centre x, centre y (fractions)
ANCHOR_TIME_S = 1.0
ANCHOR_DIST = 0.12

CONFIGS: list[tuple[str, ScoringOptions]] = [
    ("Plain CLIP", ScoringOptions(prompt_ensemble=False, background_norm=False)),
    ("Prompt ensemble only", ScoringOptions(background_norm=False)),
    ("Background normalisation only", ScoringOptions(prompt_ensemble=False)),
    ("**Both = Drishti default**", ScoringOptions()),
    ("+ all-attributes AND (experimental, off)", ScoringOptions(attribute_and=True)),
]


@dataclass
class QueryOutcome:
    query: str
    rank: int | None  # 1-based rank of the first relevant person, None if not in results
    trail_stops: int
    trail_correct: int
    last_seen_ok: bool | None


def distance_to(path: list[list[float]], t: float, x: float, y: float) -> float:
    """Closest approach of a tracked path to (x, y) within ANCHOR_TIME_S of t."""
    near = [
        ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        for pt, px, py in path
        if abs(pt - t) <= ANCHOR_TIME_S
    ]
    return min(near, default=float("inf"))


def resolve(store: Store, anchor: Anchor) -> str | None:
    """The one indexed person an anchor refers to: the nearest path, if within ANCHOR_DIST."""
    cam, t, x, y = anchor
    best = min(
        (
            (distance_to(tr.get("path", []), t, x, y), tr["key"])
            for tr in store.tracks
            if tr["camera_id"] == cam
        ),
        default=(float("inf"), None),
    )
    return best[1] if best[0] <= ANCHOR_DIST else None


def relevant(engine: SearchEngine, hit: Hit, gt: list[Anchor]) -> bool:
    return hit["track"] in {resolve(engine.store, a) for a in gt}


def run_query(
    engine: SearchEngine, q: ParsedQuery, gt: list[Anchor], opts: ScoringOptions
) -> QueryOutcome:
    hits = engine._hits(engine.rank(engine.score_crops(q, opts), opts)[:20])
    rank = next((i + 1 for i, h in enumerate(hits) if relevant(engine, h, gt)), None)
    _, last = build_trail(hits)
    on_trail = [h for h in hits if h["on_trail"]]
    return QueryOutcome(
        query=q.original,
        rank=rank,
        trail_stops=len(on_trail),
        trail_correct=sum(relevant(engine, h, gt) for h in on_trail),
        last_seen_ok=None
        if last is None
        else any(relevant(engine, h, gt) for h in on_trail if h["track"] == last["track"]),
    )


def anchors(item: dict[str, object]) -> list[Anchor]:
    people = item["people"]
    assert isinstance(people, list)
    return [(str(c), float(t), float(x), float(y)) for c, t, x, y in people]


def summarise(outcomes: list[QueryOutcome]) -> dict[str, float]:
    n = len(outcomes)
    stops = sum(o.trail_stops for o in outcomes)
    judged = [o.last_seen_ok for o in outcomes if o.last_seen_ok is not None]
    return {
        "R@1": sum(o.rank == 1 for o in outcomes) / n,
        "R@5": sum(o.rank is not None and o.rank <= 5 for o in outcomes) / n,
        "MRR": sum(1 / o.rank for o in outcomes if o.rank) / n,
        "trail_precision": (sum(o.trail_correct for o in outcomes) / stops) if stops else 0.0,
        "last_seen": (sum(judged) / len(judged)) if judged else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark", default=str(ROOT / "eval" / "benchmark.json"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "RESULTS.md"))
    args = ap.parse_args()
    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))["queries"]
    engine = SearchEngine(Store())
    missing = [(b["q"], a) for b in bench for a in anchors(b) if resolve(engine.store, a) is None]
    if missing:
        raise SystemExit(f"benchmark anchors match no indexed person (re-label?): {missing}")

    rows: list[tuple[str, dict[str, float]]] = []
    final: list[QueryOutcome] = []
    default = ScoringOptions()
    for name, opts in CONFIGS:
        outs = [run_query(engine, parse(b["q"]), anchors(b), opts) for b in bench]
        rows.append((name, summarise(outs)))
        if opts == default:
            final = outs

    # Kannada: the same queries with and without the offline translator
    kn = [b for b in bench if parse(b["q"]).language != "en"]
    no_tr = [
        run_query(
            engine,
            replace(parse(b["q"]), english=b["q"], attributes=()),
            anchors(b),
            ScoringOptions(),
        )
        for b in kn
    ]
    with_tr = [run_query(engine, parse(b["q"]), anchors(b), ScoringOptions()) for b in kn]

    lines = [
        "# Drishti benchmark results",
        "",
        f"{len(bench)} labelled descriptions over {len(engine.store.cameras)} cameras and "
        f"{len(engine.store.tracks)} indexed people (CPU, `{engine.store.root.name}` index). "
        "Generated by `python -m drishti.evaluate`.",
        "",
        "## Ablation: what each scoring idea contributes",
        "",
        "| Configuration | Recall@1 | Recall@5 | MRR | Trail precision | Last-seen correct |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['R@1']:.0%} | {m['R@5']:.0%} | {m['MRR']:.2f} | "
            f"{m['trail_precision']:.0%} | {m['last_seen']:.0%} |"
        )
    a, b = summarise(no_tr), summarise(with_tr)
    lines += [
        "",
        f"## Kannada and Kanglish queries ({len(kn)})",
        "",
        "| | Recall@1 | Recall@5 | MRR |",
        "|---|---|---|---|",
        f"| Raw text sent to CLIP | {a['R@1']:.0%} | {a['R@5']:.0%} | {a['MRR']:.2f} |",
        f"| With Drishti's offline translator | {b['R@1']:.0%} | {b['R@5']:.0%} | {b['MRR']:.2f} |",
        "",
        "## Per query (full system)",
        "",
        "| Query | Rank of first correct person | Trail stops (correct) | Last seen |",
        "|---|---|---|---|",
    ]
    for o in final:
        ls = "-" if o.last_seen_ok is None else ("✓" if o.last_seen_ok else "✗")
        lines.append(
            f"| {o.query} | {o.rank or 'not in top 20'} | "
            f"{o.trail_stops} ({o.trail_correct}) | {ls} |"
        )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: 14 + len(rows)]))
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
