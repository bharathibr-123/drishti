from drishti.records import Hit, Label
from drishti.trail import build_trail


def _hit(cam: str, start: str, end: str, label: Label = "strong", look: float = 0.6) -> Hit:
    return Hit(
        track=cam,
        camera_id=cam,
        camera=cam,
        score=1.0,
        z=3.0,
        label=label,
        same_look=look,
        on_trail=False,
        thumb="x",
        t=0.0,
        t_start=0.0,
        t_end=1.0,
        clock=start,
        clock_start=start,
        clock_end=end,
        fragments=1,
    )


def test_trail_orders_and_merges_sightings() -> None:
    trail, last = build_trail(
        [
            _hit("C3", "T19:40", "T19:41", look=1.0),
            _hit("C1", "T19:30", "T19:31"),
            _hit("C1", "T19:31", "T19:32"),
            _hit("C2", "T19:35", "T19:36", "weak"),
        ]
    )
    assert [t["camera_id"] for t in trail] == ["C1", "C3"]
    assert trail[0]["clock_end"] == "T19:32"
    assert last is not None and last["camera_id"] == "C3"


def test_different_looking_person_never_becomes_last_seen() -> None:
    hits = [
        _hit("C2", "T19:31", "T19:32", look=1.0),
        _hit("C5", "T19:36", "T19:37", "possible", look=0.1),
    ]
    trail, last = build_trail(hits)
    assert [t["camera_id"] for t in trail] == ["C2"]
    assert last is not None and last["camera_id"] == "C2"
    assert hits[0]["on_trail"] and not hits[1]["on_trail"]


def test_no_trail_when_best_match_is_weak() -> None:
    assert build_trail([_hit("C1", "T1", "T2", "weak", look=1.0)]) == ([], None)


def test_relative_recording_start_for_demo_cameras() -> None:
    from datetime import timedelta

    from drishti.search import _STARTED, clock, recording_start

    assert recording_start("-00:14:00") == _STARTED - timedelta(minutes=14)
    assert recording_start("2026-10-20T19:30:00") is not None
    assert recording_start(None) is None
    assert clock({"start_time": "2026-10-20T19:30:00"}, 90.0) == "2026-10-20T19:31:30"
