import numpy as np

from drishti.scoring import ScoringOptions, combine, label_for, rank_tracks, softmin


def test_softmin_is_close_to_min_and_tolerates_noise() -> None:
    x = np.array([[0.30, 0.10], [0.28, 0.30]])
    out = softmin(x)
    assert out[0] < 0.30 and out[0] > 0.27  # both attributes match -> stays high
    assert out[1] < 0.13  # one attribute fails -> pulled to the failing one


def test_attribute_and_penalises_partial_matches() -> None:
    full = np.array([0.30, 0.30])
    generic = np.array([0.20, 0.20])
    # crop 0 matches both attributes, crop 1 only the first
    attrs = np.array([[0.30, 0.31], [0.30, 0.19]])
    on = combine(full, generic, attrs, ScoringOptions(attribute_and=True))
    off = combine(full, generic, attrs, ScoringOptions())
    assert off[0] == off[1]  # plain scoring cannot tell them apart
    assert on[0] > on[1]


def test_background_norm_removes_crops_that_match_everything() -> None:
    # crop 1 is a big sharp crop: high similarity to the query AND to "a person"
    full = np.array([0.28, 0.30])
    generic = np.array([0.18, 0.29])
    on = combine(full, generic, None, ScoringOptions(background_weight=1.0))
    off = combine(full, generic, None, ScoringOptions(background_norm=False))
    assert off.argmax() == 1 and on.argmax() == 0


def test_rank_tracks_aggregates_views_and_labels() -> None:
    rng = np.random.default_rng(0)
    crowd = list(0.1 + 0.02 * rng.standard_normal(40))  # 20 other people, two views each
    scores = np.array([0.9, 0.85, *crowd])
    tracks = ["A", "A", *[f"P{i // 2}" for i in range(40)]]
    ranked = rank_tracks(scores, tracks)
    assert ranked[0].key == "A"
    assert ranked[0].label == "strong"
    assert all(r.label == "weak" for r in ranked[1:])


def test_possible_needs_to_be_close_to_the_best() -> None:
    assert label_for(1.7, top_z=4.2) == "weak"  # far behind a clear winner
    assert label_for(1.7, top_z=2.6) == "possible"
    assert label_for(2.6, top_z=2.6) == "strong"


def test_options_name_for_ablation_tables() -> None:
    assert ScoringOptions().name == "ensemble+bg-norm"
    assert ScoringOptions(attribute_and=True).name == "ensemble+bg-norm+attr-AND"
    plain = ScoringOptions(prompt_ensemble=False, background_norm=False, attribute_and=False)
    assert plain.name == "plain CLIP"


def test_live_alert_threshold_is_clamped_and_relative() -> None:
    from drishti.live import MIN_ALERT_Z, alert_threshold
    from drishti.scoring import STRONG_Z

    assert alert_threshold(10.0) == STRONG_Z  # a very distinctive description: full bar
    assert alert_threshold(1.0) == MIN_ALERT_Z  # vague description: never below the floor
    assert abs(alert_threshold(2.8) - 2.1) < 1e-9  # in between: relative to the archive


def test_live_alert_needs_description_or_description_plus_appearance() -> None:
    from drishti.live import should_alert

    assert should_alert(2.6, None, threshold=2.0) == "description"
    assert should_alert(1.5, None, threshold=2.0) is None  # moderate match alone: no alert
    assert should_alert(1.5, 0.45, threshold=2.0) == "description + appearance"
    assert should_alert(1.5, 0.10, threshold=2.0) is None  # looks like someone else
    assert should_alert(0.5, 0.90, threshold=2.0) is None  # lookalike in other clothes


def test_same_description_is_not_watched_twice() -> None:
    import numpy as np

    from drishti.live import Watch, same_query_watch
    from drishti.query import parse

    w = Watch(1, "A man in a  dark blue t-shirt", parse("a man"), 0.0, 1.0, 2.0, None, "o", "r")
    assert same_query_watch([w], "a man in a dark blue T-shirt ") is w
    assert same_query_watch([w], "a woman in a teal jacket") is None
    assert np.isfinite(w.threshold)