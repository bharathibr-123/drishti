import numpy as np

from drishti.privacy import blur_crop, blur_regions, head_box


def test_head_from_face_keypoints_covers_a_close_up_face() -> None:
    # close-up: person box 64..339 x 73..427, face keypoints around x 142-238, y 144-165
    person = [64, 73, 339, 427]
    kpts = [[168, 165, 1], [194, 144, 1], [154, 147, 1], [238, 162, 1], [142, 164, 0]]
    x1, y1, x2, y2 = head_box(person, kpts)
    assert x1 < 154 and x2 > 238  # both eyes and the visible ear are inside
    assert y1 < 100 and y2 > 200  # well above the eyes and below the mouth
    # the old fixed "top 22% of the box" rule would have stopped at y=151, mid-face
    assert y2 > 73 + 0.22 * (427 - 73)


def test_head_falls_back_to_top_band_without_keypoints() -> None:
    assert head_box([0, 0, 50, 200], None) == [0.0, 0.0, 50.0, 44.0]
    # only one confident keypoint (e.g. seen from behind) -> fallback as well
    one = [[25, 10, 0.9], [0, 0, 0.1], [0, 0, 0.1], [0, 0, 0.1], [0, 0, 0.1]]
    assert head_box([0, 0, 50, 200], one) == [0.0, 0.0, 50.0, 44.0]


def test_blur_changes_only_the_region() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_regions(img.copy(), [[10, 10, 40, 40]])
    assert not np.array_equal(out[10:40, 10:40], img[10:40, 10:40])
    assert np.array_equal(out[60:, 60:], img[60:, 60:])


def test_blur_crop_uses_fractional_head_box() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    out = blur_crop(img.copy(), [0.2, 0.0, 0.8, 0.5])
    assert not np.array_equal(out[:100, 20:80], img[:100, 20:80])
    assert np.array_equal(out[150:], img[150:])


def test_head_box_accepts_numpy_keypoints() -> None:
    kpts = np.array([[168, 165, 1], [194, 144, 1], [154, 147, 1], [238, 162, 1], [142, 164, 0]])
    assert head_box([64, 73, 339, 427], kpts) == head_box([64, 73, 339, 427], kpts.tolist())
