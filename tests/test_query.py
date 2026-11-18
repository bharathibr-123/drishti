import pytest

from drishti.kannada import detect_language, translate
from drishti.query import parse, split_attributes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "ಹಳದಿ ಟೀ ಶರ್ಟ್ ಹಾಕಿರುವ ಹುಡುಗ, ಕೆಂಪು ಬಲೂನ್ ಹಿಡಿದಿದ್ದಾನೆ",
            "a boy wearing a yellow t-shirt, holding a red balloon",
        ),
        (
            "haladi t-shirt hakiruva huduga, kempu balloon hididdane",
            "a boy wearing a yellow t-shirt, holding a red balloon",
        ),
        ("೬ ವರ್ಷದ ಹುಡುಗಿ ಗುಲಾಬಿ ಫ್ರಾಕ್", "a 6 year old girl wearing a pink frock"),
        ("ಬಿಳಿ ಹೂವಿನ ಟೀ ಶರ್ಟ್ ಹಾಕಿರುವ ಹೆಂಗಸು", "a woman wearing a white floral t-shirt"),
        ("neeli shirt jeans gandasu", "a man wearing a blue shirt, wearing jeans"),
        ("ಕಪ್ಪು ಕನ್ನಡಕ ಹಾಕಿದ್ದ ಅಜ್ಜ", "an old man wearing black glasses"),
        ("yellow t-shirt huduga", "a boy wearing a yellow t-shirt"),  # code-mixed
        (
            "ಕಪ್ಪು ಉಡುಪು ಹಾಕಿರುವ ಉದ್ದ ಕೂದಲಿನ ಹೆಂಗಸು",
            "a woman wearing a black dress, with long hair",
        ),
        ("ಬೋಳು ತಲೆಯ ಕನ್ನಡಕ ಹಾಕಿರುವ ಗಂಡಸು", "a bald man wearing glasses"),
    ],
)
def test_translate_kannada_and_kanglish(text: str, expected: str) -> None:
    assert translate(text) == expected


def test_english_passes_through_unchanged() -> None:
    assert translate("a woman in a teal jacket") == "a woman in a teal jacket"
    assert detect_language("a woman in a teal jacket") == "en"
    # regression: English fillers/synonyms in the lexicon must not flip detection
    for q in [
        "a man in a black sweater with white stripes on the sleeves",
        "an old man with a little bag and a long dress",
        "a bald man in a cap holding a balloon",
    ]:
        assert detect_language(q) == "en", q


def test_language_detection() -> None:
    assert detect_language("ಕೆಂಪು ಅಂಗಿ") == "kn"
    assert detect_language("kempu angi huduga") == "kn-latn"


def test_split_attributes_keeps_connectors() -> None:
    parts = split_attributes("little boy wearing a yellow t-shirt, holding a red balloon")
    assert parts == [
        "little boy",
        "a person wearing a yellow t-shirt",
        "a person holding a red balloon",
    ]


def test_single_attribute_query_is_not_split() -> None:
    assert split_attributes("a man in a hat") == ["a man", "a person in a hat"]
    assert split_attributes("a small child") == []


def test_parse_kannada_query_end_to_end() -> None:
    q = parse("ಕೆಂಪು ಅಂಗಿ ಹಾಕಿರುವ ಹುಡುಗ")
    assert q.language == "kn"
    assert q.english == "a boy wearing a red shirt"
    assert q.attributes == ("a boy", "a person wearing a red shirt")
