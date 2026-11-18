"""Understand descriptions given in Kannada, romanised Kannada ("Kanglish") or a mix with English.

A worried parent describes their child in the language they think in:

    "ಹಳದಿ ಟೀ ಶರ್ಟ್ ಹಾಕಿರುವ ಹುಡುಗ, ಕೆಂಪು ಬಲೂನ್ ಹಿಡಿದಿದ್ದಾನೆ"
    "haladi t-shirt hakiruva huduga, kempu balloon hididdane"

Both become  "a boy wearing a yellow t-shirt, holding a red balloon",  which the vision model
understands. This runs offline with no API: a lexicon tags each word (colour, garment, object,
person, verb...), and a small rule-based parser turns Kannada's subject-object-verb order into an
English description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- lexicon
# word -> (tag, english). Tags: C colour, G garment, O object, H hair, P person, V verb,
# M modifier of the next noun, S modifier of the person ("bald man"), X filler (dropped).
# Romanised spellings vary, so common variants are listed.

# fmt: off
_RAW: dict[str, list[str]] = {
    "C:red": ["ಕೆಂಪು", "ಕೆಂಪಿನ", "kempu", "kempina", "kempane"],
    "C:yellow": ["ಹಳದಿ", "haladi", "haldi", "haladhi", "aladi"],
    "C:blue": ["ನೀಲಿ", "neeli", "nili", "neelee"],
    "C:green": ["ಹಸಿರು", "ಹಸಿರಿನ", "hasiru", "hasirina", "hasru"],
    "C:white": ["ಬಿಳಿ", "ಬಿಳಿಯ", "bili", "biliya", "bilee", "bile"],
    "C:black": ["ಕಪ್ಪು", "ಕಪ್ಪಿನ", "kappu", "kappina"],
    "C:orange": ["ಕಿತ್ತಳೆ", "ಕೇಸರಿ", "kittale", "kithale", "kesari"],
    "C:pink": ["ಗುಲಾಬಿ", "gulabi"],
    "C:purple": ["ನೇರಳೆ", "nerale"],
    "C:brown": ["ಕಂದು", "kandu"],
    "C:grey": ["ಬೂದು", "boodu", "budu"],
    "C:dark": ["ಗಾಢ", "gaadha", "gada"],
    "C:blonde": ["ಹೊಂಬಣ್ಣದ", "hombannada", "blonde"],
    "G:shirt": ["ಅಂಗಿ", "ಶರ್ಟ್", "angi", "shirt", "sharttu"],
    "G:t-shirt": ["ಟೀಶರ್ಟ್", "ಟಿಶರ್ಟ್", "tshirt", "t-shirt", "teeshirt"],
    "G:trousers": ["ಪ್ಯಾಂಟ್", "ಪ್ಯಾಂಟು", "pant", "pants", "pantu"],
    "G:jeans": ["ಜೀನ್ಸ್", "jeans"],
    "G:saree": ["ಸೀರೆ", "seere", "sire", "saree", "sari"],
    "G:churidar": ["ಚೂಡಿದಾರ್", "ಚೂಡಿದಾರ", "churidar", "chudidar"],
    "G:skirt": ["ಲಂಗ", "ಸ್ಕರ್ಟ್", "langa", "skirt"],
    "G:frock": ["ಫ್ರಾಕ್", "frock"],
    "G:dress": ["ಉಡುಪು", "ಡ್ರೆಸ್", "udupu", "dress"],
    "G:jacket": ["ಜಾಕೆಟ್", "jacket"],
    "G:sweater": ["ಸ್ವೆಟರ್", "sweater"],
    "G:vest": ["ವೆಸ್ಟ್", "vest"],
    "G:shorts": ["ಚಡ್ಡಿ", "chaddi", "shorts"],
    "G:cap": ["ಟೋಪಿ", "ಕ್ಯಾಪ್", "topi", "cap"],
    "G:hat": ["hat"],
    "G:helmet": ["ಹೆಲ್ಮೆಟ್", "helmet"],
    "G:glasses": ["ಕನ್ನಡಕ", "kannadaka", "glasses", "specs"],
    "G:dupatta": ["ದುಪಟ್ಟಾ", "ವೇಲ್", "dupatta", "veil"],
    "H:hair": ["ಕೂದಲು", "ಕೂದಲಿನ", "ಕೂದಲಿರುವ", "koodalu", "kudalu", "koodalina", "hair"],
    "O:balloon": ["ಬಲೂನ್", "ಬಲೂನು", "balloon", "baloon", "balunu"],
    "O:bag": ["ಚೀಲ", "ಬ್ಯಾಗ್", "cheela", "chila", "bag"],
    "O:umbrella": ["ಕೊಡೆ", "ಛತ್ರಿ", "kode", "chatri", "umbrella"],
    "O:bottle": ["ಬಾಟಲ್", "ಬಾಟಲಿ", "bottle", "baatli"],
    "O:toy": ["ಆಟಿಕೆ", "aatike", "toy"],
    "O:phone": ["ಫೋನ್", "phone", "mobile"],
    "P:boy": ["ಹುಡುಗ", "ಹುಡುಗನು", "huduga", "hudga"],
    "P:girl": ["ಹುಡುಗಿ", "ಹುಡುಗಿಯು", "hudugi", "hudgi"],
    "P:child": ["ಮಗು", "ಮಗುವು", "ಮಗುವಿನ", "magu", "maguvu"],
    "P:woman": ["ಹೆಂಗಸು", "ಮಹಿಳೆ", "ಹೆಣ್ಣು", "hengasu", "mahile", "hennu"],
    "P:man": ["ಗಂಡಸು", "ಪುರುಷ", "ಗಂಡು", "gandasu", "purusha", "gandu"],
    "P:old man": ["ಅಜ್ಜ", "ತಾತ", "ajja", "thatha", "tata"],
    "P:old woman": ["ಅಜ್ಜಿ", "ajji"],
    "P:worker": ["ಕೆಲಸಗಾರ", "kelasagara"],
    "V:wearing": [
        "ಹಾಕಿರುವ", "ಹಾಕಿದ್ದ", "ಹಾಕಿದ್ದಾನೆ", "ಹಾಕಿದ್ದಾಳೆ", "ಧರಿಸಿರುವ", "ಧರಿಸಿದ್ದ", "ತೊಟ್ಟಿರುವ",
        "ಉಟ್ಟಿರುವ", "hakiruva", "haakiruva", "hakidda", "hakiddane", "hakiddale", "dharisiruva",
        "tottiruva", "uttiruva", "hakondiro", "hakkondiro"
    ],
    "V:holding": [
        "ಹಿಡಿದಿರುವ", "ಹಿಡಿದಿದ್ದ", "ಹಿಡಿದಿದ್ದಾನೆ", "ಹಿಡಿದಿದ್ದಾಳೆ", "ಹಿಡಿದುಕೊಂಡಿರುವ",
        "hididiruva", "hididda", "hididdane", "hididdale", "hidkondiro", "hidkondidda"
    ],
    "V:carrying": ["ಹೊತ್ತಿರುವ", "ಹೊತ್ತಿದ್ದ", "hottiruva", "hottidda"],
    "V:with": ["ಜೊತೆ", "ಜೊತೆಗೆ", "ಇರುವ", "jote", "jothe", "iruva"],
    "M:small": ["ಚಿಕ್ಕ", "ಸಣ್ಣ", "chikka", "sanna", "small", "little"],
    "M:long": ["ಉದ್ದ", "ಉದ್ದದ", "udda", "uddha", "uddada", "long"],
    "M:short": ["ಗಿಡ್ಡ", "gidda", "short"],
    "M:curly": ["ಗುಂಗುರು", "gunguru", "curly"],
    "M:floral": ["ಹೂವಿನ", "ಹೂವಿರುವ", "hoovina", "huvina"],
    "M:striped": ["ಪಟ್ಟೆ", "ಗೆರೆ", "patte", "gere"],
    "M:checked": ["ಚೌಕಳಿ", "chaukali"],
    "X:": [
        "ಬಣ್ಣದ", "ಬಣ್ಣದ್ದು", "ಮತ್ತು", "ಒಂದು", "ಒಬ್ಬ", "ಒಬ್ಬಳು", "banna", "bannada", "mattu",
        "ondu", "obba", "obbalu", "a", "an", "the", "ತಲೆಯ", "taleya"
    ],
    "S:tall": ["ಎತ್ತರದ", "ettarada", "tall"],
    "S:bald": ["ಬೋಳು", "bolu", "bald"],
    "S:old": ["ವಯಸ್ಸಾದ", "vayassada", "old"],
}
# fmt: on
_YEARS = {"ವರ್ಷದ", "ವರ್ಷ", "varshada", "varsha", "year", "years", "yr"}

LEXICON: dict[str, tuple[str, str]] = {}
for _key, _words in _RAW.items():
    _tag, _english = _key.split(":", 1)
    for _w in _words:
        LEXICON[_w.lower()] = (_tag, _english)

# English words people mix in ("yellow t-shirt huduga"). Tagged so the parser can place them.
_ENGLISH_TAGS = {
    "C": "red yellow blue green white black orange pink purple brown grey gray dark navy maroon",
    "G": "shirt t-shirt top kurta trousers jeans saree skirt frock dress jacket sweater vest "
    "shorts cap hat helmet glasses leggings hoodie uniform",
    "O": "balloon bag umbrella bottle toy phone backpack",
    "P": "boy girl child kid woman man lady person worker",
}
for _tag, _words_str in _ENGLISH_TAGS.items():
    for _w in _words_str.split():
        LEXICON.setdefault(_w, (_tag, _w))

_KANNADA_SCRIPT = re.compile(r"[ಀ-೿]")
# Ordinary English words in the lexicon (fillers, synonyms) must not make an English query
# look like romanised Kannada: "on the sleeves" is English.
_ENGLISH_WORDS = {w for words in _ENGLISH_TAGS.values() for w in words.split()} | {
    "the",
    "an",
    "a",
    "little",
    "small",
    "tall",
    "long",
    "short",
    "curly",
    "bald",
    "old",
    "blonde",
    "hair",
    "specs",
    "mobile",
    "veil",
    "pant",
    "pants",
    "dress",
    "skirt",
    "frock",
    "sari",
    "saree",
    "toy",
    "tshirt",
    "teeshirt",
    "baloon",
}
_ROMAN_KANNADA = {w for w in LEXICON if w.isascii() and len(w) > 2 and w not in _ENGLISH_WORDS}
_KN_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")


def detect_language(text: str) -> str:
    """'kn' for Kannada script, 'kn-latn' for romanised Kannada, else 'en'."""
    if _KANNADA_SCRIPT.search(text):
        return "kn"
    words = set(_tokenise(text))
    return "kn-latn" if words & _ROMAN_KANNADA else "en"


def _tokenise(text: str) -> list[str]:
    text = text.lower().translate(_KN_DIGITS)
    # keep "ಟೀ ಶರ್ಟ್" / "t shirt" as one garment
    text = re.sub(r"\b(t|tee)\s*-?\s*shirt", "t-shirt", text)
    text = re.sub(r"(ಟೀ|ಟಿ)\s*ಶರ್ಟ್", "ಟೀಶರ್ಟ್", text)
    return [t for t in re.split(r"[\s,.;!?।]+", text) if t]


_DEFAULT_VERB = {"G": "wearing", "O": "holding", "H": "with"}


@dataclass
class _Phrase:
    words: list[str]
    head_tag: str  # G garment, O object or H hair
    verb: str | None = None


def translate(text: str) -> str:
    """Turn a Kannada / Kanglish / mixed description into English. English passes through."""
    if detect_language(text) == "en":
        return text.strip()

    tokens = _tokenise(text)
    subject: list[str] = []
    subject_mods: list[str] = []
    phrases: list[_Phrase] = []
    pending: list[str] = []  # colours / modifiers waiting for their noun
    unknown: list[str] = []
    age: str | None = None

    for i, tok in enumerate(tokens):
        if tok.isdigit():
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if nxt in _YEARS:
                age = f"{tok} year old"
            continue
        if tok in _YEARS:
            continue
        tag, en = LEXICON.get(tok, ("?", tok))
        if tag == "X":
            continue
        if tag == "S":
            subject_mods.append(en)
        elif tag in ("C", "M"):
            pending.append(en)
        elif tag in ("G", "O", "H"):
            phrases.append(_Phrase(words=[*pending, en], head_tag=tag))
            pending = []
        elif tag == "P":
            subject = [*pending, en]
            pending = []
        elif tag == "V":
            # SOV: the verb applies to the phrase just before it
            if phrases and phrases[-1].verb is None:
                phrases[-1].verb = en
        else:
            unknown.append(tok)

    if pending:  # dangling colour/modifier, e.g. "huduga kempu" -> describe the subject
        subject = [*pending, *(subject or ["person"])]
    if not subject:
        subject = ["person"]
    subject = [*subject_mods, *subject]
    if age:
        subject = [age, *subject]

    parts = []
    for ph in phrases:
        verb = ph.verb or _DEFAULT_VERB[ph.head_tag]
        if ph.head_tag == "H":
            verb = "with"  # "wearing long hair" reads oddly; hair is always "with"
        noun = " ".join(ph.words)
        article = "" if ph.head_tag == "H" else _article(noun)
        parts.append(f"{verb} {article}{noun}")
    head = " ".join(subject)
    english = f"{_article(head)}{head}"
    if parts:
        english += " " + parts[0] + "".join(f", {p}" for p in parts[1:])
    if unknown:
        english += " " + " ".join(unknown)  # keep untranslated words: they may still help
    return english


_PLURAL = {"jeans", "trousers", "shorts", "glasses", "leggings", "pants"}


def _article(noun: str) -> str:
    if noun.split()[-1] in _PLURAL:
        return ""  # "wearing blue jeans"
    if noun.split()[0][0].isdigit():
        return "a "
    return "an " if noun[0] in "aeiou" else "a "
