"""Language detection restricted to English / German (Dev A).

langdetect (seeded, so deterministic) decides for normal-length text. Short or ambiguous
text, or text langdetect assigns to a third language (nl, da, af, ...), falls back to a
lexical score: German function words, umlauts and ß versus English function words.
Default "en".
"""
from __future__ import annotations

import re

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0

MIN_WORDS_FOR_LANGDETECT = 5
_WORD = re.compile(r"[a-zäöüß]+")
_DE = frozenset("""
bitte nicht ich und der die das ist sind wir sie mein meine meinen kann können keine kein danke hallo
funktioniert mit für auf seit noch auch aber oder wenn dass haben wird wurde nach beim zum zur ein eine
einen dem den des vom sehr geehrte geehrtes geehrter grüßen freundlichen leider problem passwort rechnung
zugang anmelden anmeldung fehler zurücksetzen vergessen helfen hilfe uns ihr ihre ihnen es gibt heute
""".split())
_EN = frozenset("""
the and please not is are we you can cannot can't with for since hello hi thanks thank my our your
have has was were will would could should this that there from into after login password invoice error
help dear regards issue problem it to of in on at be an any all
""".split())


def _lexical(text: str) -> str | None:
    t = text.lower()
    words = _WORD.findall(t)
    de = sum(w in _DE for w in words) + 1.5 * len(re.findall(r"[äöüß]", t))
    en = sum(w in _EN for w in words)
    if de > en:
        return "de"
    if en > de:
        return "en"
    return None


def detect_language(text) -> str:
    """Return "en" or "de" for the given ticket text."""
    text = (text or "").strip()
    if not text:
        return "en"
    if len(_WORD.findall(text.lower())) < MIN_WORDS_FOR_LANGDETECT:
        return _lexical(text) or "en"
    try:
        probs = {l.lang: l.prob for l in detect_langs(text)}
    except Exception:  # LangDetectException on symbol-only text
        return _lexical(text) or "en"
    de, en = probs.get("de", 0.0), probs.get("en", 0.0)
    if max(de, en) < 0.5 or abs(de - en) < 0.3:  # third language or mixed text
        return _lexical(text) or ("de" if de > en else "en")
    return "de" if de > en else "en"
