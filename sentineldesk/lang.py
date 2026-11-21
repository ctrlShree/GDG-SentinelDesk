"""Language detection (stub: German keyword heuristic)."""
import re

_DE = re.compile(r"\b(der|die|das|und|nicht|ich|ist|bitte|mein|kann|wir|zurücksetzen|passwort|hallo)\b|[äöüß]", re.I)


def detect_language(text: str) -> str:
    return "de" if len(_DE.findall(text or "")) >= 2 else "en"
