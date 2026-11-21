import pytest

from sentineldesk import TARGETS
from sentineldesk.lang import detect_language
from sentineldesk.triage import HIGH_LABELS, Triager, apply_t_high

t = Triager()


def test_contract():
    r = t.predict("VPN down", "VPN fails for all hospital staff, EMR unreachable")
    assert set(r) == set(TARGETS)
    for k in TARGETS:
        assert set(r[k]) >= {"label", "confidence", "probs"}
        assert 0 <= r[k]["confidence"] <= 1 and abs(sum(r[k]["probs"].values()) - 1) < 1e-3
        assert r[k]["label"] in r[k]["probs"]
        assert isinstance(r[k]["confidence"], float)


def test_null_subject_and_lang():
    r = t.predict(None, "Passwort zurücksetzen bitte")
    assert set(r) == set(TARGETS)
    assert t.predict("", "x")["type"]["label"]
    assert detect_language("Ich kann mich nicht anmelden, bitte helfen") == "de"
    assert detect_language("I cannot log in, please help") == "en"


@pytest.mark.parametrize("text,lang", [
    ("Passwort vergessen", "de"),
    ("Rechnung fehlt", "de"),
    ("Hallo, das VPN funktioniert seit heute Morgen nicht mehr.", "de"),
    ("VPN down", "en"),
    ("Invoice copy please", "en"),
    ("Our clinic lost VPN access about 10 minutes ago and doctors cannot reach the EMR.", "en"),
    ("", "en"),
    ("!!! ???", "en"),
])
def test_language_cases(text, lang):
    assert detect_language(text) == lang


def test_t_high_rule():
    probs = {"high": 0.25, "critical": 0.05, "medium": 0.45, "low": 0.25}
    assert apply_t_high(probs, 0.5) == "medium"
    assert apply_t_high(probs, 0.3) == "high"
    assert apply_t_high(probs, None) == "medium"
    assert apply_t_high({"critical": 0.4, "high": 0.1, "low": 0.5}, 0.3) == "critical"


def test_high_priority_label_is_known():
    r = t.predict("URGENT: production down", "All servers are down, customers cannot access anything.")
    if t.trained:
        assert set(r["priority"]["probs"]) & HIGH_LABELS
