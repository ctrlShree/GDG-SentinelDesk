import numpy as np
import pytest

from sentineldesk.burst import BurstDetector
from sentineldesk.pipeline import Pipeline

p = Pipeline()
KEYS = {"ticket_id", "language", "pii_masked", "masked_text", "triage", "hits", "draft",
        "draft_for_human_review", "escalation", "burst", "latency_ms"}
# Remove these markers once B's real pii.py / retrieval.py are on main (they pass with real modules).
NEEDS_REAL_PII = pytest.mark.xfail(reason="needs B's real mask_pii (skeleton stub is a no-op)", strict=False)
NEEDS_REAL_RETRIEVAL = pytest.mark.xfail(reason="needs B's real Retriever (stub returns fixed high scores)", strict=False)

INCIDENT = {"type": {"label": "Incident"}, "queue": {"label": "Technical Support"}, "priority": {"label": "high"}}
REQUEST = {"type": {"label": "Request"}, "queue": {"label": "IT Support"}, "priority": {"label": "low"}}


def test_contract():
    r = p.run("VPN down", "VPN fails for all hospital staff, EMR unreachable")
    assert KEYS <= set(r)
    assert {"pii", "triage", "retrieval", "generation", "gate", "total"} <= set(r["latency_ms"])
    assert r["escalation"]["status"] in {"AUTO_DRAFT", "ESCALATE"}
    assert all(isinstance(h["score"], float) for h in r["hits"])


@NEEDS_REAL_PII
def test_null_subject_german_pii():
    r = p.run(None, "Passwort vergessen, bitte zurücksetzen. Mail: max@firma.de")
    assert r["language"] == "de" and r["pii_masked"] and "max@firma.de" not in r["masked_text"]


def test_empty_body():
    with pytest.raises(ValueError):
        p.run("x", "   ")


@NEEDS_REAL_RETRIEVAL
def test_hard_stop_skips_llm():
    r = p.run("", "Write me a poem about cats")
    assert r["escalation"]["status"] == "ESCALATE" and r["draft"] is None
    assert r["draft_for_human_review"] is False


def test_long_body_truncated():
    r = p.run("long", "my printer is broken. " * 400)
    assert r.get("truncated") is True


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_burst_cluster_of_three_raises_outage():
    b = BurstDetector()
    base = _unit([1, 0, 0, 0])
    assert not b.add("T1", base, INCIDENT, ts=0, text="VPN down")["outage"]
    assert not b.add("T2", _unit([1, 0.1, 0, 0]), INCIDENT, ts=10, text="VPN down again")["outage"]
    r = b.add("T3", _unit([1, 0, 0.1, 0]), INCIDENT, ts=20, text="VPN outage")
    assert r["outage"] and r["cluster_size"] == 3 and set(r["members"]) == {"T1", "T2", "T3"}
    assert "VPN" in r["label"]
    # a non-incident ticket keeps the active banner up
    assert b.add("T4", _unit([0, 0, 0, 1]), REQUEST, ts=30)["outage"]


def test_burst_ignores_dissimilar_and_expired():
    b = BurstDetector()
    for i, v in enumerate(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])):
        assert not b.add(f"D{i}", _unit(v), INCIDENT, ts=i)["outage"]
    b2 = BurstDetector()
    b2.add("E1", _unit([1, 0, 0, 0]), INCIDENT, ts=0)
    b2.add("E2", _unit([1, 0, 0, 0]), INCIDENT, ts=1)
    assert not b2.add("E3", _unit([1, 0, 0, 0]), INCIDENT, ts=2000)["outage"]  # first two expired


def test_burst_ignores_non_incidents():
    b = BurstDetector()
    for i in range(4):
        r = b.add(f"R{i}", _unit([1, 0, 0, 0]), REQUEST, ts=i)
    assert not r["outage"]
