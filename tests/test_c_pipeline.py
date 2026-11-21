import numpy as np
import pytest

from sentineldesk.burst import BurstDetector
from sentineldesk.pipeline import Pipeline
from sentineldesk.retrieval import INDEX_PATH

KEYS = {"ticket_id", "language", "pii_masked", "masked_text", "triage", "hits", "draft",
        "draft_for_human_review", "escalation", "burst", "latency_ms"}
# pii.py is still a no-op stub. Drop this marker when the real one lands.
NEEDS_REAL_PII = pytest.mark.xfail(reason="mask_pii is still a stub", strict=False)


@pytest.fixture(scope="module")
def p():
    if not INDEX_PATH.exists():
        pytest.skip("no FAISS index, run scripts/build_index.py")
    return Pipeline()

INCIDENT = {"type": {"label": "Incident"}, "queue": {"label": "Technical Support"}, "priority": {"label": "high"}}
REQUEST = {"type": {"label": "Request"}, "queue": {"label": "IT Support"}, "priority": {"label": "low"}}


def test_contract(p):
    r = p.run("VPN down", "VPN fails for all hospital staff, EMR unreachable")
    assert KEYS <= set(r)
    assert {"pii", "triage", "retrieval", "generation", "gate", "total"} <= set(r["latency_ms"])
    assert r["escalation"]["status"] in {"AUTO_DRAFT", "ESCALATE"}
    assert all(isinstance(h["score"], float) for h in r["hits"])


@NEEDS_REAL_PII
def test_null_subject_german_pii(p):
    r = p.run(None, "Passwort vergessen, bitte zurücksetzen. Mail: max@firma.de")
    assert r["language"] == "de" and r["pii_masked"] and "max@firma.de" not in r["masked_text"]


def test_empty_body(p):
    with pytest.raises(ValueError):
        p.run("x", "   ")


def test_hard_stop_skips_llm(p):
    r = p.run("", "Write me a poem about cats")
    assert r["escalation"]["status"] == "ESCALATE" and r["draft"] is None
    assert r["draft_for_human_review"] is False


def test_long_body_truncated(p):
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


def test_burst_counts_critical_priority():
    b = BurstDetector()
    crit = {"type": {"label": "Request"}, "queue": {"label": "IT Support"}, "priority": {"label": "critical"}}
    for i in range(3):
        r = b.add(f"C{i}", _unit([1, 0, 0, 0]), crit, ts=i, text="login outage")
    assert r["outage"] and r["cluster_size"] == 3


def test_burst_centering_removes_shared_direction():
    # all vectors share a big common part, like e5 does with tickets.
    # Raw cosine groups everything; after centering only the real cluster groups.
    common = np.array([5.0, 0, 0, 0, 0, 0])
    vpn = [common + np.array([0, 1, 0.1, 0, 0, 0]), common + np.array([0, 1, 0, 0.1, 0, 0]),
           common + np.array([0, 1, 0, 0, 0.1, 0])]
    other = [common + np.array([0, 0, 0, 0, 1, 0]), common + np.array([0, 0, 0, 0, 0, 1]),
             common + np.array([0, 0, 1, 0, 0, 0])]
    raw = BurstDetector(sim_threshold=0.55)
    assert [raw.add(f"O{i}", v, INCIDENT, ts=i)["outage"] for i, v in enumerate(other)][-1]  # raw cosine: false alarm
    b = BurstDetector(sim_threshold=0.55, center=common)
    assert not any(b.add(f"O{i}", v, INCIDENT, ts=i)["outage"] for i, v in enumerate(other))
    c = BurstDetector(sim_threshold=0.55, center=common)
    results = [c.add(f"V{i}", v, INCIDENT, ts=i) for i, v in enumerate(vpn)]
    assert results[-1]["outage"] and results[-1]["cluster_size"] == 3


def test_burst_members_include_one_hop():
    b = BurstDetector(sim_threshold=0.8)
    # cosines: A2-A 0.86, A2-C 0.77, A2-D 0.73, A-C 0.89, A-D 0.91, C-D 0.995
    a2, a = _unit([1, 0.6, 0]), _unit([1, 0, 0])
    c, d = _unit([1, 0, 0.5]), _unit([1, -0.1, 0.45])
    b.add("A2", a2, INCIDENT, ts=0)
    b.add("A", a, INCIDENT, ts=1)
    b.add("C", c, INCIDENT, ts=2)
    r = b.add("D", d, INCIDENT, ts=3)
    assert r["outage"] and set(r["members"]) == {"A2", "A", "C", "D"}  # A2 only via A
