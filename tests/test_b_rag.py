import pandas as pd

from sentineldesk.data import load_test_inputs
from sentineldesk.generator import draft_reply
from sentineldesk.retrieval import Retriever
from sentineldesk.validator import check_draft

r = Retriever()

H = [{"ticket_id": "TKT-00001", "score": 0.9, "answer": "Reset via the portal link.", "subject": "", "body": "",
      "language": "en", "type": "Request", "queue": "IT Support", "priority": "low"}]


# ---- CP1: retrieval ----
def test_search_contract():
    hits = r.search("VPN connection drops for all users", k=5)
    assert len(hits) == 5
    assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
    assert all(h["ticket_id"].startswith("TKT-") and h["answer"] for h in hits)


def test_no_test_in_index():
    meta_ids = set(pd.read_parquet("models/kb_meta.parquet").ticket_id)
    assert not meta_ids & set(load_test_inputs().ticket_id)


def test_cross_lingual():
    # e5 maps EN and DE translations close together (C's burst detector clusters at >= 0.85) ...
    e = r.embed(["VPN outage: no one can connect to the VPN since this morning",
                 "VPN-Ausfall: Seit heute Morgen kann sich niemand mit dem VPN verbinden"])
    assert float(e[0] @ e[1]) >= 0.85
    # ... and both languages find on-topic precedents. (Shared top-10 IDs are NOT expected:
    # the KB has same-language near-duplicates that always outrank the other language.)
    for q in ("VPN is not working", "VPN funktioniert nicht"):
        assert any("vpn" in (h["subject"] + h["body"]).lower() for h in r.search(q, k=5))


def test_empty_query():
    assert r.search("") == []


# ---- CP2: validator + generator ----
def test_validator():
    bad = check_draft("We guarantee a 50% refund [TKT-00001] [TKT-99999]", ["TKT-00001", "TKT-99999"], H, "help")
    assert not bad["grounded"] and "TKT-99999" not in bad["cited_ids"]
    good = check_draft("Please reset via the portal link [TKT-00001].", ["TKT-00001"], H, "reset")
    assert good["grounded"]


def test_draft_contract():
    hits = r.search("How do I reset my password?")
    d = draft_reply("", "How do I reset my password?", "en", hits)
    assert set(d) >= {"reply", "cited_ids", "grounded", "issues", "fallback"}
    assert set(d["cited_ids"]) <= {h["ticket_id"] for h in hits}


def test_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    d = draft_reply("", "Passwort vergessen", "de", r.search("Passwort vergessen"))
    assert d["reply"] and d["cited_ids"] and d["fallback"]


def test_no_hits():
    d = draft_reply("", "anything", "en", [])
    assert not d["grounded"] and d["cited_ids"] == []
