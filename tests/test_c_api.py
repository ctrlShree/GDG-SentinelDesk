from fastapi.testclient import TestClient

from api import app

c = TestClient(app)


def test_triage_endpoint():
    r = c.post("/api/triage", json={"body": "Reset my password please"})
    assert r.status_code == 200 and r.json()["escalation"]["status"] in {"AUTO_DRAFT", "ESCALATE"}


def test_null_subject_ok():
    assert c.post("/api/triage", json={"subject": None, "body": "Invoice copy please"}).status_code == 200


def test_empty_body_422():
    r = c.post("/api/triage", json={"subject": "x", "body": "  "})
    assert r.status_code == 422 and "detail" in r.json()


def test_aux_endpoints():
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and {"models_loaded", "llm_available"} <= set(h)
    ex = c.get("/api/examples").json()
    assert isinstance(ex, list) and ex and {"id", "label", "subject", "body"} <= set(ex[0])
    assert c.post("/api/burst/reset").json() == {"ok": True}
    sim = c.post("/api/burst/simulate").json()
    assert "burst" in sim and len(sim["results"]) == 4
    assert isinstance(c.get("/api/metrics").json(), dict)
