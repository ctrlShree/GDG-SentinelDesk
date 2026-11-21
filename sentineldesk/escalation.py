"""Escalation gate. Rules (PLAN.md):
1 priority=high & type=Incident (hard stop)  2 any confidence < CONF_MIN  3 top similarity < TAU_SIM (hard stop)
4 unsupported/unsafe asks not covered by hits  5 prompt injection  6 draft not grounded
"""
import json
import re

from sentineldesk import MODELS_DIR, TARGETS

CONF_MIN = 0.45
TAU_SIM = 0.75  # overridden by models/gate.json once tuned


def _tau() -> float:
    p = MODELS_DIR / "gate.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8")).get("tau_sim", TAU_SIM))
    return TAU_SIM


_UNSUPPORTED = re.compile(
    r"\b(refund|compensation|reimburse\w*|sla credit|chargeback|legal action|lawsuit|"
    r"rückerstattung|entschädigung|erstattung|anwalt)\b", re.I)
_INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|"
    r"you are now|system prompt|reveal your (prompt|instructions)|ignoriere (alle )?(vorherigen )?anweisungen)",
    re.I)


def pre_check(triage: dict, hits: list[dict], text: str) -> dict:
    reasons, hard_stop = [], False
    if triage["priority"]["label"] == "high" and triage["type"]["label"] == "Incident":
        reasons.append("High-priority incident: route to on-call engineer")
        hard_stop = True
    low = [f"{t}={triage[t]['confidence']:.2f}" for t in TARGETS if triage[t]["confidence"] < CONF_MIN]
    if low:
        reasons.append(f"Low classifier confidence ({', '.join(low)} < {CONF_MIN})")
    top = hits[0]["score"] if hits else 0.0
    tau = _tau()
    if top < tau:
        reasons.append(f"No close precedent (top similarity {top:.2f} < {tau:.2f})")
        hard_stop = True
    m = _UNSUPPORTED.search(text or "")
    if m and not any(m.group(0).lower() in (h.get("answer") or "").lower() for h in hits):
        reasons.append(f"Unsupported request ('{m.group(0)}') not covered by precedents")
    if _INJECTION.search(text or ""):
        reasons.append("Possible prompt injection detected")
        hard_stop = True
    return {"reasons": reasons, "hard_stop": hard_stop}


def decide(triage: dict, hits: list[dict], text: str, draft: dict | None = None) -> dict:
    reasons = list(pre_check(triage, hits, text)["reasons"])
    if draft is None:
        reasons.append("No draft generated")
    elif not draft.get("grounded", False):
        reasons.append("Draft not grounded in precedents: " + "; ".join(draft.get("issues") or ["validator failed"]))
    escalate = bool(reasons)
    return {"escalate": escalate, "status": "ESCALATE" if escalate else "AUTO_DRAFT", "reasons": reasons}
