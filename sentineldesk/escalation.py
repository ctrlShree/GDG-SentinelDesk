"""Escalation & safety gate (Dev A). Every rule adds a readable reason; see docs/gate.md.

1 Critical (hard stop): priority high/critical & type Incident, or an outage signal in the
  text ("VPN down for all staff", "ausgefallen") on an Incident/Problem/high-priority ticket.
  The real classifier calls the hospital VPN preset Problem/medium, so rule 1 alone misses it.
2 Low classifier confidence: per-target thresholds. Queue has ~50 classes, so its max
  probability is ~0.2 on normal tickets and needs a much lower bar than type/priority.
3 No sufficiently similar precedent (hard stop): top retrieval score < TAU_SIM, or no hits.
4 Unsupported request: risky intent (refund, pricing, legal, destructive, credentials)
  that no retrieved precedent answer covers.
5 Possible prompt injection (hard stop: never send it to the LLM).
6 Draft not grounded in evidence (B's validator), only in decide().
"""
from __future__ import annotations

import json
import re

from sentineldesk import MODELS_DIR, TARGETS
from sentineldesk.triage import HIGH_LABELS

TAU_CONF = 0.45
# Tuned on out-of-sample kb validation probabilities by scripts/tune_gate.py (see docs/gate.md);
# models/gate.json overrides these when present.
CONF_THRESHOLDS = {"type": 0.50, "queue": 0.20, "priority": 0.45}
TAU_SIM = 0.80  # placeholder until B's score distribution; models/gate.json overrides
GATE_CONFIG = MODELS_DIR / "gate.json"

_OUTAGE = re.compile(
    r"\b(outage|is down|are down|went down|goes down|down for (all|everyone|everybody)|unreachable|"
    r"not reachable|nobody can|no one can|none of (our|the)|all (users|staff|employees)|"
    r"ausgefallen|ausfall|störung|nicht erreichbar|niemand kann|kein mitarbeiter|alle mitarbeiter)\b",
    re.I)

RISKY_INTENTS = {
    "refund": r"refund|money back|compensat\w*|chargeback|erstattung|rückerstattung|erstatten|"
              r"entschädigung|gutschrift",
    "pricing": r"price|pricing|quote|discount|preis\w*|angebot|rabatt",
    "legal": r"legal|lawsuit|contract|liability|guarantee\w*|sla credit|vertrag|haftung|garantie\w*",
    "destructive": r"delete all|wipe|alle daten löschen",
    "credentials": r"admin password|share (your |the )?credentials|zugangsdaten",
}
_INTENT_RE = {k: re.compile(rf"\b({v})", re.I) for k, v in RISKY_INTENTS.items()}

_INJECTION = re.compile(
    r"(ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)|ignore all instructions|"
    r"disregard (all |the )?(previous|prior|above|system)|system prompt|you are now|"
    r"ignoriere .*anweisungen|vergiss .*regeln)",
    re.I)


def _gate_config() -> dict:
    """models/gate.json, written by scripts/tune_gate.py --write (KB-only tuning)."""
    if GATE_CONFIG.exists():
        try:
            return json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _tau_sim() -> float:
    return float(_gate_config().get("tau_sim", TAU_SIM))


def _conf_thresholds() -> dict:
    return {**CONF_THRESHOLDS, **_gate_config().get("conf_thresholds", {})}


def risky_intents(text: str) -> list[str]:
    return [k for k, rx in _INTENT_RE.items() if rx.search(text or "")]


def intent_covered(intent: str, hits: list[dict]) -> bool:
    return any(_INTENT_RE[intent].search(h.get("answer") or "") for h in hits)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def _critical(triage: dict, text: str) -> str | None:
    prio, typ = triage["priority"]["label"], triage["type"]["label"]
    if prio in HIGH_LABELS and typ == "Incident":
        return f"Critical: {prio}-priority Incident, paging on-call engineer"
    m = _OUTAGE.search(text or "")
    if m and (typ in {"Incident", "Problem"} or prio in HIGH_LABELS):
        return f"Critical: outage signal ('{m.group(0)}') on a {prio}-priority {typ}, paging on-call engineer"
    return None


def pre_check(triage: dict, hits: list[dict], text: str) -> dict:
    reasons, hard = [], False
    if crit := _critical(triage, text):
        reasons.append(crit)
        hard = True
    top = hits[0]["score"] if hits else 0.0
    tau = _tau_sim()
    if top < tau:
        reasons.append(f"No sufficiently similar precedent (top score {top:.2f} < {tau:.2f})")
        hard = True
    conf_min = _conf_thresholds()
    low = [f"{t} {triage[t]['confidence']:.2f}" for t in TARGETS
           if triage[t]["confidence"] < conf_min[t]]
    if low:
        reasons.append("Low classifier confidence: " + ", ".join(low))
    uncovered = [i for i in risky_intents(text) if not intent_covered(i, hits)]
    if uncovered:
        reasons.append(f"Unsupported request ({', '.join(uncovered)}) not covered by historical resolutions")
    if looks_like_injection(text):
        reasons.append("Possible prompt injection: instructions to the assistant were ignored")
        hard = True
    return {"reasons": reasons, "hard_stop": hard}


def decide(triage: dict, hits: list[dict], text: str, draft: dict | None = None) -> dict:
    reasons = pre_check(triage, hits, text)["reasons"]
    if draft is not None and not draft.get("grounded", False):
        reasons.append("Draft not grounded in evidence: " + ("; ".join(draft.get("issues") or []) or "validator failed"))
    esc = bool(reasons)
    return {"escalate": esc, "status": "ESCALATE" if esc else "AUTO_DRAFT", "reasons": reasons}
