"""End-to-end SentinelDesk pipeline (Dev C).

Order: mask PII -> detect language -> triage -> retrieve top 5 -> pre_check gate
-> (skip the LLM on a hard stop) -> grounded draft -> final gate -> burst update.
"""
from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from sentineldesk.burst import BurstDetector
from sentineldesk.data import ticket_text
from sentineldesk.escalation import decide, pre_check
from sentineldesk.generator import draft_reply
from sentineldesk.lang import detect_language
from sentineldesk.pii import mask_pii
from sentineldesk.retrieval import Retriever
from sentineldesk.triage import Triager

log = logging.getLogger(__name__)

MAX_BODY_CHARS = 4000
TOP_K = 5
STAGES = ("pii", "triage", "retrieval", "generation", "gate")


class Timer:
    """Times named stages; repeated stage names accumulate."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.ms = {s: 0.0 for s in STAGES}

    def __call__(self, stage, fn):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.ms[stage] = self.ms.get(stage, 0.0) + (time.perf_counter() - start) * 1000

    def as_dict_with_total(self) -> dict:
        out = {k: round(v, 1) for k, v in self.ms.items()}
        out["total"] = round((time.perf_counter() - self.t0) * 1000, 1)
        return out


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays so FastAPI can serialise the result."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def new_ticket_id() -> str:
    return f"LIVE-{uuid.uuid4().hex[:8].upper()}"


class Pipeline:
    def __init__(self):
        self.triager = Triager()
        self.retriever = Retriever()
        self.burst = BurstDetector()

    def reset_burst(self) -> None:
        self.burst = BurstDetector()

    def run(self, subject, body, ts=None) -> dict:
        if not (body or "").strip():
            raise ValueError("Ticket body is empty")
        T = Timer()
        tid = new_ticket_id()
        truncated = len(body) > MAX_BODY_CHARS

        (ms, k1), (mb, k2) = T("pii", lambda: (mask_pii(subject or ""), mask_pii(body[:MAX_BODY_CHARS])))
        text = ticket_text(ms, mb)
        lang = detect_language(text)
        triage = T("triage", lambda: self.triager.predict(ms, mb))
        hits = T("retrieval", lambda: self.retriever.search(text, k=TOP_K))
        pre = T("gate", lambda: pre_check(triage, hits, text))

        draft, gen_error = None, None
        if not pre["hard_stop"]:
            try:
                draft = T("generation", lambda: draft_reply(ms, mb, lang, hits))
            except Exception as e:  # never let the LLM path take the console down
                log.exception("draft_reply failed")
                gen_error = f"Draft generation failed ({type(e).__name__})"

        esc = T("gate", lambda: decide(triage, hits, text, draft))
        if gen_error:
            esc = {"escalate": True, "status": "ESCALATE", "reasons": [*esc["reasons"], gen_error]}

        try:
            emb = self.retriever.embed([text], is_query=True)[0]
            burst = self.burst.add(tid, emb, triage, ts=ts, text=text)
        except Exception:
            log.exception("burst update failed")
            burst = None

        result = {
            "ticket_id": tid,
            "language": lang,
            "pii_masked": bool(k1 or k2),
            "masked_text": text,
            "triage": triage,
            "hits": hits,
            "draft": draft,
            "draft_for_human_review": bool(esc["escalate"] and draft is not None),
            "escalation": esc,
            "burst": burst,
            "latency_ms": T.as_dict_with_total(),
        }
        if truncated:
            result["truncated"] = True  # extra key; frontend shows a note
        return to_jsonable(result)
