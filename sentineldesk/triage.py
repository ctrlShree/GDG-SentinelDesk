"""Triage classifier (stub: keyword rule until scripts/train_triage.py lands)."""
from sentineldesk import TARGETS
from sentineldesk.data import ticket_text

_LABELS = {
    "type": ["Incident", "Request", "Problem", "Change"],
    "queue": ["Technical Support", "IT Support", "Product Support", "Customer Service",
              "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
              "Sales and Pre-Sales", "Human Resources", "General Inquiry"],
    "priority": ["high", "medium", "low"],
}


def _dist(labels: list[str], top: str, conf: float) -> dict[str, float]:
    rest = (1.0 - conf) / (len(labels) - 1)
    return {l: round(conf if l == top else rest, 4) for l in labels}


class Triager:
    def predict(self, subject: str, body: str) -> dict:
        text = ticket_text(subject, body).lower()
        outage = "outage" in text or "down" in text
        picks = {
            "type": ("Incident", 0.86) if outage else ("Request", 0.78),
            "queue": ("Service Outages and Maintenance", 0.74) if outage else ("Technical Support", 0.66),
            "priority": ("high", 0.81) if outage else ("low", 0.69),
        }
        out = {}
        for t in TARGETS:
            label, conf = picks[t]
            out[t] = {"label": label, "confidence": conf, "probs": _dist(_LABELS[t], label, conf)}
        return out
