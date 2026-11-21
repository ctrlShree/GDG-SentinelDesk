"""Triage classifier (Dev A): type, queue, priority, each with a confidence.

Loads models/triage.joblib ({target: sklearn Pipeline}) and models/triage_config.json
(written by scripts/train_triage.py). The model is loaded once per process and shared by
every Triager(). If it hasn't been trained yet, a keyword heuristic keeps the contract so
the API and tests still run.

Priority uses a tuned threshold `t_high`: if P(high) + P(critical) >= t_high the label is the
more likely of {high, critical}, otherwise the argmax of the remaining classes. This trades
some priority precision for high-priority recall (missing an outage is the costly error).

Calibration: the balanced logistic regressions are underconfident (val queue: mean confidence
0.49 vs accuracy 0.63). scripts/calibrate_triage.py fits one temperature T per target on the
out-of-sample validation probabilities; probabilities are rescaled as softmax(log p / T).
This never changes the argmax, only the confidences (and t_high is re-tuned after it).
"""
from __future__ import annotations

import functools
import json
import logging

import numpy as np

from sentineldesk import MODELS_DIR, TARGETS
from sentineldesk.data import ticket_text

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "triage.joblib"
CONFIG_PATH = MODELS_DIR / "triage_config.json"
HIGH_LABELS = frozenset({"high", "critical"})


def temper(P: np.ndarray, T: float | None) -> np.ndarray:
    """Temperature-scale probability rows: softmax(log p / T). T < 1 sharpens, T > 1 flattens."""
    if not T or T == 1.0:
        return P
    logits = np.log(np.clip(P, 1e-12, 1.0)) / T
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def choose_t_high(p_high: np.ndarray, y_is_high: np.ndarray, target_recall: float = 0.80) -> float:
    """Largest threshold on P(high-priority) whose recall is >= target_recall (0.01 grid)."""
    best = 0.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        if y_is_high.sum() and (p_high >= t)[y_is_high].mean() >= target_recall:
            best = float(t)
    return best


def apply_t_high(probs: dict, t_high: float | None) -> str:
    """Priority decision rule shared by training (threshold tuning) and inference."""
    hi = {k: v for k, v in probs.items() if k in HIGH_LABELS}
    if t_high is None or not hi:
        return max(probs, key=probs.get)
    if sum(hi.values()) >= t_high:
        return max(hi, key=hi.get)
    rest = {k: v for k, v in probs.items() if k not in HIGH_LABELS}
    return max(rest, key=rest.get) if rest else max(probs, key=probs.get)


def _entry(label: str, probs: dict) -> dict:
    probs = {str(k): float(v) for k, v in probs.items()}
    if label in HIGH_LABELS and label != max(probs, key=probs.get):
        conf = sum(v for k, v in probs.items() if k in HIGH_LABELS)  # chosen via t_high
    else:
        conf = probs[label]
    return {"label": str(label), "confidence": round(float(min(max(conf, 0.0), 1.0)), 4),
            "probs": {k: round(v, 4) for k, v in probs.items()}}


@functools.lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists():
        log.warning("No trained triage model at %s; using keyword heuristic. Run scripts/train_triage.py",
                    MODEL_PATH)
        return None, {}
    import joblib

    models = joblib.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    return models, config


class Triager:
    def __init__(self):
        self.models, self.config = _load()
        self.t_high = self.config.get("t_high")
        self.temperature = self.config.get("temperature", {})

    @property
    def trained(self) -> bool:
        return self.models is not None

    def predict(self, subject, body) -> dict:
        return self.predict_many([subject], [body])[0]

    def predict_many(self, subjects, bodies) -> list[dict]:
        texts = [ticket_text(s, b) for s, b in zip(subjects, bodies)]
        if self.models is None:
            return [_heuristic(x) for x in texts]
        per_target = {t: (self.models[t].classes_,
                          temper(self.models[t].predict_proba(texts), self.temperature.get(t)))
                      for t in TARGETS}
        out = []
        for i in range(len(texts)):
            r = {}
            for t, (classes, P) in per_target.items():
                probs = dict(zip(classes, P[i]))
                label = apply_t_high(probs, self.t_high) if t == "priority" else max(probs, key=probs.get)
                r[t] = _entry(label, probs)
            out.append(r)
        return out


# ------------------------------------------------------------------ untrained fallback
_LABELS = {
    "type": ["Incident", "Request", "Problem", "Change"],
    "queue": ["Technical Support", "IT Support", "Product Support", "Customer Service",
              "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
              "Sales and Pre-Sales", "Human Resources", "General Inquiry"],
    "priority": ["high", "medium", "low"],
}


def _heuristic(text: str) -> dict:
    t = text.lower()
    outage = any(w in t for w in ("outage", "down", "ausgefallen", "unreachable", "nicht erreichbar"))
    picks = {
        "type": ("Incident", 0.86) if outage else ("Request", 0.78),
        "queue": ("Service Outages and Maintenance", 0.74) if outage else ("Technical Support", 0.66),
        "priority": ("high", 0.81) if outage else ("low", 0.69),
    }
    out = {}
    for tgt in TARGETS:
        label, conf = picks[tgt]
        labels = _LABELS[tgt]
        rest = (1.0 - conf) / (len(labels) - 1)
        out[tgt] = {"label": label, "confidence": conf,
                    "probs": {l: round(conf if l == label else rest, 4) for l in labels}}
    return out
