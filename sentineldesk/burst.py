"""Live outage / burst detection (Dev C).

Keeps a sliding window of recent incident-like tickets and raises an outage when a
new ticket has >= MIN_NEIGHBOURS semantically similar tickets in that window
(i.e. a cluster of >= MIN_NEIGHBOURS + 1). Once raised, the outage stays active
while its members are still inside the window, so the banner doesn't flicker.
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter, deque

import numpy as np

WINDOW_S = 15 * 60
SIM_THRESHOLD = float(os.getenv("BURST_SIM_THRESHOLD", "0.85"))
MIN_NEIGHBOURS = 2
INCIDENT_TYPES = {"Incident", "Problem"}
HIGH_PRIORITIES = {"high", "critical"}  # the dataset also has "critical"

_STOP = set("""the and for with that this from have has are was were not our your you all any can
cannot will been into they them their there what when which while about after again also very
just more most some such than then over under only other its it's ist und die der das nicht ein eine
mit von für auf sind wir ich sie bitte kein keine seit noch auch aber oder wenn dass haben wird""".split())


def _incident_like(triage: dict) -> bool:
    return (triage.get("priority", {}).get("label") in HIGH_PRIORITIES
            or triage.get("type", {}).get("label") in INCIDENT_TYPES)


def _label(texts: list[str], triages: list[dict]) -> str:
    queue = Counter(t.get("queue", {}).get("label", "") for t in triages).most_common(1)[0][0]
    words = Counter()
    for text in texts:
        words.update({w for w in re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", text.lower()) if w not in _STOP})
    keywords = [w for w, n in words.most_common(3) if n >= 2]
    topic = " / ".join(w.upper() if len(w) <= 4 else w for w in keywords) or "similar failures"
    return f"{topic} ({queue})" if queue else topic


class BurstDetector:
    def __init__(self, window_s: int = WINDOW_S, sim_threshold: float = SIM_THRESHOLD,
                 min_neighbours: int = MIN_NEIGHBOURS):
        self.window_s = window_s
        self.sim_threshold = sim_threshold
        self.min_neighbours = min_neighbours
        self._items: deque = deque()  # (ts, ticket_id, emb, triage, text)
        self._active: dict | None = None

    def _evict(self, now: float) -> None:
        while self._items and now - self._items[0][0] > self.window_s:
            self._items.popleft()

    @staticmethod
    def _quiet() -> dict:
        return {"outage": False, "cluster_size": 0, "label": "", "members": []}

    def add(self, ticket_id, embedding, triage, ts=None, text="") -> dict:
        now = time.time() if ts is None else float(ts)
        self._evict(now)

        if _incident_like(triage):
            emb = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            emb = emb / norm if norm else emb
            neighbours = [it for it in self._items if float(it[2] @ emb) >= self.sim_threshold]
            self._items.append((now, ticket_id, emb, triage, text))
            if len(neighbours) >= self.min_neighbours:
                members = neighbours + [self._items[-1]]
                self._active = {
                    "outage": True,
                    "cluster_size": len(members),
                    "label": _label([m[4] for m in members], [m[3] for m in members]),
                    "members": [m[1] for m in members],
                }
                return dict(self._active)

        if self._active:
            live = {it[1] for it in self._items}
            still = [m for m in self._active["members"] if m in live]
            if len(still) > self.min_neighbours:
                self._active = {**self._active, "members": still, "cluster_size": len(still)}
                return dict(self._active)
            self._active = None
        return self._quiet()
