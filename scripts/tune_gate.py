"""Tune the escalation gate's thresholds on the KB only (Dev A). Never reads the test split.

    python scripts/tune_gate.py [--val-preds models/full/val_preds.parquet] [--write]

TAU_CONF (per target): from out-of-sample validation probabilities saved by
train_triage.py. Confidences are bucketed into 0.05-wide bands; the threshold is the upper
edge of the highest band whose accuracy is below MAX_BAND_ACC[target]. type/priority use 0.5
("below this the classifier is wrong more often than right"; priority drives paging). queue
uses 0.25: with ~50 classes a 0.4-accurate guess is still far above chance and a wrong queue
only misroutes, so flagging it at 0.5 would escalate a third of all tickets.

TAU_SIM: needs B's real retriever. Rule 3 is a hard stop (no draft at all), so it must not
fire on genuine support tickets. The synthetic KB tickets all score 0.90-0.95 against each other
(even with near-duplicates removed), while realistic hand-written tickets score lower
(0.86-0.92), so KB tickets alone would set tau too high. tau = (lowest top-1 score among genuine
tickets: hand-written edge cases expected to AUTO_DRAFT + 1st percentile of KB validation) - SIM_MARGIN.
The off-domain catch rate is reported. Skipped while retrieval is a stub.

--write saves models/gate.json, which sentineldesk/escalation.py reads at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk import MODELS_DIR, TARGETS  # noqa: E402
from sentineldesk.triage import _entry, apply_t_high, temper  # noqa: E402

MAX_BAND_ACC = {"type": 0.50, "queue": 0.25, "priority": 0.50}
BAND = 0.05
MIN_BAND_N = 15
SIM_MARGIN = 0.005
# Never let a rule silently switch off: if no band is bad enough, fall back to this floor.
FLOOR = {"type": 0.45, "queue": 0.15, "priority": 0.45}
OFF_DOMAIN = [
    "Write me a poem about cats", "asdfgh qwertz lorem ipsum 12345", "What is the capital of France?",
    "Tell me a joke about penguins", "Schreib mir ein Gedicht über den Frühling", "Who won the football match yesterday?",
    "Give me a recipe for banana bread", "Wie wird das Wetter morgen in Berlin?", "Translate 'good morning' to Japanese",
    "Recommend a good science fiction novel", "zzzz ???? !!!! ####", "What's the meaning of life?",
    "Erzähl mir einen Witz", "How many legs does a spider have?", "Plan a 3-day trip to Rome for me",
]


def tune_conf(vp: pd.DataFrame, t_high, temperature: dict | None = None) -> tuple[dict, dict]:
    thresholds, report = {}, {}
    for t in TARGETS:
        m = vp[t].notna() & (vp[t].astype(str) != "")
        rows = vp[m]
        conf, correct = [], []
        for probs_json, y in zip(rows[f"probs_{t}"], rows[t].astype(str)):
            probs = json.loads(probs_json)
            T = (temperature or {}).get(t)
            if T:
                keys = list(probs)
                probs = dict(zip(keys, temper(np.array([probs[k] for k in keys]), T)))
            label = apply_t_high(probs, t_high) if t == "priority" else max(probs, key=probs.get)
            conf.append(_entry(label, probs)["confidence"])
            correct.append(label == y)
        conf, correct = np.array(conf), np.array(correct)
        best, bands = 0.0, []
        for lo in np.round(np.arange(0.0, 1.0, BAND), 2):
            b = (conf >= lo) & (conf < lo + BAND)
            if b.sum() >= MIN_BAND_N:
                acc = float(correct[b].mean())
                bands.append([float(lo), round(acc, 3), round(float(b.mean()), 4)])
                if acc < MAX_BAND_ACC[t]:
                    best = round(float(lo + BAND), 2)
        best = max(best, FLOOR[t])
        below = conf < best
        thresholds[t] = best
        report[t] = {"threshold": best, "max_band_acc": MAX_BAND_ACC[t], "bands_lo_acc_share": bands,
                     "flag_rate": round(float(below.mean()), 4),
                     "acc_below": round(float(correct[below].mean()), 4) if below.any() else None,
                     "acc_above": round(float(correct[~below].mean()), 4),
                     "conf_p10_p50_p90": [round(float(x), 3) for x in np.percentile(conf, [10, 50, 90])]}
    return thresholds, report


def tune_sim(vp: pd.DataFrame, n: int) -> dict | None:
    from sentineldesk import retrieval
    from sentineldesk.data import ticket_text

    if "stub" in (retrieval.__doc__ or "").lower():
        print("retrieval is still a stub: skipping TAU_SIM")
        return None
    r = retrieval.Retriever()
    rows = vp.sample(min(n, len(vp)), random_state=42)
    ins = []
    for tid, s, b in zip(rows["ticket_id"], rows["subject"], rows["body"]):
        text = ticket_text(s, b)
        hits = [h for h in r.search(text, k=6) if h["ticket_id"] != tid and (h.get("body") or "") != b]
        ins.append(hits[0]["score"] if hits else 0.0)
    from sentineldesk import ROOT
    edge = [json.loads(x) for x in (ROOT / "tests" / "edge_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    genuine = [e for e in edge if e.get("expected_status") == "AUTO_DRAFT"]  # tickets rule 3 must never block
    gen_scores = {e["id"]: r.search(ticket_text(e.get("subject"), e["body"]), k=1)[0]["score"] for e in genuine}
    outs = np.array([(r.search(q, k=1) or [{"score": 0.0}])[0]["score"] for q in OFF_DOMAIN])
    ins = np.array(ins)
    floor_kb = float(np.percentile(ins, 1))
    lowest_id = min(gen_scores, key=gen_scores.get)
    tau = round(min(floor_kb, gen_scores[lowest_id]) - SIM_MARGIN, 3)
    return {"tau_sim": tau,
            "kb_val_top1_p1_p50": [round(floor_kb, 3), round(float(np.median(ins)), 3)],
            "handwritten_genuine_min": [lowest_id, round(gen_scores[lowest_id], 3)],
            "off_domain_top1_max_p50": [round(float(outs.max()), 3), round(float(np.median(outs)), 3)],
            "kb_val_below_tau": round(float((ins < tau).mean()), 4),
            "off_domain_caught": round(float((outs < tau).mean()), 4),
            "off_domain_missed": [q for q, o in zip(OFF_DOMAIN, outs) if o >= tau]}


def escalation_rate(vp: pd.DataFrame, n: int) -> dict:
    import os
    os.environ["GEMINI_API_KEY"] = ""
    from collections import Counter

    from sentineldesk.pipeline import Pipeline

    pipe = Pipeline()
    rows = vp.sample(min(n, len(vp)), random_state=7)
    res = [pipe.run(s, b) for s, b in zip(rows["subject"], rows["body"]) if (b or "").strip()]
    reasons = Counter(x.split(":")[0].split("(")[0].strip() for r in res for x in r["escalation"]["reasons"])
    return {"n": len(res), "escalation_rate": round(float(np.mean([r["escalation"]["escalate"] for r in res])), 4),
            "reasons": dict(reasons.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-preds", type=Path, default=MODELS_DIR / "val_preds.parquet")
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--write", action="store_true", help="save models/gate.json")
    args = ap.parse_args()

    cfg_path = args.val_preds.parent / "triage_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    t_high, temperature = cfg.get("t_high"), cfg.get("temperature")
    vp = pd.read_parquet(args.val_preds)
    print(f"validation rows: {len(vp)} (t_high={t_high}, temperature={temperature})")

    conf, conf_report = tune_conf(vp, t_high, temperature)
    print("TAU_CONF per target:", json.dumps(conf_report, indent=2))
    sim = tune_sim(vp, args.n_sim)
    if sim:
        print("TAU_SIM:", json.dumps(sim, indent=2))

    gate_path = MODELS_DIR / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    gate["conf_thresholds"] = conf
    gate["conf_report"] = conf_report
    if sim:
        gate["tau_sim"] = sim["tau_sim"]
        gate["sim_report"] = sim
    if args.write:
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(f"wrote {gate_path}")
        print("escalation rate on kb validation tickets (sanity range 0.20-0.45; note the final model saw "
              "these tickets in training, so confidences are optimistic):",
              json.dumps(escalation_rate(vp, 200), indent=2))


if __name__ == "__main__":
    main()
