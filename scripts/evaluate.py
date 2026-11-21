"""Holdout evaluation (Dev A). The ONLY caller of load_split() (enforced by tests/test_leakage.py).

    python scripts/evaluate.py --n 2000            # Gemini off by default (template fallback, fast)
    python scripts/evaluate.py --n 200 --use-llm   # include real Gemini latency / grounding

Only `subject` and `body` of test tickets are fed to the models; labels are used for scoring.

Writes:
    reports/metrics.json          frozen format (PLAN_OF_ACTION.md section 4) + extra keys
    reports/metrics.md            human-readable summary
    reports/confusion_<t>.png     confusion matrices
    reports/classification_report_<t>.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else 0.0


REASON_KEYS = [
    ("Critical", "Critical"),
    ("No sufficiently similar precedent", "No precedent"),
    ("Low classifier confidence", "Low confidence"),
    ("Unsupported request", "Unsupported request"),
    ("Possible prompt injection", "Prompt injection"),
    ("Draft not grounded", "Draft not grounded"),
    ("Draft generation failed", "Generation failed"),
]


def reason_key(reason: str) -> str:
    for prefix, key in REASON_KEYS:
        if reason.startswith(prefix):
            return key
    return re.split(r"[:(]", reason)[0].strip() or "Other"


def classifier_metrics(test, kb, triager, reports_dir: Path) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    from sentineldesk import TARGETS
    from sentineldesk.triage import HIGH_LABELS

    t0 = time.time()
    preds = []
    for i in range(0, len(test), 2000):
        chunk = test.iloc[i:i + 2000]
        preds += triager.predict_many(chunk["subject"].tolist(), chunk["body"].tolist())
    print(f"classified {len(test)} test tickets in {time.time() - t0:.0f}s", flush=True)
    for t in TARGETS:
        test[f"pred_{t}"] = [p[t]["label"] for p in preds]

    # Test tickets whose exact body also appears in kb (the dataset has many duplicates).
    unseen = ~test["body"].isin(set(kb["body"]))
    out = {"macro_f1": {}, "accuracy": {}, "macro_f1_unseen_text": {}, "n_labelled": {}}
    for t in TARGETS:
        m = test[t].notna() & (test[t].astype(str) != "")
        y, p = test.loc[m, t].astype(str), test.loc[m, f"pred_{t}"]
        out["macro_f1"][t] = round(f1_score(y, p, average="macro"), 4)
        out["accuracy"][t] = round(accuracy_score(y, p), 4)
        out["n_labelled"][t] = int(m.sum())
        mu = m & unseen
        out["macro_f1_unseen_text"][t] = round(
            f1_score(test.loc[mu, t].astype(str), test.loc[mu, f"pred_{t}"], average="macro"), 4)
        (reports_dir / f"classification_report_{t}.txt").write_text(
            classification_report(y, p, digits=3, zero_division=0), encoding="utf-8")
        labels = sorted(set(y) | set(p))
        cm = confusion_matrix(y, p, labels=labels)
        _plot_cm(cm, labels, f"{t} (n={m.sum()})", reports_dir / f"confusion_{t}.png")
    out["n_unseen_text"] = int(unseen.sum())

    # Core subset: rows from the main source (have a type), standard 10 queues, 3 priorities.
    core = test["type"].notna()
    out["macro_f1_core"] = {}
    for t in TARGETS:
        y = test.loc[core, t].astype(str)
        out["macro_f1_core"][t] = round(f1_score(y, test.loc[core, f"pred_{t}"], labels=sorted(set(y)),
                                                 average="macro", zero_division=0), 4)

    y_hi = test["priority"].isin(HIGH_LABELS)
    p_hi = test["pred_priority"].isin(HIGH_LABELS)
    out["high_priority_recall"] = round(float(p_hi[y_hi].mean()), 4)
    out["high_priority_precision"] = round(float(y_hi[p_hi].mean()), 4) if p_hi.any() else 0.0
    return out


def _plot_cm(cm, labels, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(labels)
    size = max(4, min(22, 0.35 * n + 3))
    fig, ax = plt.subplots(figsize=(size, size))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7 if n > 12 else 9)
    ax.set_yticks(range(n), labels, fontsize=7 if n > 12 else 9)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix: {title} (row-normalised colour)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_pipeline(pipe, rows) -> list[dict]:
    out = []
    for i, (s, b) in enumerate(rows):
        try:
            out.append(pipe.run(s, b))
        except ValueError:  # empty body
            out.append(None)
        if (i + 1) % 250 == 0:
            print(f"  pipeline {i + 1}/{len(rows)}", flush=True)
    return out


def pipeline_metrics(results, langs_true) -> dict:
    ok = [(r, l) for r, l in zip(results, langs_true) if r is not None]
    res = [r for r, _ in ok]
    esc = [r["escalation"]["escalate"] for r in res]
    drafts = [r["draft"] for r in res if r["draft"] is not None]
    stages = ("pii", "triage", "retrieval", "generation", "gate")
    reasons = Counter(reason_key(x) for r in res for x in r["escalation"]["reasons"])
    total = [r["latency_ms"]["total"] for r in res]
    return {
        "n_pipeline": len(res),
        "escalation_rate": round(float(np.mean(esc)), 4) if res else 0.0,
        "grounded_rate": round(float(np.mean([bool(d.get("grounded")) for d in drafts])), 4) if drafts else 0.0,
        "draft_rate": round(len(drafts) / len(res), 4) if res else 0.0,
        "fallback_rate": round(float(np.mean([bool(d.get("fallback")) for d in drafts])), 4) if drafts else 0.0,
        "latency_ms": {
            "p50": _pct(total, 50), "p95": _pct(total, 95),
            "by_stage": {s: {"p50": _pct([r["latency_ms"][s] for r in res], 50),
                             "p95": _pct([r["latency_ms"][s] for r in res], 95)} for s in stages},
        },
        "gate_reason_counts": dict(reasons.most_common()),
        "language_accuracy": round(float(np.mean([r["language"] == l for r, l in ok])), 4) if ok else 0.0,
    }


def edge_case_results(pipe) -> dict:
    from sentineldesk import ROOT

    path = ROOT / "tests" / "edge_cases.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    passed, failures = 0, []
    for row in rows:
        r = pipe.run(row.get("subject") or "", row["body"])
        status, reasons = r["escalation"]["status"], r["escalation"]["reasons"]
        want_s, want_r = row.get("expected_status"), row.get("expected_reason_contains") or ""
        ok = (not want_s or status == want_s) and (not want_r or any(want_r.lower() in x.lower() for x in reasons))
        passed += ok
        if not ok:
            failures.append({"id": row["id"], "expected": [want_s, want_r], "got": [status, reasons]})
    return {"passed": passed, "total": len(rows), "failures": failures}


def write_markdown(m: dict, path: Path) -> None:
    from sentineldesk import TARGETS

    L = [f"# SentinelDesk holdout metrics\n",
         f"Test split: **{m['n_test']}** tickets (last 20% of the frozen shuffle, seed 42). "
         f"Only subject + body are fed to the models.\n"]
    if m.get("components", {}).get("stub_modules"):
        L.append(f"> ⚠️ Provisional: these modules were still stubs: {', '.join(m['components']['stub_modules'])}. "
                 f"Retrieval/generation numbers will change.\n")
    L += ["## Classifier (full test split)\n", "| Target | Macro-F1 | Accuracy | Macro-F1 core subset | "
          "Macro-F1 unseen text |", "|---|---|---|---|---|"]
    for t in TARGETS:
        L.append(f"| {t} | **{m['macro_f1'][t]:.3f}** | {m['accuracy'][t]:.3f} | {m['macro_f1_core'][t]:.3f} | "
                 f"{m['macro_f1_unseen_text'][t]:.3f} |")
    L += ["",
          f"*Core subset* = tickets from the main source (have a `type`; 10 queues; high/medium/low). "
          f"*Unseen text* = the {m['n_unseen_text']} test tickets whose body does not occur verbatim in the KB "
          f"(the dataset contains many duplicate tickets, which inflates the headline numbers).\n",
          "## Safety\n",
          f"- High-priority recall (classifier, high + critical): **{m['high_priority_recall']:.3f}** "
          f"(precision {m['high_priority_precision']:.3f}, `t_high` = {m.get('t_high')})",
          f"- **High-Priority Incident Recall** (true high/critical Incidents the gate escalated, "
          f"n={m['n_hp_incidents']}): **{m['high_priority_incident_recall']:.3f}**; paged as Critical: "
          f"{m['hp_incident_critical_rate']:.3f}",
          f"- Escalation rate on {m['n_pipeline']} sampled test tickets: **{m['escalation_rate']:.3f}**",
          f"- Drafts grounded: {m['grounded_rate']:.3f} (template fallback rate {m['fallback_rate']:.3f})",
          f"- Edge cases: {m['edge_cases']['passed']}/{m['edge_cases']['total']} pass\n",
          "### Gate reasons (sampled tickets)\n", "| Reason | Count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in m["gate_reason_counts"].items()]
    lat = m["latency_ms"]
    L += ["", "## Latency (ms)\n", f"End to end p50 **{lat['p50']}**, p95 **{lat['p95']}**\n",
          "| Stage | p50 | p95 |", "|---|---|---|"]
    L += [f"| {s} | {v['p50']} | {v['p95']} |" for s, v in lat["by_stage"].items()]
    L += ["", f"Language detection agreement with dataset label: {m['language_accuracy']:.3f} "
              f"(the dataset tags some English tickets as `de`, so true accuracy is higher).\n",
          "## Leakage safeguards\n",
          "- Split frozen: shuffle seed 42, first 80% KB, last 20% test; fingerprint printed by "
          "`python -m sentineldesk.data`.",
          "- Classifier trained and thresholds tuned on the KB only (`models/triage_config.json: trained_on=kb`).",
          "- Retrieval index built from the KB only; `tests/test_leakage.py` checks no test id is in it.",
          "- Only this script reads test labels; `tests/test_leakage.py` fails if any other file does."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="sampled test tickets for pipeline metrics")
    ap.add_argument("--n-hpi", type=int, default=0, help="cap on high-priority incidents run (0 = all)")
    ap.add_argument("--use-llm", action="store_true", help="keep GEMINI_API_KEY (slow); default off")
    args = ap.parse_args()
    if not args.use_llm:
        os.environ["GEMINI_API_KEY"] = ""

    from sentineldesk import REPORTS_DIR, TARGETS
    from sentineldesk import generator, retrieval, pii, validator
    from sentineldesk.data import load_split
    from sentineldesk.pipeline import Pipeline
    from sentineldesk.triage import HIGH_LABELS

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    kb, test = load_split()
    print(f"kb={len(kb)} test={len(test)}", flush=True)

    pipe = Pipeline()
    m = {"available": True, "n_test": int(len(test)), "t_high": pipe.triager.t_high,
         "classifier_trained": pipe.triager.trained}
    m.update(classifier_metrics(test, kb, pipe.triager, REPORTS_DIR))

    # High-Priority Incident Recall: true high/critical Incidents escalated by the full pipeline.
    hpi = test[test["priority"].isin(HIGH_LABELS) & (test["type"] == "Incident")]
    if args.n_hpi and len(hpi) > args.n_hpi:
        hpi = hpi.sample(args.n_hpi, random_state=SEED)
    print(f"running {len(hpi)} high-priority incidents through the pipeline", flush=True)
    t0 = time.time()
    hres = [r for r in run_pipeline(pipe, list(zip(hpi["subject"], hpi["body"]))) if r is not None]
    m["n_hp_incidents"] = len(hres)
    m["high_priority_incident_recall"] = round(float(np.mean([r["escalation"]["escalate"] for r in hres])), 4)
    m["hp_incident_critical_rate"] = round(float(np.mean(
        [any(x.startswith("Critical") for x in r["escalation"]["reasons"]) for r in hres])), 4)
    print(f"  done in {time.time() - t0:.0f}s: recall {m['high_priority_incident_recall']}", flush=True)

    # Pipeline metrics on a random sample of the test split.
    sample = test.sample(min(args.n, len(test)), random_state=SEED)
    print(f"running {len(sample)} sampled test tickets through the pipeline", flush=True)
    pipe.reset_burst()
    res = run_pipeline(pipe, list(zip(sample["subject"], sample["body"])))
    m.update(pipeline_metrics(res, sample["language"].tolist()))

    pipe.reset_burst()
    m["edge_cases"] = edge_case_results(pipe)
    pipe.reset_burst()

    stubs = [mod.__name__.split(".")[-1] for mod in (retrieval, generator, pii, validator)
             if "stub" in (mod.__doc__ or "").lower()]
    m["components"] = {"stub_modules": stubs, "llm": bool(args.use_llm)}
    m["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_markdown(m, REPORTS_DIR / "metrics.md")
    print(json.dumps({k: m[k] for k in ("macro_f1", "macro_f1_unseen_text", "high_priority_recall",
                                        "high_priority_incident_recall", "escalation_rate", "latency_ms",
                                        "gate_reason_counts", "edge_cases")}, indent=2))
    print(f"wrote {REPORTS_DIR / 'metrics.json'} and metrics.md")


if __name__ == "__main__":
    main()
