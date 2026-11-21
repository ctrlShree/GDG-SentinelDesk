"""Train the triage classifier (Dev A). Uses load_kb() ONLY; never touches the test split.

    python scripts/train_triage.py            # validate on 10% of kb, then retrain on all of kb
    python scripts/train_triage.py --fast     # smaller feature space (quick iteration)

Outputs:
    models/triage.joblib        {target: sklearn Pipeline}
    models/triage_config.json   t_high, label sets, validation metrics, provenance

Data notes (see docs/gate.md): ~21% of kb rows come from a second source with no `type`
and no `answer`, niche queues ("Pets & Animals/..."), and extra priorities `critical` and
`very_low`. `type` is trained on labelled rows only; queue/priority use every row.
"High priority" means priority in HIGH_LABELS = {high, critical}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk import MODELS_DIR, TARGETS  # noqa: E402
from sentineldesk.data import load_kb, ticket_text  # noqa: E402
from sentineldesk.triage import HIGH_LABELS, apply_t_high  # noqa: E402

SEED = 42
VAL_FRAC = 0.10
TARGET_HIGH_RECALL = 0.80


def make_features(fast: bool = False) -> FeatureUnion:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           max_features=100_000 if fast else 300_000, strip_accents=None, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                           max_features=100_000 if fast else 300_000, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def _fit_clf(X, y) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0).fit(X, y)


def fit_all(df, fast: bool, tag: str) -> dict:
    """Fit the TF-IDF features once on df, then the three classifiers in parallel processes.

    Returns {target: Pipeline(features, clf)}; the fitted feature union is shared."""
    t0 = time.time()
    feats = make_features(fast).fit(df["text"])
    X = feats.transform(df["text"]).tocsr()
    print(f"[{tag}] features {X.shape}, nnz={X.nnz:,} in {time.time() - t0:.0f}s", flush=True)
    masks = {t: (df[t].notna() & (df[t].astype(str) != "")).to_numpy() for t in TARGETS}
    t0 = time.time()
    clfs = Parallel(n_jobs=len(TARGETS), backend="loky")(
        delayed(_fit_clf)(X[masks[t]], df.loc[masks[t], t].astype(str).to_numpy()) for t in TARGETS)
    print(f"[{tag}] fitted {len(TARGETS)} classifiers in parallel in {time.time() - t0:.0f}s", flush=True)
    return {t: Pipeline([("features", feats), ("clf", c)]) for t, c in zip(TARGETS, clfs)}


def labelled(df, target):
    return df[df[target].notna() & (df[target].astype(str) != "")]


def choose_t_high(p_high: np.ndarray, y_is_high: np.ndarray, target_recall: float) -> float:
    """Largest threshold on P(high-priority) whose recall on validation is >= target_recall."""
    best = 0.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = p_high >= t
        if y_is_high.sum() and pred[y_is_high].mean() >= target_recall:
            best = float(t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller feature space for quick iteration")
    ap.add_argument("--skip-final", action="store_true", help="validate only, don't retrain/save")
    args = ap.parse_args()

    kb = load_kb()
    kb["text"] = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    tr, va = train_test_split(kb, test_size=VAL_FRAC, random_state=SEED)
    print(f"kb={len(kb)}  train={len(tr)}  val={len(va)}")

    val_models = fit_all(tr, args.fast, "val")
    val_metrics = {}
    for t in TARGETS:
        b = labelled(va, t)
        m = val_models[t]
        pred = m.predict(b["text"])
        val_metrics[t] = {"macro_f1": round(f1_score(b[t], pred, average="macro"), 4),
                          "accuracy": round(accuracy_score(b[t], pred), 4),
                          "n_train": int(labelled(tr, t).shape[0]), "n_val": len(b),
                          "n_classes": len(m.classes_)}
        print(f"[val] {t:9s} {val_metrics[t]}", flush=True)

    # High-priority threshold on validation
    pm = val_models["priority"]
    b = labelled(va, "priority")
    P = pm.predict_proba(b["text"])
    hi_idx = [i for i, c in enumerate(pm.classes_) if c in HIGH_LABELS]
    p_high = P[:, hi_idx].sum(axis=1)
    y_high = b["priority"].isin(HIGH_LABELS).to_numpy()
    t_high = choose_t_high(p_high, y_high, TARGET_HIGH_RECALL)
    probs = [dict(zip(pm.classes_, row)) for row in P]
    pred_t = [apply_t_high(p, t_high) for p in probs]
    pred_argmax = pm.classes_[P.argmax(axis=1)]
    hp = {
        "t_high": t_high,
        "high_recall_argmax": round(float(np.isin(pred_argmax, list(HIGH_LABELS))[y_high].mean()), 4),
        "high_recall_t_high": round(float(np.isin(pred_t, list(HIGH_LABELS))[y_high].mean()), 4),
        "high_precision_t_high": round(float(y_high[np.isin(pred_t, list(HIGH_LABELS))].mean()), 4),
        "priority_macro_f1_t_high": round(f1_score(b["priority"], pred_t, average="macro"), 4),
    }
    print(f"[val] high-priority threshold: {hp}", flush=True)

    # Validation High-Priority *Incident* recall (classifier rule only; the gate adds more on top)
    tm = val_models["type"]
    both = va[va["type"].notna()]
    ty = tm.predict(both["text"])
    pp = [apply_t_high(dict(zip(pm.classes_, r)), t_high) for r in pm.predict_proba(both["text"])]
    true_hi_inc = (both["priority"].isin(HIGH_LABELS) & (both["type"] == "Incident")).to_numpy()
    pred_hi_inc = np.isin(pp, list(HIGH_LABELS)) & (ty == "Incident")
    hp["hp_incident_recall_rule1"] = round(float(pred_hi_inc[true_hi_inc].mean()), 4)
    print(f"[val] high-priority Incident recall via rule 1 alone: {hp['hp_incident_recall_rule1']}")

    config = {
        "trained_on": "kb",
        "n_train": int(len(kb)),
        "seed": SEED,
        "t_high": t_high,
        "high_labels": sorted(HIGH_LABELS),
        "target_high_recall": TARGET_HIGH_RECALL,
        "validation": {**val_metrics, "high_priority": hp},
        "fast": bool(args.fast),
    }
    if args.skip_final:
        print(json.dumps(config, indent=2))
        return

    final = fit_all(kb, args.fast, "final")
    for t in TARGETS:
        config["labels_" + t] = [str(c) for c in final[t].classes_]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "triage.joblib", compress=3)
    (MODELS_DIR / "triage_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved {MODELS_DIR / 'triage.joblib'} and triage_config.json")


if __name__ == "__main__":
    main()
