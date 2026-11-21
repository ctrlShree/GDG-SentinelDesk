"""Temperature-scale the triage classifier (Dev A). KB validation data only; no retraining.

    python scripts/calibrate_triage.py            # report only
    python scripts/calibrate_triage.py --write    # update models/triage_config.json

Reads the out-of-sample validation probabilities saved by train_triage.py
(models/val_preds.parquet). For each target, one temperature T minimises the negative
log-likelihood: p' = softmax(log p / T). The argmax never changes, so type/queue predictions
are identical; priority's t_high is re-tuned on the calibrated probabilities.

Honest check: T is fitted on one half of the validation set and ECE/NLL are reported on the
other half; the written T is then refitted on the whole validation set.

Run order after training: train_triage.py -> calibrate_triage.py --write -> tune_gate.py --write
-> evaluate.py.
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
from sentineldesk.triage import HIGH_LABELS, apply_t_high, choose_t_high, temper  # noqa: E402

TARGET_HIGH_RECALL = 0.80


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.1), np.log(5.0), 200))
    losses = [nll(temper(P, T), y) for T in grid]
    k = int(np.argmin(losses))
    fine = np.linspace(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], 101)
    return float(fine[int(np.argmin([nll(temper(P, T), y) for T in fine]))])


def matrix(vp: pd.DataFrame, t: str):
    m = vp[t].notna() & (vp[t].astype(str) != "")
    rows = [json.loads(x) for x in vp.loc[m, f"probs_{t}"]]
    labels = list(rows[0])
    P = np.array([[r[l] for l in labels] for r in rows])
    y = np.array([labels.index(v) for v in vp.loc[m, t].astype(str)])
    return labels, P, y, vp.loc[m, "priority"].astype(str).to_numpy() if t == "priority" else None


def stats(P, y):
    return {"ece": round(ece(P.max(1), P.argmax(1) == y), 4), "nll": round(nll(P, y), 4),
            "mean_conf": round(float(P.max(1).mean()), 4), "acc": round(float((P.argmax(1) == y).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    vp = pd.read_parquet(args.models_dir / "val_preds.parquet")
    cfg_path = args.models_dir / "triage_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)

    temps, report = {}, {}
    for t in TARGETS:
        labels, P, y, _ = matrix(vp, t)
        half = rng.permutation(len(y)) < len(y) // 2
        T_half = fit_temperature(P[half], y[half])
        before, after = stats(P[~half], y[~half]), stats(temper(P[~half], T_half), y[~half])
        temps[t] = round(fit_temperature(P, y), 4)
        report[t] = {"temperature": temps[t], "heldout_half_before": before, "heldout_half_after": after}
        print(f"{t:9s} T={temps[t]:.3f}  held-out ECE {before['ece']:.3f} -> {after['ece']:.3f}  "
              f"NLL {before['nll']:.3f} -> {after['nll']:.3f}  mean conf {before['mean_conf']:.3f} -> "
              f"{after['mean_conf']:.3f} (acc {before['acc']:.3f})")

    # Re-tune t_high on calibrated priority probabilities.
    labels, P, y, y_lab = matrix(vp, "priority")
    Pc = temper(P, temps["priority"])
    hi = [i for i, l in enumerate(labels) if l in HIGH_LABELS]
    y_high = np.isin(y_lab, list(HIGH_LABELS))
    t_high = choose_t_high(Pc[:, hi].sum(1), y_high, TARGET_HIGH_RECALL)
    pred = np.array([apply_t_high(dict(zip(labels, row)), t_high) for row in Pc])
    p_high = np.isin(pred, list(HIGH_LABELS))
    hp = {"t_high": t_high, "recall": round(float(p_high[y_high].mean()), 4),
          "precision": round(float(y_high[p_high].mean()), 4)}
    print(f"t_high on calibrated priority: {cfg.get('t_high')} -> {t_high} "
          f"(recall {hp['recall']}, precision {hp['precision']})")

    if args.write:
        cfg.setdefault("t_high_uncalibrated", cfg.get("t_high"))
        cfg["t_high"] = t_high
        cfg["temperature"] = temps
        cfg["calibration"] = {"method": "temperature scaling (NLL) on kb validation", **report,
                              "high_priority_calibrated": hp}
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"updated {cfg_path}")


if __name__ == "__main__":
    main()
