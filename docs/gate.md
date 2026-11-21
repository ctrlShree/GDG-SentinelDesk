# Escalation & safety gate

`sentineldesk/escalation.py`. Every rule that fires adds one readable reason to the console.
Any reason → **ESCALATE**; no reasons → **AUTO_DRAFT**. A *hard stop* also skips the LLM entirely.

| # | Rule | Hard stop | Reason text starts with |
|---|---|---|---|
| 1 | priority ∈ {high, critical} **and** type = Incident, **or** outage language ("down for all staff", "nobody can", "ausgefallen", "nicht erreichbar"…) on an Incident/Problem/high-priority ticket | ✅ | `Critical:` |
| 2 | any (calibrated) classifier confidence below its threshold (type 0.45 · queue 0.25 · priority 0.50) | | `Low classifier confidence:` |
| 3 | top retrieval similarity < τ_sim (or no hits) | ✅ | `No sufficiently similar precedent` |
| 4 | risky ask (refund, pricing, legal/SLA/guarantee, destructive, credentials; EN + DE lexicons) that no retrieved precedent *answer* covers | | `Unsupported request` |
| 5 | prompt injection ("ignore previous instructions", "system prompt", "you are now", "ignoriere … Anweisungen"…) | ✅ | `Possible prompt injection` |
| 6 | B's validator says the draft is not grounded (unknown citation, invented number, unsupported commitment) | | `Draft not grounded in evidence` |

Order in the pipeline: mask PII → language → triage → retrieve top 5 → `pre_check` (rules 1–5) → if hard stop, no LLM call → `draft_reply` → `decide` (rules 1–6).

## How the thresholds were chosen (KB only, never the test split)

**Calibration (temperature scaling).** The class-balanced logistic regressions were *under*confident (validation queue: mean confidence 0.49 vs accuracy 0.63), which made ordinary tickets look "low confidence". `scripts/calibrate_triage.py` fits one temperature per target on the out-of-sample validation probabilities, minimising log-loss: p' = softmax(log p / T). It never changes the predicted label, only the confidence. Fitted on one half of the validation set and measured on the other half:

| Target | T | ECE before → after | Mean confidence → after (accuracy) |
|---|---|---|---|
| type | 0.83 | 0.026 → 0.014 | 0.857 → 0.878 (0.880) |
| queue | 0.61 | 0.137 → 0.021 | 0.487 → 0.640 (0.624) |
| priority | 0.80 | 0.059 → 0.021 | 0.685 → 0.733 (0.741) |

The confidences in the console now mean what they say: a 0.7 is right about 70% of the time.

**`t_high` = 0.39 (priority, on calibrated probabilities).** `scripts/train_triage.py` holds out 10% of the KB (seed 42). The priority model predicts *high-priority* when P(high) + P(critical) ≥ `t_high`; we take the largest `t_high` whose validation recall for high/critical is ≥ 0.80 (0.801, precision 0.774; plain argmax gave recall 0.779). Missing an outage costs more than an extra escalation.

**Confidence thresholds.** `scripts/tune_gate.py` uses the same out-of-sample, calibrated validation probabilities. Confidences are bucketed into 0.05 bands; each threshold is the top of the highest band where accuracy falls below a bar, with a floor so no rule silently switches off:

| Target | Bar | Threshold | Validation tickets flagged | Accuracy below / above |
|---|---|---|---|---|
| type | 0.50 (wrong more often than right) | **0.45** (floor; no band that bad) | 0.1% | – / 0.88 |
| priority | 0.50 (drives paging) | **0.50** | 13.6% | 0.43 / 0.79 |
| queue | 0.25 | **0.25** | 2.8% | 0.19 / 0.64 |

Queue gets a lower bar on purpose: with ~50 classes a 40%-accurate guess is far above chance, and a wrong queue only misroutes. The strict bar would have flagged about a third of all tickets. Result: escalation rate 31.5% on KB validation tickets and 40.8% on the test sample, inside the 20–45% sanity range.

**τ_sim.** Placeholder 0.80 until B's real retriever lands. `tune_gate.py` then takes the top-1 score (self-hit removed) for 200 KB validation tickets against 15 off-domain queries and picks the score that separates them best (Youden J), written to `models/gate.json`.

To retrain and retune: `make train` (= `train_triage.py` → `calibrate_triage.py --write` → `tune_gate.py --write`), then `python scripts/evaluate.py --n 2000`.

## Holdout results (`python scripts/evaluate.py --n 2000`, LLM off)
- Macro-F1 type 0.894 · queue 0.897 · priority 0.805 (unseen-text subset 0.856 · 0.882 · 0.768).
- **High-Priority Incident Recall: 0.822** (1,692 true high/critical Incidents in the test split, escalated by the full pipeline); 0.765 are paged as `Critical`.
- Classifier high-priority recall 0.814 (precision 0.771).
- Gate reasons on 2,000 sampled tickets: Critical 480 · Low confidence 349 · Unsupported request 69.

## Known limitations
- The demo presets are shorter and less formal than the synthetic training tickets. Some come out more cautious than intended: the German password reset is classified as a high-priority Incident, and the sales question gets calibrated queue confidence 0.23. Both escalate, which is the safe direction.
- The poem/gibberish presets rely on rule 3 and only behave once B's real retriever replaces the stub (which returns 0.92 for everything).
- Lexicons are keyword-based, so a politely paraphrased refund request without the keywords can slip past rule 4. Rule 6 (validator) is the backstop.
