# Escalation & safety gate

`sentineldesk/escalation.py`. Every rule that fires adds one readable reason to the console.
Any reason → **ESCALATE**; no reasons → **AUTO_DRAFT**. A *hard stop* also skips the LLM entirely.

| # | Rule | Hard stop | Reason text starts with |
|---|---|---|---|
| 1 | priority ∈ {high, critical} **and** type = Incident, **or** outage language ("down for all staff", "nobody can", "ausgefallen", "nicht erreichbar"…) on an Incident/Problem/high-priority ticket | ✅ | `Critical:` |
| 2 | any classifier confidence below its threshold (type 0.50 · queue 0.20 · priority 0.45) | | `Low classifier confidence:` |
| 3 | top retrieval similarity < τ_sim (or no hits) | ✅ | `No sufficiently similar precedent` |
| 4 | risky ask (refund, pricing, legal/SLA/guarantee, destructive, credentials; EN + DE lexicons) that no retrieved precedent *answer* covers | | `Unsupported request` |
| 5 | prompt injection ("ignore previous instructions", "system prompt", "you are now", "ignoriere … Anweisungen"…) | ✅ | `Possible prompt injection` |
| 6 | B's validator says the draft is not grounded (unknown citation, invented number, unsupported commitment) | | `Draft not grounded in evidence` |

Order in the pipeline: mask PII → language → triage → retrieve top 5 → `pre_check` (rules 1–5) → if hard stop, no LLM call → `draft_reply` → `decide` (rules 1–6).

## How the thresholds were chosen (KB only, never the test split)

**`t_high` = 0.38 (priority).** `scripts/train_triage.py` holds out 10% of the KB (seed 42). The priority model predicts *high-priority* when P(high) + P(critical) ≥ `t_high`; we take the largest `t_high` whose validation recall for high/critical is ≥ 0.80 (0.805, precision 0.770; plain argmax gave recall 0.779). Missing an outage costs more than an extra escalation.

**Confidence thresholds.** `scripts/tune_gate.py` uses the *out-of-sample* validation probabilities (`models/val_preds.parquet`). Confidences are bucketed into 0.05 bands; each threshold is the top of the highest band where accuracy falls below a bar:

| Target | Bar | Threshold | Validation tickets flagged | Accuracy below / above |
|---|---|---|---|---|
| type | 0.50 (wrong more often than right) | **0.50** | 0.9% | 0.42 / 0.88 |
| priority | 0.50 (drives paging) | **0.45** | 10.3% | 0.41 / 0.78 |
| queue | 0.25 | **0.20** | 5.1% | 0.22 / 0.65 |

Queue gets a lower bar on purpose: with ~50 classes a 40%-accurate guess is far above chance, and a wrong queue only misroutes. The strict bar would have flagged a third of all tickets. Result: escalation rate 31% on KB validation tickets and 41.5% on the test sample, inside the 20–45% sanity range.

**τ_sim.** Placeholder 0.80 until B's real retriever lands. `tune_gate.py` then takes the top-1 score (self-hit removed) for 200 KB validation tickets against 15 off-domain queries and picks the score that separates them best (Youden J), written to `models/gate.json`.

To retune: `python scripts/train_triage.py && python scripts/tune_gate.py --write`.

## Holdout results (`python scripts/evaluate.py --n 2000`, LLM off)
- **High-Priority Incident Recall: 0.832** (1,692 true high/critical Incidents in the test split, escalated by the full pipeline); 0.772 are paged as `Critical`.
- Classifier high-priority recall 0.819 (precision 0.765).
- Gate reasons on 2,000 sampled tickets: Critical 483 · Low confidence 352 · Unsupported request 69.

## Known limitations
- The demo presets are shorter and less formal than the synthetic training tickets. Some come out more cautious than intended: the German password reset is classified as a high-priority Incident, and the sales question gets queue confidence 0.18. Both escalate, which is the safe direction.
- The poem/gibberish presets rely on rule 3 and only behave once B's real retriever replaces the stub (which returns 0.92 for everything).
- Lexicons are keyword-based, so a politely paraphrased refund request without the keywords can slip past rule 4. Rule 6 (validator) is the backstop.
