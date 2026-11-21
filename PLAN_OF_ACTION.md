# SentinelDesk: Merged Plan of Action (3 developers, Node.js frontend)

Repo: https://github.com/ctrlShree/GDG-SentinelDesk · This file replaces the old PLAN.md. Everyone's Claude Code reads it.
Problem: *SentinelDesk – Autonomous Incident Triage & Grounded Resolution Copilot* (GDG). Required: (1) triage classifier with holdout metrics, (2) escalation gate, (3) grounded multilingual RAG reply with ticket-ID citations, (4) live operator console. Bonuses: cross-lingual retrieval, PII masking, outage burst banner.

## Team
| Dev | Role | Owns |
|---|---|---|
| **A** | Skeleton, triage, escalation gate & evaluation | `__init__.py`, `data.py`, `lang.py`, `triage.py`, `escalation.py`, `train_triage.py`, `evaluate.py`, `edge_cases.jsonl`, `test_a_*.py`, `test_leakage.py` |
| **B** | Retrieval, PII, grounded generation & validator | `retrieval.py`, `build_index.py`, `pii.py`, `llm.py`, `generator.py`, `validator.py`, `test_b_rag.py` |
| **C** | Pipeline, REST API, **Node.js frontend**, burst & demo | `pipeline.py`, `burst.py`, `api.py` (FastAPI), `frontend/` (Next.js), `app.py` (fallback UI), `test_c_pipeline.py`, `README.md` |

Why this split: the **gate sits with A** (it uses A's confidences, and A's evaluation measures it), and the **validator sits with B** (it checks B's drafts). **PII moved from C to B** to make room for the Node.js frontend; it guards what goes into B's Gemini calls anyway.

## Architecture
```
Next.js console (frontend/, Node 20)  ──/api/* proxy──►  FastAPI (api.py, Python)  ──►  Pipeline().run
        :3000                                                 :8000                     (A + B modules)
```
The Python ML code stays in Python, and the frontend never touches models. It only calls the REST API.

## Working rules
1. **A pushes a skeleton with working stubs to `main` by 0:20**, including a stub `api.py`, so the frontend can call real endpoints from day one.
2. Edit **only files you own** (CODEOWNERS).
3. **Python function signatures and REST JSON shapes are frozen.** A change needs agreement from all three in chat.
4. `pytest -q` (and `npm run lint` in `frontend/` for C) → `git pull --rebase` → `git push`. Push small and often.
5. `data/`, `models/`, `reports/`, `.env`, `frontend/node_modules/`, and `frontend/.next/` are gitignored.
6. **Leakage rule:** the test split's `answer`/`type`/`queue`/`priority` never reach training, the index, or prompts. Only `scripts/evaluate.py` reads test labels, and `tests/test_leakage.py` enforces this.

## Stack
- **Backend (Python 3.11+):** pandas, scikit-learn, sentence-transformers, faiss-cpu, langdetect, google-genai, python-dotenv, **fastapi, uvicorn**, pytest. Streamlit only for the fallback UI.
- **Frontend (Node.js 20 LTS):** **Next.js 14 (App Router) + TypeScript + Tailwind CSS**. No UI kit required. Optional: `lucide-react` icons.
- Classifier: TF-IDF (word 1–2-gram + char 3–5-gram) feeding one LogisticRegression per target, `class_weight="balanced"`
- Embeddings: `intfloat/multilingual-e5-small` (local; `passage:`/`query:` prefixes), giving EN↔DE cross-lingual matching
- LLM: Gemini through **one helper** `llm.generate_json`, JSON mode, model from `GEMINI_MODEL`, with an **extractive template fallback**.

## Repo layout & frozen contracts
```
sentineldesk/
  __init__.py    ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR, TARGETS=("type","queue","priority")      [A]
  data.py        load_kb() · load_test_inputs() · load_split() [evaluate.py ONLY] · ticket_text(subject, body)  [A]
  lang.py        detect_language(text) -> "en" | "de"                                                [A]
  triage.py      Triager().predict(subject, body) -> {t: {"label","confidence","probs"}} for t in TARGETS  [A]
  escalation.py  pre_check(triage, hits, text) -> {"reasons": list[str], "hard_stop": bool}
                 decide(triage, hits, text, draft=None) -> {"escalate","status":"AUTO_DRAFT"|"ESCALATE","reasons"}  [A]
  retrieval.py   Retriever().search(text, k=5) -> [{"ticket_id","score","subject","body","answer",
                                                    "language","type","queue","priority"}]  (sorted desc)
                 Retriever().embed(texts, is_query=True) -> np.ndarray (L2-normalised)               [B]
  pii.py         mask_pii(text) -> (masked_text, kinds: list[str])                                   [B]
  llm.py         generate_json(system, prompt, schema, timeout_s=20) -> dict | None                  [B]
  validator.py   check_draft(reply, cited_ids, hits, ticket_text) -> {"grounded","cited_ids","issues"}  [B]
  generator.py   draft_reply(subject, body, lang, hits) -> {"reply","cited_ids","grounded","issues","fallback"}  [B]
  burst.py       BurstDetector().add(ticket_id, embedding, triage, ts=None, text="") ->
                   {"outage","cluster_size","label","members"}                                       [C]
  pipeline.py    Pipeline().run(subject, body) -> PipelineResult = {"ticket_id","language","pii_masked","masked_text",
                   "triage","hits","draft","draft_for_human_review","escalation","burst",
                   "latency_ms":{"pii","triage","retrieval","generation","gate","total"}}            [C]
                 Pipeline().reset_burst()                                                            [C]
api.py           FastAPI app (one Pipeline instance, CORS for localhost:3000)                        [C]
app.py           Streamlit fallback console (minimal; only if the frontend fails)                    [C]
frontend/        Next.js console (see below)                                                         [C]
scripts/train_triage.py · scripts/evaluate.py [A] · scripts/build_index.py [B]
tests/ test_a_triage.py, test_a_gate.py, edge_cases.jsonl, test_leakage.py [A] · test_b_rag.py [B] · test_c_pipeline.py, test_c_api.py [C]
requirements.txt (sectioned) · .env.example · .gitignore · README.md [C]
```

### REST API (frozen JSON shapes, owned by C)
| Method & path | Request | Response |
|---|---|---|
| `POST /api/triage` | `{"subject"?: str, "body": str}` | `PipelineResult` · 422 `{"detail"}` if body is empty |
| `POST /api/burst/simulate` | none | `{"results": PipelineResult[], "burst": {...}}` (4 VPN tickets, EN+DE) |
| `POST /api/burst/reset` | none | `{"ok": true}` |
| `GET /api/examples` | none | `[{"id","label","subject","body"}]` (demo presets from `edge_cases.jsonl`) |
| `GET /api/metrics` | none | `reports/metrics.json` contents, or `{"available": false}` |
| `GET /api/health` | none | `{"status":"ok","models_loaded":bool,"llm_available":bool}` |

### Frontend layout
```
frontend/
  package.json · next.config.mjs (rewrites /api/:path* -> ${API_URL}/api/:path*) · .env.local.example (API_URL=http://localhost:8000)
  app/layout.tsx · app/page.tsx · app/globals.css
  components/ OutageBanner · TicketForm · PredictionTiles · EscalationBadge · PrecedentsTable · DraftPanel · LatencyBreakdown · MetricsSidebar
  lib/types.ts (TS mirror of PipelineResult) · lib/api.ts (fetch helpers)
```

## Data split (frozen, exactly the problem-statement snippet)
`load_dataset("Tobi-Bueck/customer-support-tickets")["train"]`, `df.sample(frac=1.0, random_state=42).reset_index(drop=True)`, `ticket_id = TKT-{i:05d}`, first 80% → `kb_df`, last 20% → `test_df`. Null subjects become `""`. `python -m sentineldesk.data` prints a **fingerprint hash**, which all three developers compare at CP0.

## Pipeline order (C implements it)
mask PII → detect language → triage → retrieve (top 5) → `pre_check` → if `hard_stop`, **skip the LLM** → otherwise `draft_reply` (validator inside, one regeneration) → `decide` → burst update (burst state lives in the API process).

## Escalation rules (A), each producing a readable reason
① priority=high & type=Incident (hard stop) ② any confidence < 0.45 ③ top similarity < tuned τ_sim (hard stop) ④ unsupported or unsafe asks not covered by the hits ⑤ prompt injection ⑥ draft not grounded according to B's validator.

## Checkpoints
| CP | Time | Milestone |
|---|---|---|
| CP0 | 0:20 | Skeleton and stubs on `main`; `uvicorn api:app` returns stub JSON; Next.js app scaffolded; fingerprints match ×3 |
| CP1 | 1:00 | Classifier trained, index built, PII real; **Next.js console complete on stub API** |
| CP2 | 1:45 | All real modules wired behind the API; first true end-to-end run in the browser; `pytest` green |
| CP3 | 2:30 | **MVP done**: eval numbers in the metrics panel, edge cases pass, demo presets behave |
| CP4 | 4:30 | Hardening done, then **feature freeze**; `npm run build` passes |
| CP5 | 5:00 | **Code freeze**, tag `v1.0-demo`, fresh-clone test, rehearse twice, backup video |

> If your slot is only 3 hours, treat CP3 as the feature freeze. Keep the Streamlit `app.py` as an emergency fallback.

## Verification
```bash
pip install -r requirements.txt
python -m sentineldesk.data && python scripts/train_triage.py && python scripts/build_index.py
pytest -q && python scripts/evaluate.py --n 2000
uvicorn api:app --port 8000                            # terminal 1
cd frontend && npm install && npm run dev              # terminal 2 -> http://localhost:3000
```
Then check: the VPN outage escalates · the German reset gets a German draft with `[TKT-…]` citations · the refund demand escalates as unsupported · the injection is refused · an empty subject doesn't crash · the burst button raises the banner.

## 5-minute demo
1. Problem and architecture (30 s, C) → 2. Hospital VPN outage escalates with reasons (60 s, A) → 3. German reset gives a grounded German draft with cross-lingual citations (60 s, B) → 4. Refund/SLA demand: the validator blocks invented numbers and the gate escalates (45 s, B + A) → 5. Simulate burst shows the outage banner (45 s, C) → 6. Metrics panel and leakage statement (60 s, A).

## Copy-paste prompts for each person's Claude Code
**A:** `Repo GDG-SentinelDesk. Read PLAN.md and the dev-A files. I am Person A. First push the skeleton: every Python module in the layout with working stubs returning realistic dummy data matching the frozen contracts, a stub api.py exposing every REST endpoint in PLAN.md, a minimal Streamlit app.py, requirements and .gitignore. Then implement data.py, lang.py, triage.py, escalation.py, scripts/train_triage.py, scripts/evaluate.py, tests/edge_cases.jsonl, tests/test_a_triage.py, tests/test_a_gate.py, tests/test_leakage.py. Do not change signatures or touch other people's files.`
**B:** `Repo GDG-SentinelDesk. Read PLAN.md and the dev-B files. I am Person B. Use Gemini (google-genai, GEMINI_API_KEY / GEMINI_MODEL from .env). Implement sentineldesk/retrieval.py, scripts/build_index.py, sentineldesk/pii.py, sentineldesk/llm.py, sentineldesk/validator.py, sentineldesk/generator.py and tests/test_b_rag.py exactly to the contracts. Do not change signatures or touch other files.`
**C:** `Repo GDG-SentinelDesk. Read PLAN.md and the dev-C files. I am Person C. Implement sentineldesk/pipeline.py, sentineldesk/burst.py, api.py (FastAPI, exact REST shapes in PLAN.md), and a Next.js 14 + TypeScript + Tailwind operator console in frontend/ that calls the API through a /api rewrite, plus tests/test_c_pipeline.py, tests/test_c_api.py and README.md. Keep app.py as a minimal Streamlit fallback. Work against the stubs; do not touch other files.`
