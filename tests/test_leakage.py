"""Only scripts/evaluate.py (and data.py, which defines it) may access the test split's labels."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = {"scripts/evaluate.py", "sentineldesk/data.py", "tests/test_leakage.py"}
FORBIDDEN = ("load_split", "test.parquet")
SKIP = {".git", ".venv", "venv", "__pycache__"}


def test_test_split_only_in_eval():
    bad = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if set(p.relative_to(ROOT).parts) & SKIP or rel in ALLOWED:
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        if any(f in src for f in FORBIDDEN):
            bad.append(rel)
    assert not bad, f"test split accessed outside evaluate.py: {bad}"


def test_index_has_no_test_ids():
    meta = ROOT / "models" / "kb_meta.parquet"
    test = ROOT / "data" / "test.parquet"
    if not (meta.exists() and test.exists()):
        return  # artifacts not built yet
    import pandas as pd
    overlap = set(pd.read_parquet(meta, columns=["ticket_id"]).ticket_id) & \
              set(pd.read_parquet(test, columns=["ticket_id"]).ticket_id)
    assert not overlap, f"{len(overlap)} test tickets leaked into the index"
