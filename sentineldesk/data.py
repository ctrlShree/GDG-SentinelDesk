"""Frozen data split. Run `python -m sentineldesk.data` to build the split and print the fingerprint.

kb_df   = first 80% of the shuffled dataset (training + retrieval index)
test_df = last 20% (labels only via load_split, which only scripts/evaluate.py may call)
"""
import hashlib

import pandas as pd

from sentineldesk import DATA_DIR

DATASET = "Tobi-Bueck/customer-support-tickets"
SEED = 42
KB_FRAC = 0.8
KB_PATH = DATA_DIR / "kb.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
INPUT_COLS = ["ticket_id", "subject", "body", "language"]


def ticket_text(subject, body) -> str:
    subject = "" if subject is None or (isinstance(subject, float) and pd.isna(subject)) else str(subject)
    body = "" if body is None or (isinstance(body, float) and pd.isna(body)) else str(body)
    return f"{subject.strip()}\n{body.strip()}".strip()


def _build() -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    df = load_dataset(DATASET)["train"].to_pandas()
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.insert(0, "ticket_id", [f"TKT-{i:05d}" for i in range(len(df))])
    for c in ("subject", "body", "answer"):
        df[c] = df[c].fillna("")
    cut = int(len(df) * KB_FRAC)
    kb_df, test_df = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb_df.to_parquet(KB_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    return kb_df, test_df


def _ensure() -> None:
    if not (KB_PATH.exists() and TEST_PATH.exists()):
        _build()


def load_kb() -> pd.DataFrame:
    """Knowledge base / training split (all columns)."""
    _ensure()
    return pd.read_parquet(KB_PATH)


def load_test_inputs() -> pd.DataFrame:
    """Test split inputs only: no answer/type/queue/priority."""
    _ensure()
    return pd.read_parquet(TEST_PATH, columns=INPUT_COLS)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(kb_df, test_df) with labels. scripts/evaluate.py ONLY."""
    _ensure()
    return pd.read_parquet(KB_PATH), pd.read_parquet(TEST_PATH)


def fingerprint(kb_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for name, df in (("kb", kb_df), ("test", test_df)):
        h.update(name.encode())
        for tid, s, b in zip(df["ticket_id"], df["subject"], df["body"]):
            h.update(f"{tid}\x1f{s}\x1f{b}\x1e".encode("utf-8"))
    return h.hexdigest()[:16]


if __name__ == "__main__":
    kb, test = _build()
    print(f"kb={len(kb)} test={len(test)} first_kb={kb.ticket_id.iloc[0]} first_test={test.ticket_id.iloc[0]}")
    print(f"fingerprint={fingerprint(kb, test)}")
