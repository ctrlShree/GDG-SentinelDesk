"""Build the retrieval index from the KB split ONLY (never the test split).

    python scripts/build_index.py                 # full kb
    python scripts/build_index.py --subset 20000  # 20k random rows (seed 42) for a quick build
    python scripts/build_index.py --estimate      # time 1,000 embeddings and extrapolate; builds nothing

Writes models/kb.faiss (IndexFlatIP over L2-normalised e5 vectors = cosine) and models/kb_meta.parquet.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentineldesk import MODELS_DIR  # noqa: E402
from sentineldesk.data import load_kb, ticket_text  # noqa: E402
from sentineldesk.retrieval import INDEX_PATH, META_COLS, META_PATH, encode, get_model  # noqa: E402

SEED = 42


def load_corpus(subset: int | None):
    kb = load_kb()
    n0 = len(kb)
    kb = kb[kb["answer"].fillna("").str.strip() != ""]
    print(f"kb rows: {n0}, with non-empty answer: {len(kb)}")
    if subset and subset < len(kb):
        kb = kb.sample(n=subset, random_state=SEED)
        print(f"subset: {len(kb)} rows (seed {SEED})")
    kb = kb.reset_index(drop=True)
    for c in META_COLS:
        kb[c] = kb[c].fillna("").astype(str)
    texts = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    return kb, texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=int, default=None, help="index only N random kb rows (seed 42)")
    ap.add_argument("--estimate", action="store_true", help="time 1,000 embeddings, print the full-build estimate, exit")
    args = ap.parse_args()

    kb, texts = load_corpus(args.subset)

    t = time.perf_counter()
    get_model()
    print(f"model loaded in {time.perf_counter() - t:.1f}s")

    if args.estimate:
        sample = texts[:1000]
        encode(sample[:64], "passage: ")  # warm-up
        t = time.perf_counter()
        encode(sample, "passage: ")
        dt = time.perf_counter() - t
        per = dt / len(sample)
        print(f"{len(sample)} embeddings in {dt:.1f}s ({per * 1000:.1f} ms each)")
        print(f"estimated build for {len(texts)} rows: {per * len(texts) / 60:.1f} min")
        if per * len(texts) > 15 * 60:
            print("over 15 min -> consider: python scripts/build_index.py --subset 20000")
        return

    import faiss

    t = time.perf_counter()
    emb = encode(texts, "passage: ", show_progress=True)
    print(f"embedded {len(texts)} rows in {(time.perf_counter() - t) / 60:.1f} min, dim={emb.shape[1]}")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    kb[META_COLS].to_parquet(META_PATH, index=False)
    print(f"saved {INDEX_PATH} ({index.ntotal} vectors) and {META_PATH}")


if __name__ == "__main__":
    main()
