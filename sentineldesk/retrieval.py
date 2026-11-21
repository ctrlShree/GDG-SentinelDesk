"""Dense retrieval over the KB: multilingual-e5-small embeddings + FAISS inner-product index.

Build the index first with `python scripts/build_index.py` (writes models/kb.faiss + models/kb_meta.parquet).
"""
import numpy as np

from sentineldesk import MODELS_DIR

MODEL_NAME = "intfloat/multilingual-e5-small"
INDEX_PATH = MODELS_DIR / "kb.faiss"
META_PATH = MODELS_DIR / "kb_meta.parquet"
META_COLS = ["ticket_id", "subject", "body", "answer", "language", "type", "queue", "priority"]
BATCH_SIZE = 64

_model = None   # shared across Retriever instances (pipeline + tests) so the model loads once
_index = None
_meta = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
    """Embed texts with an e5 prefix ("query: " or "passage: "). Returns float32, L2-normalised rows."""
    model = get_model()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    v = model.encode([prefix + (t or "") for t in texts], batch_size=BATCH_SIZE,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=show_progress)
    return np.ascontiguousarray(v, dtype="float32")


def _load_index():
    global _index, _meta
    if _index is None:
        if not (INDEX_PATH.exists() and META_PATH.exists()):
            raise FileNotFoundError(f"{INDEX_PATH} / {META_PATH} missing: run `python scripts/build_index.py` first")
        import faiss
        import pandas as pd
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = pd.read_parquet(META_PATH)[META_COLS].fillna("").astype(str).to_dict("records")
        if _index.ntotal != len(_meta):
            raise RuntimeError(f"index has {_index.ntotal} vectors but meta has {len(_meta)} rows: rebuild the index")
    return _index, _meta


class Retriever:
    dim = 384

    def __init__(self):
        self.index, self.meta = _load_index()
        get_model()

    def embed(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        return encode(list(texts), "query: " if is_query else "passage: ")

    def search(self, text: str, k: int = 5) -> list[dict]:
        if not (text or "").strip() or k <= 0:
            return []
        q = self.embed([text], is_query=True)
        scores, idx = self.index.search(q, min(k, self.index.ntotal))
        hits = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            hits.append({**self.meta[i], "score": float(s)})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits
