import numpy as np
from typing import List, Tuple

def rerank_top_k(candidate_ids: List[str], embeddings: np.ndarray, jd_embedding: np.ndarray, top_k: int = 5000) -> List[Tuple[str, float]]:
    """Lightweight reranking of top_k candidates.
    Uses cosine similarity (same as primary semantic score) but recomputes on the
    subset to allow a different weighting or future model swap.
    Returns a list of (candidate_id, refined_score) sorted descending.
    """
    # Compute cosine similarity for all candidates (embeddings are rows)
    # Normalize embeddings and JD vector
    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = np.where(emb_norm == 0, 1e-8, emb_norm)
    normed_emb = embeddings / emb_norm
    jd_norm = np.linalg.norm(jd_embedding)
    jd_norm = jd_norm if jd_norm != 0 else 1e-8
    normed_jd = jd_embedding / jd_norm
    scores = normed_emb @ normed_jd  # shape (N,)
    # Select top_k indices
    if top_k >= len(scores):
        top_k_idx = np.argsort(-scores)
    else:
        top_k_idx = np.argpartition(-scores, top_k)[:top_k]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
    result = [(candidate_ids[i], float(scores[i])) for i in top_k_idx]
    return result
