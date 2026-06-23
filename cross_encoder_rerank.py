"""
cross_encoder_rerank.py - Cross-Encoder Reranking Module
========================================================
Runs a lightweight cross-encoder on the top-k candidates to refine
semantic similarity scores beyond what bi-encoder cosine similarity provides.

The cross-encoder scores each (JD, candidate_text) pair jointly, which is
more accurate but slower. We only run it on the top-k candidates (default
5000) pre-filtered by the fast bi-encoder cosine scores.

Fallback: If the cross-encoder model cannot be loaded (missing dependency,
no internet, etc.), we fall back to an enhanced cosine reranking that
re-normalises within the top-k pool for finer granularity.

Usage:
    from cross_encoder_rerank import rerank_candidates
    refined = rerank_candidates(scored_list, candidate_texts, jd_text)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────
# CROSS-ENCODER MODEL (optional dependency)
# ──────────────────────────────────────────────
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False

_model_cache = None


def _get_cross_encoder():
    """Lazily load the cross-encoder model (cached after first call)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _HAS_CROSS_ENCODER:
        return None
    try:
        _model_cache = _CrossEncoder(_CROSS_ENCODER_MODEL)
        return _model_cache
    except Exception as e:
        print(f"  [CrossEncoder] Could not load model: {e}")
        return None


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def rerank_candidates(
    scored_list: List[Dict],
    candidate_texts: Dict[str, str],
    jd_text: str,
    top_k: int = 5000,
    blend_weight: float = 0.35,
) -> List[Dict]:
    """Rerank the top-k candidates using a cross-encoder or enhanced cosine.

    Args:
        scored_list:     List of dicts with keys 'candidate_id', 'semantic_score',
                         and other score fields. Must be sorted by final_score descending.
        candidate_texts: Dict mapping candidate_id -> concatenated text for that candidate.
        jd_text:         The job description text used as the query.
        top_k:           Number of top candidates to rerank (default 5000).
        blend_weight:    Weight given to cross-encoder score when blending.
                         Final = (1-blend_weight)*original + blend_weight*cross_encoder.

    Returns:
        The same scored_list with 'semantic_score' updated for the top-k candidates
        and a new key 'cross_encoder_score' added where applicable.
    """
    if not scored_list:
        return scored_list

    top_k = min(top_k, len(scored_list))
    top_candidates = scored_list[:top_k]
    rest = scored_list[top_k:]

    model = _get_cross_encoder()

    if model is not None and jd_text:
        print(f"  [CrossEncoder] Reranking top {top_k} candidates with {_CROSS_ENCODER_MODEL}...")
        top_candidates = _cross_encoder_rerank(
            model, top_candidates, candidate_texts, jd_text, blend_weight
        )
    else:
        print(f"  [CrossEncoder] Model unavailable; using enhanced cosine reranking for top {top_k}.")
        top_candidates = _enhanced_cosine_rerank(top_candidates)

    return top_candidates + rest


def _cross_encoder_rerank(
    model,
    candidates: List[Dict],
    candidate_texts: Dict[str, str],
    jd_text: str,
    blend_weight: float,
) -> List[Dict]:
    """Run cross-encoder scoring and blend with original semantic scores."""
    pairs = []
    valid_indices = []

    for i, item in enumerate(candidates):
        cid = item["candidate_id"]
        ctext = candidate_texts.get(cid, "")
        if ctext:
            # Cross-encoder expects (query, document) pairs
            pairs.append([jd_text, ctext[:512]])  # truncate to 512 chars for speed
            valid_indices.append(i)

    if not pairs:
        return candidates

    try:
        # Predict relevance scores in batch
        raw_scores = np.array(model.predict(pairs, batch_size=128))
    except Exception as e:
        print(f"  [CrossEncoder] Prediction error: {e}")
        return candidates

    # Normalize cross-encoder scores to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.full_like(raw_scores, 0.5)

    # Blend with original semantic score
    for j, idx in enumerate(valid_indices):
        original = candidates[idx]["semantic_score"]
        cross_score = float(norm_scores[j])
        blended = (1 - blend_weight) * original + blend_weight * cross_score
        candidates[idx]["semantic_score"] = blended
        candidates[idx]["cross_encoder_score"] = cross_score

    return candidates


def _enhanced_cosine_rerank(candidates: List[Dict]) -> List[Dict]:
    """Fallback: re-normalize semantic scores within the top-k pool.

    This gives finer granularity among the top candidates even without
    a cross-encoder model, by stretching the score range to [0, 1]
    within just the top-k subset.
    """
    if not candidates:
        return candidates

    scores = np.array([c["semantic_score"] for c in candidates])
    s_min, s_max = scores.min(), scores.max()

    if s_max > s_min:
        renorm = (scores - s_min) / (s_max - s_min)
    else:
        renorm = np.full_like(scores, 0.5)

    for i, c in enumerate(candidates):
        c["semantic_score"] = float(renorm[i])
        c["cross_encoder_score"] = None  # not available

    return candidates
