"""
skill_cooccurrence.py - Skill Co-Occurrence Synergy Scoring
============================================================
Redrob Data & AI Challenge - Skill Pair Synergy Module

Rationale:
    Individual skill matching catches *breadth* but misses the
    combinatorial signal: a candidate who knows BOTH "faiss" AND
    "pytorch" is far more likely to have built real retrieval
    systems than someone who lists them separately on unrelated
    projects.  This module rewards candidates whose skill sets
    contain pairs that co-occur in production AI/ML engineering
    workflows.

Usage:
    from skill_cooccurrence import compute_cooccurrence_bonus

    bonus = compute_cooccurrence_bonus(candidate["skills"])
    # bonus ∈ [0.0, 1.0]
"""

import math


# ──────────────────────────────────────────────
# CO-OCCURRENCE PAIR DEFINITIONS
# ──────────────────────────────────────────────
# Each key is a frozenset of two lowercased skill names.
# The value is a synergy weight in [0.0, 1.0] — higher means
# the combination is a stronger positive signal for the
# "Senior AI Engineer – Founding Team" role.

CO_OCCURRENCE_PAIRS = {
    # ── Embeddings & Retrieval Stack ──────────────
    frozenset({"embeddings", "sentence-transformers"}):      0.90,
    frozenset({"embeddings", "faiss"}):                      0.85,
    frozenset({"embeddings", "vector database"}):            0.85,
    frozenset({"sentence-transformers", "faiss"}):           0.80,
    frozenset({"sentence-transformers", "pytorch"}):         0.75,

    # ── RAG Pipeline ─────────────────────────────
    frozenset({"rag", "vector database"}):                   0.85,
    frozenset({"rag", "llm"}):                               0.90,
    frozenset({"rag", "embeddings"}):                        0.85,
    frozenset({"rag", "elasticsearch"}):                     0.70,
    frozenset({"rag", "opensearch"}):                        0.70,

    # ── Vector DB Combinations ───────────────────
    frozenset({"faiss", "pytorch"}):                         0.80,
    frozenset({"pinecone", "embeddings"}):                   0.80,
    frozenset({"weaviate", "embeddings"}):                   0.80,
    frozenset({"qdrant", "embeddings"}):                     0.80,
    frozenset({"milvus", "embeddings"}):                     0.80,

    # ── LLM & Fine-Tuning ────────────────────────
    frozenset({"llm", "fine-tuning"}):                       0.80,
    frozenset({"llm", "transformers"}):                      0.75,
    frozenset({"fine-tuning", "lora"}):                      0.85,
    frozenset({"fine-tuning", "qlora"}):                     0.85,
    frozenset({"fine-tuning", "peft"}):                      0.80,
    frozenset({"llm", "python"}):                            0.55,

    # ── Core ML / DL ─────────────────────────────
    frozenset({"pytorch", "transformers"}):                  0.80,
    frozenset({"pytorch", "deep learning"}):                 0.75,
    frozenset({"machine learning", "python"}):               0.60,
    frozenset({"deep learning", "machine learning"}):        0.55,
    frozenset({"bert", "transformers"}):                     0.80,
    frozenset({"bert", "nlp"}):                              0.75,

    # ── Search & Evaluation ──────────────────────
    frozenset({"ndcg", "information retrieval"}):            0.75,
    frozenset({"mrr", "information retrieval"}):             0.75,
    frozenset({"learning-to-rank", "xgboost"}):              0.70,
    frozenset({"hybrid search", "elasticsearch"}):           0.70,
    frozenset({"search", "ranking"}):                        0.65,
    frozenset({"a/b testing", "evaluation framework"}):      0.60,
}

# Pre-compute the theoretical maximum so we can normalise in O(1).
_MAX_POSSIBLE_BONUS = sum(CO_OCCURRENCE_PAIRS.values())


# ──────────────────────────────────────────────
# BONUS COMPUTATION
# ──────────────────────────────────────────────
def compute_cooccurrence_bonus(candidate_skills: list) -> float:
    """Return a normalised synergy bonus for a candidate's skill list.

    Parameters
    ----------
    candidate_skills : list[dict]
        Each dict has at minimum a ``"name"`` key (str).  The fields
        ``"proficiency"``, ``"duration_months"``, and ``"endorsements"``
        are accepted but not required — the co-occurrence score is
        purely presence-based so even a beginner-level listing counts.

    Returns
    -------
    float
        A score in **[0.0, 1.0]** where 1.0 means the candidate
        possesses every synergistic skill pair defined above.

    Examples
    --------
    >>> skills = [{"name": "FAISS"}, {"name": "PyTorch"}, {"name": "RAG"}]
    >>> 0.0 <= compute_cooccurrence_bonus(skills) <= 1.0
    True
    """
    if not candidate_skills or _MAX_POSSIBLE_BONUS == 0:
        return 0.0

    # Build a lowercased set of the candidate's skill names.
    skill_names = set()
    for s in candidate_skills:
        name = (s.get("name") or "").strip().lower()
        if name:
            skill_names.add(name)

    if len(skill_names) < 2:
        return 0.0

    # Accumulate raw synergy from every matched pair.
    raw_bonus = 0.0
    for pair, weight in CO_OCCURRENCE_PAIRS.items():
        if pair.issubset(skill_names):
            raw_bonus += weight

    # Normalise into [0, 1] — uses sqrt scaling so that diminishing
    # returns kick in naturally: having 5 pairs is great, but having
    # 30 pairs doesn't become absurdly dominant.
    linear_ratio = raw_bonus / _MAX_POSSIBLE_BONUS
    normalised   = math.sqrt(linear_ratio)          # sqrt compresses top-end
    return min(normalised, 1.0)
