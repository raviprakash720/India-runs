"""
precompute.py - Offline Pre-computation for Candidate Ranker
=============================================================
This script computes and caches features from the 100K candidate pool
to enable fast scoring at rank time.

Two modes:
  MODE 1 (Fast, ~30s): TF-IDF semantic proxy using JD keyword vectors.
                       No neural model needed. Works offline instantly.
  MODE 2 (Full, ~3min): sentence-transformer embeddings using a 3-layer
                        MiniLM model (paraphrase-MiniLM-L3-v2) which is
                        ~3x faster than L6.

Usage:
    python precompute.py          # uses MODE 1 by default (fast)
    python precompute.py --full   # uses MODE 2 (neural)
"""

import json
import os
import sys
import time
import math
import numpy as np
from datetime import datetime
from collections import Counter

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_FILE = os.path.join(
    BASE_DIR,
    "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "candidates.jsonl"
)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# JD KEYWORD VOCABULARY
# Terms extracted from the job description with importance weights
# ──────────────────────────────────────────────
JD_TERMS = {
    # Critical (weight 3.0) — core required
    "embedding": 3.0, "embeddings": 3.0, "sentence-transformer": 3.0,
    "sentence transformer": 3.0, "vector database": 3.0, "vector search": 3.0,
    "pinecone": 3.0, "weaviate": 3.0, "qdrant": 3.0, "milvus": 3.0,
    "faiss": 3.0, "opensearch": 3.0, "hybrid search": 3.0,
    "ndcg": 3.0, "mrr": 3.0, "retrieval": 3.0, "ranking": 3.0,
    "information retrieval": 3.0,
    # Important (weight 2.0)
    "python": 2.0, "pytorch": 2.0, "transformers": 2.0, "bert": 2.0,
    "llm": 2.0, "fine-tuning": 2.0, "rag": 2.0, "nlp": 2.0,
    "evaluation framework": 2.0, "a/b testing": 2.0, "search": 2.0,
    "machine learning": 2.0, "deep learning": 2.0, "recommendation": 2.0,
    # Nice-to-have (weight 1.0)
    "lora": 1.0, "qlora": 1.0, "peft": 1.0, "xgboost": 1.0,
    "learning-to-rank": 1.0, "elasticsearch": 1.0, "open source": 1.0,
    "distributed": 1.0, "inference": 1.0, "startup": 1.0,
    "product": 1.0, "scale": 1.0, "production": 1.0,
}

# Pre-build JD term list sorted by length descending (for greedy matching)
JD_TERMS_SORTED = sorted(JD_TERMS.keys(), key=len, reverse=True)


def compute_tfidf_semantic_score(text):
    """
    Fast keyword-overlap semantic score.
    Computes weighted term frequency of JD keywords in candidate text.
    Normalized to [0, 1].
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.0
    max_possible = sum(JD_TERMS.values())

    for term in JD_TERMS_SORTED:
        if term in text_lower:
            score += JD_TERMS[term]

    return min(score / (max_possible * 0.15), 1.0)  # normalize: 15% coverage = score 1.0


def build_candidate_text(cand):
    """Build rich text representation of a candidate."""
    profile  = cand.get("profile", {})
    skills   = cand.get("skills", [])
    career   = cand.get("career_history", [])

    parts = []
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    if profile.get("current_title"):
        parts.append(profile["current_title"])

    # Add skill names (weighted by proficiency)
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        if name:
            # Repeat expert skills to up-weight them
            if prof == "expert":
                parts.extend([name, name, name])
            elif prof == "advanced":
                parts.extend([name, name])
            else:
                parts.append(name)

    # Add job titles and companies
    for job in career:
        if job.get("title"):
            parts.append(job["title"])
        if job.get("description"):
            parts.append(job["description"][:200])  # first 200 chars

    return " ".join(parts)


def main():
    t0 = time.time()
    full_mode = "--full" in sys.argv

    print("=" * 60)
    print(f"Pre-computation Mode: {'NEURAL (sentence-transformers)' if full_mode else 'FAST (TF-IDF keyword)'}")
    print("=" * 60)

    # ── Load candidates ──
    print("\n[1/3] Loading candidate profiles...")
    candidate_ids = []
    texts = []

    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidate_ids.append(cand["candidate_id"])
            texts.append(build_candidate_text(cand))

    print(f"  Loaded {len(candidate_ids):,} candidates.")

    # ── Compute semantic scores ──
    print(f"\n[2/3] Computing semantic scores...")

    if full_mode:
        try:
            from domain_finetune import load_model
            print("  Loading domain fine-tuned model or fallback base model...")
            model = load_model()
            if model is None:
                raise ImportError()
        except (ImportError, ModuleNotFoundError):
            from sentence_transformers import SentenceTransformer
            print("  Loading paraphrase-MiniLM-L3-v2 (3-layer, ~3x faster than L6)...")
            model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        jd_query = (
            "Senior AI Engineer with experience in embeddings-based retrieval, "
            "sentence-transformers, vector databases Pinecone Weaviate Qdrant FAISS, "
            "hybrid search, Python, evaluation frameworks NDCG MRR MAP, LLM fine-tuning, "
            "RAG, information retrieval, ranking systems at product companies."
        )
        print("  Encoding JD query...")
        jd_embedding = model.encode(jd_query, convert_to_numpy=True)

        print(f"  Encoding {len(texts):,} candidate texts...")
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Save neural embeddings
        np.save(os.path.join(CACHE_DIR, "candidate_embeddings.npy"), embeddings.astype(np.float32))
        np.save(os.path.join(CACHE_DIR, "jd_embedding.npy"), jd_embedding.astype(np.float32))
        print("  Saved neural embeddings.")

    else:
        # Fast TF-IDF keyword-based semantic scores
        print(f"  Computing keyword-overlap scores for {len(texts):,} candidates...")
        scores = np.array([compute_tfidf_semantic_score(t) for t in texts], dtype=np.float32)

        # Save as "embeddings" in 1D form — rank.py handles both 1D and 2D
        np.save(os.path.join(CACHE_DIR, "candidate_semantic_scores.npy"), scores)
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}], mean={scores.mean():.4f}")

        # Also create dummy embedding/jd files so rank.py knows to use the scores file
        np.save(os.path.join(CACHE_DIR, "jd_embedding.npy"), np.array([1.0], dtype=np.float32))

    # ── Save ID mapping ──
    ids_path = os.path.join(CACHE_DIR, "candidate_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    print(f"  Saved ID mapping -> {ids_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pre-computation done in {elapsed:.1f}s")
    print(f"Cache directory: {CACHE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
