# Legacy copy of original rank script – unchanged

"""
rank.py - Intelligent Candidate Discovery & Ranking System
==========================================================
Redrob Data & AI Challenge - Offline Candidate Ranker

Architecture:
  1. Load pre-computed candidate embeddings (from precompute.py)
  2. Score candidates using 4-component weighted formula:
     - Semantic Similarity (35%): cosine sim vs JD embedding
     - Skill Depth Score (35%):   coverage & depth of required skills
     - Career Fit Score (30%):    YoE, company type, disqualifiers
     - Behavioral Modifiers:      response rate, recency, availability
  3. Apply honeypot hard-filters (score = 0.0)
  4. Sort top-100, generate reasoning, output submission.csv

Usage:
    python rank.py

Constraints:
  - Offline (no API calls)
  - CPU-only
  - <= 16 GB RAM
  - <= 5 min wall clock
"""

import json
import os
import csv
import time
import math
from datetime import datetime

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
OUTPUT_CSV = os.path.join(BASE_DIR, "submission.csv")

# ──────────────────────────────────────────────
# CONSTANTS & WEIGHTS
# ──────────────────────────────────────────────
W_SEMANTIC   = 0.35
W_SKILL      = 0.35
W_CAREER     = 0.30

# Skill proficiency scores
PROFICIENCY_MAP = {
    "expert":       1.00,
    "advanced":     0.75,
    "intermediate": 0.50,
    "beginner":     0.25,
}

# Core required skills (lower‑cased for matching)
CORE_REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "vector databases", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "hybrid search",
    "ndcg", "mrr", "map", "ranking evaluation", "a/b testing",
    "evaluation framework", "information retrieval",
    "python", "pytorch", "transformers", "bert", "llm",
    "fine-tuning", "rag", "retrieval-augmented generation",
    "machine learning", "deep learning",
    "lora", "qlora", "peft", "learning-to-rank", "xgboost",
    "recommendation system", "search", "ranking", "nlp",
}

BONUS_SKILLS = {"lora", "qlora", "peft", "learning-to-rank", "xgboost", "nlp", "open source", "open-source", "distributed systems"}

SERVICES_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis", "ltimindtree", "l&t infotech", "hexaware", "niit technologies", "zensar"}

REFERENCE_DATE = datetime(2026, 6, 1)

# (Original helper functions omitted for brevity – they remain unchanged)

if __name__ == "__main__":
    pass
