"""
lgbm_ensemble.py - LightGBM Non-Linear Score Refinement
========================================================
Redrob Data & AI Challenge - Offline Candidate Ranker

Trains a lightweight gradient-boosted tree on engineered features
(semantic, skill, career, behavioral) to capture non-linear
interactions that the weighted-linear formula in rank.py cannot.

Since this is an OFFLINE challenge with NO labeled training data,
pseudo-labels are generated from the composite score itself:
  y = semantic * W_S + skill * W_SK + career * W_C  (× behavioral mult)

The LightGBM model learns residual non-linear interactions between
features (e.g., "high skill + low career → different correction than
high skill + high career") that a linear blend misses.

Usage:
    from lgbm_ensemble import train_and_predict
    refined = train_and_predict(scored_candidates)
    if refined is not None:
        for item in scored:
            item["final_score"] = refined[item["candidate_id"]]
"""

import math

# ──────────────────────────────────────────────
# OPTIONAL DEPENDENCY — graceful fallback
# ──────────────────────────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    lgb = None
    _HAS_LGBM = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False


# ──────────────────────────────────────────────
# FEATURE NAMES (column order in X)
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "semantic_score",
    "skill_score",
    "career_score",
    "behavioral_multiplier",
    "years_of_experience",
    "num_skills",
    "num_career_entries",
    "response_rate",
    "notice_period_days",
]


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────
def _extract_row(item: dict) -> list:
    """Extract a single feature vector from a scored-candidate dict.

    Parameters
    ----------
    item : dict
        One element from the scored candidates list produced by rank.py.
        Expected keys: semantic_score, skill_score, career_score,
        multiplier, cand (the raw candidate dict).

    Returns
    -------
    list[float]
        Feature values in the order defined by FEATURE_NAMES.
    """
    cand    = item.get("cand", {})
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0) or 0
    num_skills = len(cand.get("skills", []))
    num_career = len(cand.get("career_history", []))
    response_rate = (
        signals.get("recruiter_response_rate")
        or profile.get("recruiter_response_rate")
        or 0.5
    )
    notice_days = (
        signals.get("notice_period_days")
        or profile.get("notice_period_days")
        or 0
    )
    if notice_days is None:
        notice_days = 0

    return [
        float(item.get("semantic_score", 0.5)),
        float(item.get("skill_score", 0.0)),
        float(item.get("career_score", 0.5)),
        float(item.get("multiplier", 1.0)),
        float(yoe),
        float(num_skills),
        float(num_career),
        float(response_rate),
        float(notice_days),
    ]


def prepare_features(scored_candidates: list) -> tuple:
    """Build feature matrix X and pseudo-label vector y.

    Parameters
    ----------
    scored_candidates : list[dict]
        List of scored candidate dicts from rank.py.  Each dict must
        contain at least the keys: semantic_score, skill_score,
        career_score, multiplier, final_score, cand.

    Returns
    -------
    (X, y) : tuple[np.ndarray, np.ndarray]
        X has shape (N, len(FEATURE_NAMES));  y has shape (N,).
        Returns (None, None) when numpy is unavailable.
    """
    if not _HAS_NUMPY:
        return None, None

    rows = [_extract_row(item) for item in scored_candidates]
    X = np.array(rows, dtype=np.float32)

    # Pseudo-labels: the existing composite final_score
    y = np.array(
        [float(item.get("final_score", 0.0)) for item in scored_candidates],
        dtype=np.float32,
    )

    return X, y


# ──────────────────────────────────────────────
# TRAIN + PREDICT
# ──────────────────────────────────────────────
def train_and_predict(scored_candidates: list) -> dict | None:
    """Train a small LightGBM model and return refined scores.

    The model learns non-linear feature interactions from the
    pseudo-labeled scored candidates and produces a refined score
    that is normalized to [0, 1].

    Parameters
    ----------
    scored_candidates : list[dict]
        Scored candidate list from rank.py.

    Returns
    -------
    dict[str, float] | None
        Mapping of candidate_id → refined_score (float in [0, 1]).
        Returns None if lightgbm or numpy is not installed, or if
        there are too few candidates to train on.
    """
    if not _HAS_LGBM or not _HAS_NUMPY:
        return None

    if len(scored_candidates) < 20:
        # Not enough data to justify a model
        return None

    # ── 1. Prepare features and pseudo-labels ──
    X, y = prepare_features(scored_candidates)
    if X is None or y is None:
        return None

    # ── 2. Train a lightweight LightGBM regressor ──
    model = lgb.LGBMRegressor(
        num_leaves=15,
        n_estimators=50,
        learning_rate=0.1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,           # suppress training logs
    )
    model.fit(X, y)

    # ── 3. Predict refined scores ──
    raw_preds = model.predict(X)

    # ── 4. Normalize predictions to [0, 1] ──
    p_min = raw_preds.min()
    p_max = raw_preds.max()
    if p_max > p_min:
        normed = (raw_preds - p_min) / (p_max - p_min)
    else:
        normed = np.full_like(raw_preds, 0.5)

    # ── 5. Build candidate_id → refined_score mapping ──
    refined = {}
    for idx, item in enumerate(scored_candidates):
        cid = item.get("candidate_id", "")
        refined[cid] = float(np.clip(normed[idx], 0.0, 1.0))

    return refined


# ──────────────────────────────────────────────
# CLI SMOKE TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("lgbm_ensemble.py — smoke test")
    print(f"  LightGBM available : {_HAS_LGBM}")
    print(f"  NumPy available    : {_HAS_NUMPY}")

    if not _HAS_LGBM:
        print("  ⚠  lightgbm not installed — ensemble will be skipped.")
        print("     Install with:  pip install lightgbm")
    else:
        # Tiny synthetic test
        dummy = [
            {
                "candidate_id": f"cand_{i}",
                "semantic_score": 0.3 + 0.005 * i,
                "skill_score": 0.2 + 0.006 * i,
                "career_score": 0.4 + 0.004 * i,
                "multiplier": 1.0,
                "final_score": 0.3 + 0.005 * i,
                "cand": {
                    "profile": {"years_of_experience": 3 + i % 10},
                    "skills": [{"name": "python"}] * (i % 8),
                    "career_history": [{"company": "Acme"}] * (i % 4),
                    "redrob_signals": {
                        "recruiter_response_rate": 0.5,
                        "notice_period_days": 30,
                    },
                },
            }
            for i in range(100)
        ]
        result = train_and_predict(dummy)
        if result:
            scores = sorted(result.values(), reverse=True)
            print(f"  Trained on {len(dummy)} synthetic candidates.")
            print(f"  Score range: [{scores[-1]:.4f}, {scores[0]:.4f}]")
            print("  ✔  Smoke test passed.")
        else:
            print("  ✘  train_and_predict returned None.")
