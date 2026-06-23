"""
dynamic_activity.py - Dynamic Activity Scoring
================================================
Redrob Data & AI Challenge - Hot-Candidate Boost

Uses candidate platform activity signals to give a real-time
'hot-candidate' boost.  Because this is an OFFLINE challenge,
activity signals are extracted from the candidate's existing data
(redrob_signals and profile fields) rather than making API calls.

Scoring components (all contribute to a [0.0, 1.0] activity score):
  1. Last-active recency   – exponential decay from REFERENCE_DATE
  2. Profile completeness  – presence of key profile fields
  3. Engagement signals    – response rate, endorsements, skill count
  4. Open-to-work flag     – boolean boost
  5. Notice period         – shorter notice → higher activity signal

The activity score is then mapped to a multiplicative boost factor
(0.85 – 1.15) via `compute_hot_candidate_boost`.

Usage:
    from dynamic_activity import compute_hot_candidate_boost
    boost = compute_hot_candidate_boost(candidate)
    final_score = base_score * boost
"""

import math
from datetime import datetime

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
REFERENCE_DATE = datetime(2026, 6, 1)

# Component weights (must sum to 1.0)
W_RECENCY        = 0.30
W_COMPLETENESS   = 0.20
W_ENGAGEMENT     = 0.25
W_OPEN_TO_WORK   = 0.10
W_NOTICE_PERIOD  = 0.15


# ──────────────────────────────────────────────
# HELPER — DATE PARSING
# ──────────────────────────────────────────────
def parse_date(s: str):
    """Parse a date string trying common formats.

    Supported formats: %Y-%m-%d, %Y-%m, %Y.
    Returns a datetime on success, None otherwise.
    """
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


# ──────────────────────────────────────────────
# SUB-SCORERS
# ──────────────────────────────────────────────
def _recency_score(candidate: dict) -> float:
    """Exponential-decay score based on last-active date.

    More recent activity → higher score.  Uses a half-life of
    180 days (~6 months): activity from 6 months ago gets 0.5,
    12 months ago gets 0.25, etc.

    Falls back to 0.0 if no date is found.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}

    raw = (
        signals.get("last_active_date")
        or profile.get("last_active_date")
    )
    dt = parse_date(raw) if isinstance(raw, str) else None
    if dt is None:
        return 0.0

    days_ago = (REFERENCE_DATE - dt).days
    if days_ago <= 0:
        return 1.0

    half_life = 180  # days
    decay = math.exp(-0.693 * days_ago / half_life)  # ln(2) ≈ 0.693
    return max(0.0, min(1.0, decay))


def _completeness_score(candidate: dict) -> float:
    """Score based on how many key profile fields are present.

    Checks: summary, headline, current_title, skills (count ≥ 1),
    career_history (count ≥ 1).  Each present field contributes
    equally (0.2 per field, max 1.0).
    """
    profile = candidate.get("profile", {}) or {}
    skills  = candidate.get("skills", []) or []
    career  = candidate.get("career_history", []) or []

    present = 0
    if profile.get("summary"):
        present += 1
    if profile.get("headline"):
        present += 1
    if profile.get("current_title"):
        present += 1
    if len(skills) >= 1:
        present += 1
    if len(career) >= 1:
        present += 1

    return present / 5.0


def _engagement_score(candidate: dict) -> float:
    """Score derived from engagement / responsiveness signals.

    Components (equally weighted thirds):
      - recruiter_response_rate  (0.0-1.0, from redrob_signals)
      - endorsement density      (total endorsements / 50, capped at 1.0)
      - skill breadth            (num_skills / 20, capped at 1.0)
    """
    signals = candidate.get("redrob_signals", {}) or {}
    skills  = candidate.get("skills", []) or []

    # Response rate (already 0–1 in most data)
    response_rate = float(signals.get("recruiter_response_rate", 0) or 0)
    response_rate = max(0.0, min(1.0, response_rate))

    # Total endorsements across all skills
    total_endorsements = sum(
        int(s.get("endorsements", 0) or 0) for s in skills
    )
    endorsement_score = min(1.0, total_endorsements / 50.0)

    # Skill breadth
    skill_count = len(skills)
    breadth_score = min(1.0, skill_count / 20.0)

    return (response_rate + endorsement_score + breadth_score) / 3.0


def _open_to_work_score(candidate: dict) -> float:
    """Returns 1.0 if the candidate has flagged open-to-work, else 0.0."""
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}

    flag = (
        signals.get("open_to_work")
        or profile.get("open_to_work")
    )
    if isinstance(flag, bool):
        return 1.0 if flag else 0.0
    if isinstance(flag, str):
        return 1.0 if flag.lower() in ("true", "yes", "1") else 0.0
    return 0.0


def _notice_period_score(candidate: dict) -> float:
    """Shorter notice period → higher activity signal.

    Mapping (days):
      ≤ 0 (immediate) → 1.0
      ≤ 15            → 0.9
      ≤ 30            → 0.75
      ≤ 60            → 0.5
      ≤ 90            → 0.3
      > 90            → 0.1

    Returns 0.5 (neutral) when notice period is unknown.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}

    raw = (
        signals.get("notice_period_days")
        or profile.get("notice_period_days")
    )
    if raw is None:
        return 0.5  # unknown → neutral

    try:
        days = int(raw)
    except (ValueError, TypeError):
        return 0.5

    if days <= 0:
        return 1.0
    if days <= 15:
        return 0.9
    if days <= 30:
        return 0.75
    if days <= 60:
        return 0.5
    if days <= 90:
        return 0.3
    return 0.1


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def compute_activity_score(candidate: dict) -> float:
    """Compute an aggregate activity score in [0.0, 1.0].

    Combines five sub-scores with fixed weights:
        recency        30 %
        completeness   20 %
        engagement     25 %
        open-to-work   10 %
        notice period  15 %

    Parameters
    ----------
    candidate : dict
        Full candidate record (must contain at least 'profile',
        'skills', 'career_history', and optionally 'redrob_signals').

    Returns
    -------
    float
        Activity score clamped to [0.0, 1.0].
    """
    score = (
        W_RECENCY       * _recency_score(candidate)
        + W_COMPLETENESS  * _completeness_score(candidate)
        + W_ENGAGEMENT    * _engagement_score(candidate)
        + W_OPEN_TO_WORK  * _open_to_work_score(candidate)
        + W_NOTICE_PERIOD * _notice_period_score(candidate)
    )
    return max(0.0, min(1.0, score))


def compute_hot_candidate_boost(candidate: dict) -> float:
    """Return a multiplicative boost factor (0.85 – 1.15).

    Mapping from activity score to boost:
        ≥ 0.8  →  1.15   (hot candidate)
        ≥ 0.6  →  1.08
        ≥ 0.4  →  1.00   (neutral)
        ≥ 0.2  →  0.92
        < 0.2  →  0.85   (cold / stale)

    Parameters
    ----------
    candidate : dict
        Full candidate record.

    Returns
    -------
    float
        Boost factor to multiply against the base ranking score.
    """
    activity = compute_activity_score(candidate)

    if activity >= 0.8:
        return 1.15
    if activity >= 0.6:
        return 1.08
    if activity >= 0.4:
        return 1.00
    if activity >= 0.2:
        return 0.92
    return 0.85
