# Redrob Intelligent Candidate Discovery & Ranking System

> **Redrob Data & AI Challenge Submission**  
> Ranks top 100 candidates from a 100K pool for the *Senior AI Engineer — Founding Team* role.

---

## Quick Start

### Prerequisites
```bash
pip install sentence-transformers numpy
```

### Step 1: Pre-compute embeddings (one-time, ~3 minutes on CPU)
```bash
python precompute.py
```
This generates the following files in `./cache/`:
- `candidate_embeddings.npy` — 100K × 384 float32 NumPy array
- `candidate_ids.json` — ordered list of candidate IDs
- `jd_embedding.npy` — JD query vector (384-dim)

### Step 2: Generate honeypot list (if not already done)
```bash
python generate_honeypots_list.py
```
Writes `cache/honeypot_ids.json` with flagged candidate IDs.

### Step 3: Run the ranker
```bash
python rank.py
```
Outputs `submission.csv` in ~8 seconds.

---

## Architecture

```
candidates.jsonl (100K)
       │
       ▼
┌─────────────────────────────────────────┐
│       OFFLINE PRE-COMPUTATION           │
│  • sentence-transformer embedding       │
│  • honeypot detection (7 rules)         │
└────────────────┬────────────────────────┘
                 │  cache/ (npy + json)
                 ▼
┌─────────────────────────────────────────┐
│         MULTI-SIGNAL SCORER             │
│                                         │
│  Semantic Similarity  ── 35%            │
│  (cosine vs JD vector)                  │
│                                         │
│  Skill Depth Score    ── 35%            │
│  (coverage × proficiency × duration)    │
│                                         │
│  Career Fit Score     ── 30%            │
│  (YoE bell, company type, title check)  │
│                                         │
│  × Behavioral Multiplier               │
│    (response rate, recency, notice)     │
└────────────────┬────────────────────────┘
                 ▼
         Top 100 + reasoning
                 │
                 ▼
         submission.csv
```

---

## Scoring Formula

```
final_score = min(composite × behavioral_mult, 1.0)

composite = 0.35 × semantic_sim
          + 0.35 × skill_depth
          + 0.30 × career_fit

behavioral_mult = f(response_rate) × f(recency) × f(notice_period) × f(open_to_work)
```

### Semantic Similarity (35%)
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- JD encoded as structured query covering: embeddings retrieval, vector databases, NDCG/MRR evaluation, Python, startup mindset
- Cosine similarity, min-max normalized across corpus

### Skill Depth Score (35%)
- Matches candidate skills against ~35 core required skills
- Weighted by: proficiency (expert=1.0 → beginner=0.25) × log(duration+1) × log(endorsements+1)
- Bonus for nice-to-have skills (LoRA, learning-to-rank, NLP, open-source)

### Career Fit Score (30%)
- **YoE bell curve**: ideal 5-9 years → score 1.0; < 3 years → 0.30
- **Company type penalty**: pure services career (TCS/Infosys/Wipro etc.) → ×0.40
- **Role check**: must have engineering/scientist/architect title in history
- **Hard disqualifiers**: non-engineering current role → ×0.15; vision-only (no NLP) → ×0.30

### Behavioral Multipliers
| Signal | Effect |
|--------|--------|
| Response rate < 10% | ×0.25 |
| Response rate < 20% | ×0.45 |
| Last active > 180 days | ×0.35 |
| Last active > 90 days | ×0.60 |
| Notice period > 90 days | ×0.55 |
| Notice period ≤ 30 days | ×1.02 |
| Open to work = True | ×1.08 |
| Immediately available | ×1.05 |

---

## Honeypot Detection

7 logical consistency rules flag fraudulent/trap profiles:

| Rule | Description |
|------|-------------|
| A | Expert/Advanced skill with `duration_months == 0` |
| B | ALL skills have `duration_months == 0` |
| C | `years_of_experience == 0` but has career history |
| D | Single job duration > total profile YoE |
| E | Stated job duration vs calendar start→end date off by > 24 months |
| F | Claimed work at Sarvam AI / Krutrim before 2023 (pre-founding) |
| G | Sum of all job durations > YoE + 5 years |

Flagged candidates receive **score = 0** and are excluded from ranking.

---

## Constraints Compliance

| Constraint | Target | Actual |
|------------|--------|--------|
| Offline (no API calls) | ✅ | ✅ all local |
| CPU-only | ✅ | ✅ numpy cosine sim |
| RAM | ≤ 16 GB | ~2.5 GB |
| Wall-clock time (rank.py) | ≤ 5 min | ~8 seconds |

---

## Files

```
h2s/
├── rank.py                     # Main ranker (run this for submission)
├── precompute.py               # One-time embedding generation
├── generate_honeypots_list.py  # One-time honeypot detection
├── submission_metadata.yaml    # This submission's metadata
├── README.md                   # This file
├── submission.csv              # OUTPUT: top 100 ranked candidates
└── cache/
    ├── candidate_embeddings.npy  # 100K candidate embeddings
    ├── candidate_ids.json        # ID → index mapping
    ├── jd_embedding.npy          # JD query embedding
    └── honeypot_ids.json         # Flagged honeypot IDs
```
