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

# Core required skills (lower-cased for matching)
CORE_REQUIRED_SKILLS = {
    # retrieval / search
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "vector databases", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "hybrid search",
    # evaluation
    "ndcg", "mrr", "map", "ranking evaluation", "a/b testing",
    "evaluation framework", "information retrieval",
    # core tech
    "python", "pytorch", "transformers", "bert", "llm",
    "fine-tuning", "rag", "retrieval-augmented generation",
    "machine learning", "deep learning",
    # bonus (lower weight)
    "lora", "qlora", "peft", "learning-to-rank", "xgboost",
    "recommendation system", "search", "ranking", "nlp",
}

# "Nice to have" — get a small boost but not required
BONUS_SKILLS = {
    "lora", "qlora", "peft", "learning-to-rank", "xgboost",
    "nlp", "open source", "open-source", "distributed systems",
}

# Consulting / IT-services firms that signal a pure-services background
SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "ltimindtree",
    "l&t infotech", "hexaware", "niit technologies", "zensar",
}

REFERENCE_DATE = datetime(2026, 6, 1)

# ──────────────────────────────────────────────
# HONEYPOT ID SET  (pre-computed by generate_honeypots_list.py)
# ──────────────────────────────────────────────
def load_honeypot_ids():
    hp_path = os.path.join(CACHE_DIR, "honeypot_ids.json")
    if os.path.exists(hp_path):
        with open(hp_path, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


# ──────────────────────────────────────────────
# RUNTIME HONEYPOT DETECTION  (belt-and-suspenders)
# ──────────────────────────────────────────────
def parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def is_honeypot(cand, honeypot_set):
    cid = cand["candidate_id"]
    if cid in honeypot_set:
        return True

    profile = cand.get("profile", {})
    skills  = cand.get("skills", [])
    career  = cand.get("career_history", [])
    yoe     = profile.get("years_of_experience", 0) or 0

    # Rule A: expert/advanced skill with 0 duration
    for s in skills:
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 0) == 0:
            return True

    # Rule B: all skills have 0 duration
    if skills and all(s.get("duration_months", 0) == 0 for s in skills):
        return True

    # Rule C: YoE=0 but has career history
    if yoe == 0 and any(j.get("duration_months", 0) > 0 for j in career):
        return True

    # Rule D: single job duration > total YoE
    for job in career:
        dur = job.get("duration_months", 0) or 0
        if yoe > 0 and dur / 12.0 > yoe + 0.1:
            return True
        # Rule E: duration vs calendar dates wildly off
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date"))
        if sd:
            ref = ed if ed else REFERENCE_DATE
            actual = (ref.year - sd.year) * 12 + (ref.month - sd.month)
            if dur - actual > 24:
                return True
        # Rule F: impossible pre-founding work
        company = (job.get("company") or "").strip()
        if company in ("Sarvam AI", "Krutrim") and sd and sd.year < 2023:
            return True

    # Rule G: total career duration hugely exceeds YoE
    total = sum(j.get("duration_months", 0) or 0 for j in career)
    if yoe > 0 and total / 12.0 > yoe + 5.0:
        return True

    return False


# ──────────────────────────────────────────────
# SKILL DEPTH SCORE  (0..1)
# ──────────────────────────────────────────────
def compute_skill_score(cand):
    skills = cand.get("skills", [])
    if not skills:
        return 0.0

    matched_core  = 0.0
    matched_bonus = 0.0
    total_core    = len(CORE_REQUIRED_SKILLS)

    skill_names_lower = {(s.get("name") or "").lower() for s in skills}

    for s in skills:
        name  = (s.get("name") or "").lower()
        prof  = PROFICIENCY_MAP.get(s.get("proficiency", ""), 0.25)
        dur   = s.get("duration_months", 0) or 0
        endorse = s.get("endorsements", 0) or 0

        # Log-scale duration bonus (max 1.5x boost)
        dur_factor    = 1.0 + 0.5 * math.log1p(dur) / math.log1p(60)
        endorse_factor = 1.0 + 0.3 * math.log1p(endorse) / math.log1p(10)

        # Core skill match (fuzzy: check if any core keyword in skill name)
        is_core  = any(core in name for core in CORE_REQUIRED_SKILLS)
        is_bonus = any(bon  in name for bon  in BONUS_SKILLS)

        if is_core:
            matched_core += prof * dur_factor * endorse_factor
        if is_bonus:
            matched_bonus += prof * 0.5

    # Coverage ratio (0..1 bounded)
    core_coverage = min(matched_core / total_core, 1.0)
    bonus_term    = min(matched_bonus / max(len(BONUS_SKILLS), 1), 1.0) * 0.15

    # Skill co-occurrence bonus
    co_bonus = 0.0
    try:
        from skill_cooccurrence import compute_cooccurrence_bonus
        co_bonus = compute_cooccurrence_bonus(skills)
    except (ImportError, ModuleNotFoundError):
        pass

    return min(core_coverage + bonus_term + 0.15 * co_bonus, 1.0)


# ──────────────────────────────────────────────
# CAREER FIT SCORE  (0..1)
# ──────────────────────────────────────────────
def compute_career_score(cand):
    profile = cand.get("profile", {})
    career  = cand.get("career_history", [])
    skills  = cand.get("skills", [])
    
    yoe     = profile.get("years_of_experience", 0) or 0
    score   = 0.5  # baseline

    # 1. Years of Experience fit — ideal 5-9 years
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5:
        yoe_score = 0.85
    elif 3 <= yoe < 4:
        yoe_score = 0.65
    elif 9 < yoe <= 12:
        yoe_score = 0.90
    elif 12 < yoe <= 15:
        yoe_score = 0.75
    elif yoe > 15:
        yoe_score = 0.55
    else:
        yoe_score = 0.30   # < 3 years
    score = yoe_score

    # 2. Company quality analysis
    companies_lower = [(j.get("company") or "").lower() for j in career]
    services_count  = sum(1 for c in companies_lower if any(sf in c for sf in SERVICES_FIRMS))
    total_companies = max(len(companies_lower), 1)
    services_ratio  = services_count / total_companies

    if services_ratio >= 1.0:       # entire career at services firms
        score *= 0.40
    elif services_ratio >= 0.75:
        score *= 0.65
    elif services_ratio >= 0.50:
        score *= 0.80

    # 3. Seniority check: at least one senior/lead/engineer role
    titles_lower = [(j.get("title") or "").lower() for j in career]
    has_senior_role = any(
        any(kw in t for kw in ("engineer", "scientist", "researcher", "developer", "architect", "lead", "principal"))
        for t in titles_lower
    )
    if not has_senior_role:
        score *= 0.60

    # 4. Hard disqualifiers
    # Disqualifier A: pure academic/research (no production deployment)
    academic_markers = ("phd student", "research intern", "research scholar", "academic researcher")
    current_title    = (profile.get("current_title") or "").lower()
    if any(m in current_title for m in academic_markers) and yoe < 3:
        score *= 0.20

    # Disqualifier B: non-AI/non-engineering current role
    non_eng_markers = ("marketing", "sales", "hr ", "human resource", "operations", "project manager",
                       "scrum master", "business analyst", "product manager", "teacher", "professor")
    if any(m in current_title for m in non_eng_markers):
        score *= 0.15

    # Disqualifier C: vision/speech/robotics only (no NLP/IR)
    all_skill_names = " ".join((s.get("name") or "").lower() for s in skills)
    has_nlp_ir = any(kw in all_skill_names for kw in ("nlp", "retrieval", "embedding", "search", "ranking", "bert", "transformer", "llm"))
    has_vision = any(kw in all_skill_names for kw in ("computer vision", "image recognition", "object detection", "speech recognition", "robotics"))
    if has_vision and not has_nlp_ir:
        score *= 0.30

    return min(max(score, 0.0), 1.0)


# ──────────────────────────────────────────────
# BEHAVIORAL MODIFIERS
# ──────────────────────────────────────────────
def compute_behavioral_multiplier(cand):
    signals   = cand.get("redrob_signals", {})
    profile   = cand.get("profile", {})
    mult      = 1.0

    # Response rate — critical signal
    response_rate = signals.get("recruiter_response_rate", 1.0) or 1.0
    if response_rate < 0.10:
        mult *= 0.25
    elif response_rate < 0.20:
        mult *= 0.45
    elif response_rate < 0.35:
        mult *= 0.70
    elif response_rate < 0.50:
        mult *= 0.90

    # Last active date — recency decay
    last_active_str = signals.get("last_active_date") or profile.get("last_active_date")
    if last_active_str:
        last_active = parse_date(str(last_active_str))
        if last_active:
            days_inactive = (REFERENCE_DATE - last_active).days
            if days_inactive > 180:
                mult *= 0.35
            elif days_inactive > 90:
                mult *= 0.60
            elif days_inactive > 60:
                mult *= 0.80
            elif days_inactive > 30:
                mult *= 0.92

    # Open to work — availability signal
    open_to_work = signals.get("open_to_work_flag") or profile.get("open_to_work")
    if open_to_work:
        mult *= 1.08

    # Notice period
    notice_days = signals.get("notice_period_days") or profile.get("notice_period_days") or 0
    if notice_days is None:
        notice_days = 0
    if notice_days <= 0:
        mult *= 1.05    # immediately available
    elif notice_days <= 30:
        mult *= 1.02
    elif notice_days <= 60:
        mult *= 0.92
    elif notice_days <= 90:
        mult *= 0.80
    else:
        mult *= 0.55    # > 90 days — JD explicitly says "bar gets higher"

    # Dynamic Activity Boost
    try:
        from dynamic_activity import compute_hot_candidate_boost
        boost = compute_hot_candidate_boost(cand)
        mult *= boost
    except (ImportError, ModuleNotFoundError):
        pass

    return min(max(mult, 0.05), 1.5)


# ──────────────────────────────────────────────
# REASONING GENERATOR
# ──────────────────────────────────────────────
def generate_reasoning(cand, final_score, semantic_score, skill_score, career_score):
    profile  = cand.get("profile", {})
    career   = cand.get("career_history", [])
    skills   = cand.get("skills", [])
    signals  = cand.get("redrob_signals", {})

    name   = profile.get("name") or "Candidate"
    yoe    = profile.get("years_of_experience", 0) or 0
    title  = profile.get("current_title") or profile.get("headline") or "AI Engineer"
    
    # Best company (last / most recent)
    company = ""
    if career:
        career_sorted = sorted(
            [j for j in career if j.get("start_date")],
            key=lambda j: j.get("start_date", ""),
            reverse=True
        )
        if career_sorted:
            company = career_sorted[0].get("company") or ""
    
    # Top skills (by proficiency then duration)
    top_skills = sorted(
        skills,
        key=lambda s: (PROFICIENCY_MAP.get(s.get("proficiency", ""), 0),
                       s.get("duration_months", 0)),
        reverse=True
    )[:3]
    top_skill_names = [s.get("name", "") for s in top_skills if s.get("name")]

    response_rate = signals.get("recruiter_response_rate", 0.5) or 0.5
    rr_pct = int(response_rate * 100)

    parts = []
    if company:
        parts.append(f"{yoe}-year AI/ML engineer, most recently at {company}")
    else:
        parts.append(f"{yoe}-year AI/ML engineering background")

    if top_skill_names:
        parts.append(f"with expertise in {', '.join(top_skill_names[:2])}")

    skill_tag = ""
    if skill_score >= 0.7:
        skill_tag = "Strong match on core retrieval & ranking stack."
    elif skill_score >= 0.4:
        skill_tag = "Partial match on required ML skills."
    else:
        skill_tag = "Limited direct alignment with JD skill requirements."

    avail_tag = ""
    notice = signals.get("notice_period_days", 0) or 0
    open_w = signals.get("open_to_work_flag", False)
    if open_w and notice <= 30:
        avail_tag = "Actively looking, available quickly."
    elif open_w:
        avail_tag = f"Open to work; notice period ~{notice} days."
    elif notice <= 30:
        avail_tag = "Low notice period — readily available."

    reasoning = ". ".join(filter(None, [
        " ".join(parts),
        skill_tag,
        avail_tag,
        f"Response rate: {rr_pct}%."
    ]))
    return reasoning


# ──────────────────────────────────────────────
# SEMANTIC SIMILARITY (numpy cosine — no GPU needed)
# ──────────────────────────────────────────────
def cosine_sim_matrix(embeddings, jd_emb):
    """Compute cosine similarity of all candidates vs JD (vectorized)."""
    import numpy as np
    # Normalize rows
    norms_cand = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_cand = np.where(norms_cand == 0, 1e-8, norms_cand)
    normed_cand = embeddings / norms_cand

    norm_jd = np.linalg.norm(jd_emb)
    if norm_jd == 0:
        norm_jd = 1e-8
    normed_jd = jd_emb / norm_jd

    return normed_cand @ normed_jd   # shape (N,)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    start_time = time.time()

    print("=" * 60)
    print("Redrob Intelligent Candidate Ranker")
    print("=" * 60)

    # ── 1. Load honeypot IDs ──
    print("\n[1/6] Loading honeypot IDs...")
    honeypot_set = load_honeypot_ids()
    print(f"  Loaded {len(honeypot_set)} pre-flagged honeypot IDs.")

    # ── 2. Load embeddings ──
    print("\n[2/6] Loading pre-computed embeddings...")
    emb_path  = os.path.join(CACHE_DIR, "candidate_embeddings.npy")
    ids_path  = os.path.join(CACHE_DIR, "candidate_ids.json")
    jd_path   = os.path.join(CACHE_DIR, "jd_embedding.npy")

    use_embeddings = (
        os.path.exists(emb_path) and
        os.path.exists(ids_path) and
        os.path.exists(jd_path)
    )

    # Priority: neural embeddings (best) > 1D keyword scores (fast fallback) > neutral
    scores_1d_path = os.path.join(CACHE_DIR, "candidate_semantic_scores.npy")
    use_neural = use_embeddings and os.path.getsize(emb_path) > 1_000_000  # >1MB = real embeddings
    use_1d_scores = os.path.exists(scores_1d_path) and os.path.exists(ids_path) and not use_neural

    if use_neural:
        import numpy as np
        candidate_embeddings = np.load(emb_path)
        with open(ids_path, encoding="utf-8") as f:
            embed_id_list = json.load(f)
        jd_embedding = np.load(jd_path)
        embed_id_to_idx = {cid: i for i, cid in enumerate(embed_id_list)}
        semantic_scores = cosine_sim_matrix(candidate_embeddings, jd_embedding)
        # Min-max normalize to [0, 1]
        smin, smax = semantic_scores.min(), semantic_scores.max()
        if smax > smin:
            semantic_scores = (semantic_scores - smin) / (smax - smin)
        print(f"  Loaded NEURAL embeddings for {len(embed_id_list):,} candidates (best quality).")
    elif use_1d_scores:
        import numpy as np
        with open(ids_path, encoding="utf-8") as f:
            embed_id_list = json.load(f)
        semantic_scores = np.load(scores_1d_path)
        embed_id_to_idx = {cid: i for i, cid in enumerate(embed_id_list)}
        print(f"  Loaded keyword semantic scores for {len(embed_id_list):,} candidates.")
    else:
        print("  WARNING: No semantic cache found. Semantic score will be 0.5 (neutral).")
        print("  Run precompute.py first for best results.")
        embed_id_to_idx = {}
        semantic_scores = None

    # ── 3. Load candidates ──
    print("\n[3/6] Loading candidate profiles...")
    candidates = []
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates)} candidate profiles.")

    # ── 4. Score all candidates ──
    print("\n[4/6] Scoring candidates...")
    scored = []
    honeypot_count = 0

    for cand in candidates:
        cid = cand["candidate_id"]

        # Hard filter: honeypots get score 0
        if is_honeypot(cand, honeypot_set):
            honeypot_count += 1
            continue

        # Semantic similarity
        if semantic_scores is not None and cid in embed_id_to_idx:
            sem_score = float(semantic_scores[embed_id_to_idx[cid]])
        else:
            sem_score = 0.50   # neutral fallback

        # Skill depth
        skill_score = compute_skill_score(cand)

        # Career fit
        career_score = compute_career_score(cand)

        # Weighted composite
        composite = (
            W_SEMANTIC * sem_score +
            W_SKILL    * skill_score +
            W_CAREER   * career_score
        )

        # Behavioral multiplier
        mult = compute_behavioral_multiplier(cand)
        final_score = min(composite * mult, 1.0)

        scored.append({
            "candidate_id":   cid,
            "final_score":    final_score,
            "semantic_score": sem_score,
            "skill_score":    skill_score,
            "career_score":   career_score,
            "multiplier":     mult,
            "cand":           cand,
        })

    print(f"  Scored {len(scored)} non-honeypot candidates.")
    print(f"  Filtered out {honeypot_count} honeypot candidates.")

    # ── 5. Advanced Reranking & Ensemble ──
    print("\n[5/6] Running Advanced Reranking & Ensemble...")
    # Initial sort
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # A. Cross-Encoder Reranking
    try:
        from cross_encoder_rerank import rerank_candidates
        from precompute import build_candidate_text
        jd_text = (
            "Senior AI Engineer with experience in embeddings-based retrieval, "
            "sentence-transformers, vector databases Pinecone Weaviate Qdrant FAISS, "
            "hybrid search, Python, evaluation frameworks NDCG MRR MAP, LLM fine-tuning, "
            "RAG, information retrieval, ranking systems at product companies."
        )
        print("  Generating candidate texts for top 5,000 candidates...")
        # Only build texts for candidates in the top 5000 to keep it extremely fast
        top_5k_ids = {item["candidate_id"] for item in scored[:5000]}
        candidate_texts = {
            item["candidate_id"]: build_candidate_text(item["cand"])
            for item in scored
            if item["candidate_id"] in top_5k_ids
        }
        print("  Applying Cross-Encoder reranking...")
        scored = rerank_candidates(scored, candidate_texts, jd_text, top_k=5000)

        # Recompute final_score for top 5k using refined semantic scores
        for item in scored[:5000]:
            composite = (
                W_SEMANTIC * item["semantic_score"] +
                W_SKILL    * item["skill_score"] +
                W_CAREER   * item["career_score"]
            )
            item["final_score"] = min(composite * item["multiplier"], 1.0)
        
        # Re-sort after semantic updates
        scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    except Exception as e:
        print(f"  Warning: Cross-Encoder reranking skipped or failed: {e}")

    # B. LightGBM Ensemble Correction
    try:
        from lgbm_ensemble import train_and_predict
        print("  Training LightGBM ensemble regressor...")
        refined_scores = train_and_predict(scored)
        if refined_scores is not None:
            print("  Applying LightGBM non-linear correction to candidate scores...")
            for item in scored:
                cid = item["candidate_id"]
                if cid in refined_scores:
                    item["final_score"] = refined_scores[cid]
            scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        else:
            print("  LightGBM ensemble unavailable; using linear composite scores.")
    except Exception as e:
        print(f"  Warning: LightGBM ensemble skipped or failed: {e}")

    top100 = scored[:100]

    # Assign ranks and generate reasoning
    rows = []
    for rank, item in enumerate(top100, start=1):
        cand = item["cand"]
        reasoning = generate_reasoning(
            cand,
            item["final_score"],
            item["semantic_score"],
            item["skill_score"],
            item["career_score"],
        )
        rows.append({
            "candidate_id": item["candidate_id"],
            "rank":         rank,
            "score":        round(item["final_score"], 6),
            "reasoning":    reasoning,
        })

    # ── 6. Write submission CSV ──
    print(f"\n[6/6] Writing {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    # ── 7. Write formatted submission XLSX ──
    try:
        from export_xlsx import export_to_xlsx
        xlsx_path = OUTPUT_CSV.replace(".csv", ".xlsx")
        print(f"  Writing formatted spreadsheet to {xlsx_path}...")
        xlsx_rows = []
        for rank, item in enumerate(top100, start=1):
            cand = item["cand"]
            reasoning = generate_reasoning(
                cand,
                item["final_score"],
                item["semantic_score"],
                item["skill_score"],
                item["career_score"],
            )
            xlsx_rows.append({
                "candidate_id": item["candidate_id"],
                "rank":         rank,
                "score":        round(item["final_score"], 6),
                "semantic_score": round(item["semantic_score"], 6),
                "skill_score":    round(item["skill_score"], 6),
                "career_score":   round(item["career_score"], 6),
                "multiplier":     round(item["multiplier"], 6),
                "reasoning":    reasoning,
            })
        metadata = {
            "team_name": "India-runs",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_candidates": len(candidates)
        }
        export_to_xlsx(xlsx_rows, xlsx_path, metadata=metadata)
        print(f"  XLSX Export successful: {xlsx_path}")
    except Exception as e:
        print(f"  Warning: XLSX export skipped or failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Elapsed time: {elapsed:.1f}s")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Top candidate: {rows[0]['candidate_id']} (score={rows[0]['score']})")
    print(f"{'=' * 60}")

    # Quick sanity checks
    print("\n[Sanity Checks]")
    ids_in_top100 = {r["candidate_id"] for r in rows}
    hp_in_top100  = ids_in_top100 & honeypot_set
    print(f"  Honeypots in top 100: {len(hp_in_top100)} (limit: <=10)")
    scores_desc   = all(rows[i]["score"] >= rows[i+1]["score"] for i in range(len(rows)-1))
    print(f"  Scores descending: {scores_desc}")
    unique_ids    = len(ids_in_top100) == 100
    print(f"  All 100 IDs unique: {unique_ids}")


if __name__ == "__main__":
    main()
