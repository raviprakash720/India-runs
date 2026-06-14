import json
import os
from datetime import datetime

candidates_file = r"c:\Users\srini\OneDrive\Desktop\h2s\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
cache_dir = r"c:\Users\srini\OneDrive\Desktop\h2s\cache"
os.makedirs(cache_dir, exist_ok=True)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

honeypot_ids = set()

with open(candidates_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        cid = cand["candidate_id"]
        
        # Rule 1: Expert or Advanced skill with 0 duration_months
        skills = cand.get("skills", [])
        for s in skills:
            if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months") == 0:
                honeypot_ids.add(cid)
        
        # Rule 2: All skills on profile have 0 duration_months
        if len(skills) > 0 and all(s.get("duration_months") == 0 for s in skills):
            honeypot_ids.add(cid)
            
        # Rule 3: YoE is 0 but has career history with positive duration
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        career = cand.get("career_history", [])
        if yoe == 0 and len(career) > 0:
            if any(j.get("duration_months", 0) > 0 for j in career):
                honeypot_ids.add(cid)
                
        # Rule 4: Job duration exceeds profile years_of_experience
        for job in career:
            dur_months = job.get("duration_months", 0)
            if dur_months / 12.0 > yoe + 0.1:
                honeypot_ids.add(cid)
            
            # Rule 5: Job duration wildly exceeds actual calendar range
            start_dt = parse_date(job.get("start_date"))
            end_dt = parse_date(job.get("end_date"))
            if start_dt:
                if end_dt:
                    actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                else:
                    ref_dt = datetime(2026, 6, 1)
                    actual_months = (ref_dt.year - start_dt.year) * 12 + (ref_dt.month - start_dt.month)
                if dur_months - actual_months > 24:
                    honeypot_ids.add(cid)
                    
            # Rule 6: Startup Pre-Founding Work (Sarvam AI / Krutrim before 2023)
            comp = job.get("company")
            if comp in ["Sarvam AI", "Krutrim"] and start_dt and start_dt.year < 2023:
                honeypot_ids.add(cid)
                
        # Rule 7: Sum of job durations exceeds years of experience by > 5 years
        total_dur = sum(j.get("duration_months", 0) for j in career)
        if total_dur / 12.0 > yoe + 5.0 and yoe > 0:
            honeypot_ids.add(cid)

honeypot_list = sorted(list(honeypot_ids))
output_path = os.path.join(cache_dir, "honeypot_ids.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(honeypot_list, f)

print(f"Saved {len(honeypot_list)} honeypot IDs to {output_path}")
