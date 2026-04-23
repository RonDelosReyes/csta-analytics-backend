import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="CSTA Predictive Analytics Server")

# Supabase Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or SUPABASE_KEY == "YOUR_SERVICE_ROLE_KEY_HERE":
    print("WARNING: Supabase credentials not properly configured in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def calculate_student_metrics(user_id: int, current_score_perc: float):
    """
    Analyzes student history to create features for predictive logic.
    """
    try:
        # Fetch last 5 attempts for this user across all assessments
        response = supabase.table("tbl_assessment_attempt") \
            .select("raw_score, total_questions, created_at") \
            .eq("user_no", user_id) \
            .order("created_at", desc=True) \
            .limit(6).execute() # We take 6 to compare current with up to 5 historical ones
        
        history = response.data
        
        # If this is their first attempt ever (excluding the one we just saved)
        if len(history) <= 1:
            return {
                "avg_score": current_score_perc,
                "trend": 0.0,
                "consistency": 1.0 # Perfect consistency on first try
            }

        # Extract historical scores (excluding the very first one in the list which is the current one)
        historical_scores = [
            (h['raw_score'] / h['total_questions']) * 100 
            for h in history[1:]
        ]
        
        avg_score = np.mean(historical_scores)
        # Trend: Difference between current and previous attempt
        prev_score = historical_scores[0]
        trend = current_score_perc - prev_score
        
        # Consistency: Standard deviation (Low is better/more consistent)
        consistency = np.std(historical_scores) if len(historical_scores) > 1 else 0.0

        return {
            "avg_score": avg_score,
            "trend": trend,
            "consistency": consistency
        }
    except Exception as e:
        print(f"Metrics Error: {e}")
        return {"avg_score": current_score_perc, "trend": 0.0, "consistency": 0.0}

@app.post("/analyze/{attempt_id}")
async def analyze_attempt(attempt_id: int):
    """
    Endpoint to run predictive analytics on a specific quiz attempt.
    Also updates the long-term performance insights (Strengths/Weaknesses).
    """
    try:
        # 1. Fetch the specific attempt data joined with category info
        res = supabase.table("tbl_assessment_attempt") \
            .select("*, tbl_assessment(category_no)") \
            .eq("attempt_id", attempt_id) \
            .single().execute()
            
        if not res.data:
            raise HTTPException(status_code=404, detail="Attempt record not found")
        
        attempt_data = res.data
        user_id = attempt_data['user_no']
        raw_score = attempt_data['raw_score']
        total_q = attempt_data['total_questions']
        cat_id = attempt_data['tbl_assessment']['category_no']
        current_perc = (raw_score / total_q) * 100

        # 2. Perform Feature Engineering
        metrics = calculate_student_metrics(user_id, current_perc)
        avg = metrics["avg_score"]
        trend = metrics["trend"]

        # 3. Mastery Prediction Formula
        predictive_score = (current_perc * 0.5) + (avg * 0.3) + (trend * 0.2)

        # 4. Strength/Weakness Logic
        is_strength = predictive_score >= 80
        is_weakness = predictive_score < 60
        
        # Classification for individual attempt record
        if predictive_score >= 90:
            level = "Mastery (Exceeding Expectations)"
        elif predictive_score >= 75:
            level = "Proficient (Stable Learning)" if trend >= -15 else "Proficient (Warning: Declining)"
        elif predictive_score >= 60:
            level = "Emerging (Positive Momentum)" if trend > 5 else "Developing (Needs Consistency)"
        else:
            level = "Requires Intervention"

        # 5. Fetch First-Time Baseline for Improvement calculation
        baseline_res = supabase.table("tbl_student_analytics") \
            .select("first_score, total_questions") \
            .eq("user_no", user_id) \
            .eq("cat_id", cat_id) \
            .maybe_single().execute()
            
        first_perc = 0
        if baseline_res.data:
            first_perc = (baseline_res.data['first_score'] / baseline_res.data['total_questions']) * 100

        # 6. Upsert into tbl_performance_insight (The Student Report Data)
        supabase.table("tbl_performance_insight").upsert({
            "user_no": user_id,
            "cat_id": cat_id,
            "first_score_perc": float(first_perc),
            "current_avg_perc": float(avg),
            "improvement_perc": float(predictive_score - first_perc),
            "mastery_score": float(predictive_score),
            "mastery_level": level.split(" (")[0], # Clean name like 'Proficient'
            "is_strength": is_strength,
            "is_weakness": is_weakness,
            "last_updated": "now()"
        }).execute()

        # 7. Update original attempt record with the analytics result
        supabase.table("tbl_assessment_attempt") \
            .update({"processed_level": level}) \
            .eq("attempt_id", attempt_id).execute()

        return {
            "status": "success",
            "predicted_level": level,
            "mastery_score": f"{predictive_score:.1f}%"
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "online", "message": "CSTA Analytics API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
