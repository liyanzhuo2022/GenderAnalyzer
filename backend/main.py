from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
from gender_bias_calculation import analyze_gender_bias

app = FastAPI()

# 请求体模型
class JobPostingRequest(BaseModel):
    job_posting: str

# 响应体模型
class AnalyzeResponse(BaseModel):
    friendliness_score: float
    dimension: str
    gender_word_distribution: Dict[str, int]
    detected_gender_words: Dict[str, list]
    additional_metrics: Dict
    rewritten_posting: str

@app.post("/analyze_job_posting", response_model=AnalyzeResponse)
async def analyze_job_posting(request: JobPostingRequest):
    text = request.job_posting.strip()
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Please provide a non-empty job_posting string."
        )
    
    try:
        result = analyze_gender_bias(text)
        return result
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )
