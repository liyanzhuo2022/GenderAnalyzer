from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from gender_bias_calculation import analyze_gender_bias
from utils import get_dimension_description
app = FastAPI()


# 允许前端跨域访问（开发阶段用 * 即可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或替换为 ["http://localhost:5173"]，生产环境更安全
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体模型
class JobPostingRequest(BaseModel):
    job_posting: str

# 响应体模型
class AnalyzeResponse(BaseModel):
    friendliness_score: float
    dimension: str
    dimension_description: str #新增字段
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
        result["dimension_description"] = get_dimension_description(result["dimension"])
        return result
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )
