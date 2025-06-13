from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from backend.gender_bias_calculation import analyze_gender_bias
from backend.utils import get_dimension_description
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 挂载 frontend 文件夹为静态资源路径
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

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
    #additional_metrics: Dict
    rewritten_posting: str

# 默认首页：返回你写好的 HTML 页面
@app.get("/")
def read_index():
    return FileResponse(f"{frontend_path}/New-Ad-Tool-VueAPI.html")

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
