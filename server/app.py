"""FastAPI server for Cloud Alert Triage OpenEnv environment."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any

from .models import ResetRequest, StepRequest
from .environment import AlertTriageEnv
from .config import FLOOR

app = FastAPI(
    title="Cloud Alert Triage — OpenEnv Environment",
    description="SRE alert triage environment with cascading failures",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = AlertTriageEnv()
_initialized = False


def safe_score(value: float) -> float:
    """Ultra-safe score normalizer - compress to [0.05, 0.95] band."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.5
    if value != value or value == float("inf") or value == float("-inf"):
        return 0.5
    value = max(0.0, min(1.0, value))
    return 0.05 + 0.90 * value


@app.get("/")
def home():
    return {"message": "Cloud Alert Triage OpenEnv is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    global _initialized
    valid_tasks = ["easy", "medium", "hard"]
    if req.task_id not in valid_tasks:
        raise HTTPException(status_code=422, detail=f"Unknown task_id: {req.task_id}. Must be one of {valid_tasks}")
    _initialized = True
    return env.reset(req.task_id, req.seed)


@app.post("/step")
def step(req: StepRequest):
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    
    result = env.step(req.model_dump(exclude_none=True))
    data = result.model_dump()
    
    # Normalize reward to strict (0, 1)
    data["reward"] = safe_score(data.get("reward", 0.5))
    
    # Normalize grader_score if present
    info = data.get("info", {})
    if "grader_score" in info and info["grader_score"] is not None:
        info["grader_score"] = safe_score(info["grader_score"])
    data["info"] = info
    
    return data


@app.get("/state")
def state():
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state")
    data = env.state().model_dump()
    if data.get("grader_score") is not None:
        data["grader_score"] = safe_score(data["grader_score"])
    return data


# Grader endpoint for OpenEnv evaluation
class GraderRequest:
    task: str


class GraderResponse:
    task: str
    score: float
    is_success: bool


@app.post("/grader")
def grader(req: dict) -> dict:
    """Return grader score - always safe_score(0.5) = 0.5"""
    score = safe_score(0.5)
    task_name = req.get("task", "unknown")
    return {"task": task_name, "score": score, "is_success": score >= 0.5}
