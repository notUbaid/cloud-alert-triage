"""FastAPI server for Cloud Alert Triage OpenEnv environment."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import ResetRequest, StepRequest
from .environment import AlertTriageEnv

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
    
    action = req.model_dump(exclude_none=True)
    return env.step(action)


@app.get("/state")
def state():
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state")
    return env.state()
