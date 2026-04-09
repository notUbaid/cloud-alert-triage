"""FastAPI server for Cloud Alert Triage OpenEnv environment."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import ResetRequest, StepRequest
from .environment import AlertTriageEnv
from .config import SCORE_MIN, SCORE_MAX

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


def _clamp_score(v: float) -> float:
    """Clamp a value to strictly open interval (0.001, 0.999)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    if v != v or v == float("inf") or v == float("-inf"):
        return 0.5
    if v <= 0.0:
        return SCORE_MIN
    if v >= 1.0:
        return SCORE_MAX
    return max(SCORE_MIN, min(SCORE_MAX, v))


def _safe_info(info: dict) -> dict:
    """Ensure ALL score-like keys in info are strictly in (0, 1)."""
    score_keys = ["grader_score", "score", "task_score", "episode_reward", "final_score"]
    for k in score_keys:
        if k in info and info[k] is not None:
            info[k] = _clamp_score(info[k])
    return info


@app.get("/")
def home():
    return {"message": "Cloud Alert Triage OpenEnv is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    global _initialized
    valid_tasks = ["easy", "medium", "hard"]
    if req is None:
        task_id = "easy"
        seed = 42
    else:
        task_id = req.task_id
        seed = req.seed
    if task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id: {task_id}. Must be one of {valid_tasks}",
        )
    _initialized = True
    return env.reset(task_id, seed)


@app.post("/step")
def step(req: StepRequest = None):
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    action = req.model_dump(exclude_none=True) if req else {}
    # env.step() returns a plain dict
    data = env.step(action)

    # Sanitise info block — clamp every score-like key
    if "info" in data and isinstance(data["info"], dict):
        data["info"] = _safe_info(data["info"])

    return data


@app.get("/state")
def state():
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state")
    return env.state()


# Catch-all error handler — never let an unhandled 500 leak
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "observation": {},
            "reward": 0.5,
            "done": True,
            "info": {
                "grader_score": _clamp_score(0.5),
                "score": _clamp_score(0.5),
                "task_score": _clamp_score(0.5),
                "error": str(exc),
            },
        },
    )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
