"""FastAPI server for Cloud Alert Triage OpenEnv environment."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import ResetRequest, StepRequest
from .environment import AlertTriageEnv
from .config import SCORE_MIN, SCORE_MAX, TASK_CONFIGS

app = FastAPI(
    title="Cloud Alert Triage — OpenEnv Environment",
    description=(
        "SRE alert triage environment where an AI agent classifies, correlates, "
        "and remediates cloud infrastructure alerts across a 17-service "
        "microservice dependency graph with cascading failures."
    ),
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
    """Clamp to strictly open interval (0.001, 0.999)."""
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
    """Clamp ALL score-like keys in info to (0.001, 0.999)."""
    for k in ["grader_score", "score", "task_score", "episode_reward", "final_score"]:
        if k in info and info[k] is not None:
            info[k] = _clamp_score(info[k])
    return info


# ── Endpoints ──

@app.get("/")
def home():
    return {"message": "Cloud Alert Triage OpenEnv is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    """List available tasks with metadata — standard OpenEnv endpoint."""
    return {
        "tasks": [
            {
                "task_id": "easy",
                "title": "Basic Alert Classification",
                "description": "5 independent alerts from 5 different services. No incidents, no false alarms.",
                "difficulty": "easy",
                "alert_count": 5,
                "max_steps": 10,
                "expected_score_range": [0.85, 0.999],
            },
            {
                "task_id": "medium",
                "title": "Correlated Incident Response",
                "description": "15 alerts across 10 services. 2 multi-hop cascading incidents and 2 false alarms with borderline metrics.",
                "difficulty": "medium",
                "alert_count": 15,
                "max_steps": 25,
                "expected_score_range": [0.65, 0.85],
            },
            {
                "task_id": "hard",
                "title": "Cascading Failure Under Noise",
                "description": "30 alerts across 15 services. 5 cascading incidents, 6 false alarms (one mislabeled CRITICAL), one stealth incident, cascade mechanic active.",
                "difficulty": "hard",
                "alert_count": 30,
                "max_steps": 45,
                "expected_score_range": [0.40, 0.70],
            },
        ],
        "action_space": {
            "triage":      {"alert_id": "str", "root_cause": "str", "severity": "str", "remediation": "str"},
            "link_alerts": {"alert_ids": "list[str]", "incident_label": "str"},
            "skip":        {"alert_id": "str"},
        },
        "valid_values": {
            "root_cause":   ["resource_exhaustion", "network_failure", "deployment_bug", "config_error", "dependency_outage", "false_alarm"],
            "severity":     ["critical", "high", "medium", "low"],
            "remediation":  ["restart_service", "scale_up", "rollback_deploy", "fix_config", "escalate_to_team", "acknowledge_and_monitor", "dismiss"],
        },
    }


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
        raise HTTPException(status_code=422, detail=f"Unknown task_id: {task_id}. Must be one of {valid_tasks}")
    _initialized = True
    return env.reset(task_id, seed)


@app.post("/step")
def step(req: StepRequest = None):
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    action = req.model_dump(exclude_none=True) if req else {}
    data = env.step(action)
    if "info" in data and isinstance(data["info"], dict):
        data["info"] = _safe_info(data["info"])
    return data


@app.get("/state")
def state():
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state")
    return env.state()


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
