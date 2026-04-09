"""Pydantic v2 models for OpenEnv interface."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# ── Alert model ──
class Alert(BaseModel):
    alert_id: str
    timestamp: str
    service: str
    metric: str
    metric_value: float
    threshold: float
    message: str
    context: Optional[str] = None
    triaged: bool = False
    agent_decision: Optional[Dict[str, Any]] = None

# ── Observation ──
class Observation(BaseModel):
    alerts: List[Alert]
    service_map: Dict[str, List[str]]
    pending_count: int
    step_number: int
    max_steps: int
    feedback: str = ""

# ── Actions ──
class TriageAction(BaseModel):
    action_type: Literal["triage"] = "triage"
    alert_id: str
    root_cause: str
    severity: str
    remediation: str

class LinkAlertsAction(BaseModel):
    action_type: Literal["link_alerts"] = "link_alerts"
    alert_ids: List[str]
    incident_label: str

class SkipAction(BaseModel):
    action_type: Literal["skip"] = "skip"
    alert_id: str

Action = Union[TriageAction, LinkAlertsAction, SkipAction]

# ── Step response ──
class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}

# ── Request models ──
class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42

class StepRequest(BaseModel):
    action_type: str
    alert_id: Optional[str] = None
    alert_ids: Optional[List[str]] = None
    incident_label: Optional[str] = None
    root_cause: Optional[str] = None
    severity: Optional[str] = None
    remediation: Optional[str] = None
