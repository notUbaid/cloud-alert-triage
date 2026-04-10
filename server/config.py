"""Enums, constants, and cascade configuration."""
from enum import Enum
from typing import Dict, List

__all__ = [
    "RootCause", "Severity", "Remediation",
    "SEVERITY_ORDER", "severity_distance",
    "ROOT_CAUSE_REMEDIATION",
    "CASCADE_TRIGGER_STEP", "CASCADE_SECOND_WAVE_STEP", "CASCADE_SEVERITIES",
    "TASK_CONFIGS", "GRADER_WEIGHTS",
    "SCORE_MIN", "SCORE_MAX",
]

# ── Root cause categories ──
class RootCause(str, Enum):
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    DEPLOYMENT_BUG = "deployment_bug"
    CONFIG_ERROR = "config_error"
    DEPENDENCY_OUTAGE = "dependency_outage"
    FALSE_ALARM = "false_alarm"

# ── Severity levels (ordered) ──
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

SEVERITY_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]

def severity_distance(a: Severity, b: Severity) -> int:
    return abs(SEVERITY_ORDER.index(a) - SEVERITY_ORDER.index(b))

# ── Remediation actions ──
class Remediation(str, Enum):
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    ROLLBACK_DEPLOY = "rollback_deploy"
    FIX_CONFIG = "fix_config"
    ESCALATE_TO_TEAM = "escalate_to_team"
    ACKNOWLEDGE_AND_MONITOR = "acknowledge_and_monitor"
    DISMISS = "dismiss"

# ── Root cause → recommended remediation mapping ──
ROOT_CAUSE_REMEDIATION: Dict[RootCause, Remediation] = {
    RootCause.RESOURCE_EXHAUSTION: Remediation.SCALE_UP,
    RootCause.NETWORK_FAILURE: Remediation.RESTART_SERVICE,
    RootCause.DEPLOYMENT_BUG: Remediation.ROLLBACK_DEPLOY,
    RootCause.CONFIG_ERROR: Remediation.FIX_CONFIG,
    RootCause.DEPENDENCY_OUTAGE: Remediation.ESCALATE_TO_TEAM,
    RootCause.FALSE_ALARM: Remediation.DISMISS,
}

# ── Cascade config ──
CASCADE_TRIGGER_STEP = 3  # After step 3, untriaged alerts start cascading
CASCADE_SECOND_WAVE_STEP = 8  # Second wave at step 8 for more aggression
CASCADE_SEVERITIES = {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}  # MEDIUM also cascades now

# ── Task definitions ──
TASK_CONFIGS = {
    "easy": {
        "alert_count": 5,
        "max_steps": 10,
        "incident_count": 0,
        "false_alarm_count": 0,
        "cascade_enabled": False,
        "stealth_enabled": False,
    },
    "medium": {
        "alert_count": 15,
        "max_steps": 25,
        "incident_count": 2,
        "false_alarm_count": 2,
        "cascade_enabled": False,
        "stealth_enabled": False,
    },
    "hard": {
        "alert_count": 30,
        "max_steps": 45,
        "incident_count": 5,
        "false_alarm_count": 6,
        "cascade_enabled": True,
        "stealth_enabled": True,
    },
}

# ── Grader component weights per task ──
GRADER_WEIGHTS = {
    "easy": {
        "root_cause_accuracy": 0.40,
        "severity_accuracy": 0.30,
        "remediation_accuracy": 0.30,
        "incident_link_f1": 0.0,
        "false_alarm_accuracy": 0.0,
    },
    "medium": {
        "root_cause_accuracy": 0.30,
        "severity_accuracy": 0.20,
        "remediation_accuracy": 0.20,
        "incident_link_f1": 0.20,
        "false_alarm_accuracy": 0.10,
    },
    "hard": {
        "root_cause_accuracy": 0.25,
        "severity_accuracy": 0.20,
        "remediation_accuracy": 0.15,
        "incident_link_f1": 0.25,
        "false_alarm_accuracy": 0.10,
    },
}

# Score clamping constants — CRITICAL: scores must be strictly in (0, 1)
SCORE_MIN = 0.001
SCORE_MAX = 0.999
