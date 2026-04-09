"""Per-step reward calculation."""
from typing import Any, Dict, List, Optional, Set

from .config import Severity, severity_distance


def compute_triage_reward(
    agent_root_cause: str,
    agent_severity: str,
    agent_remediation: str,
    true_root_cause: str,
    true_severity: str,
    true_remediation: str,
    alert_id: str,
    linked_incidents: Dict[str, Set[str]],
    true_incidents: Dict[str, List[str]],
) -> float:
    """Compute reward for a triage action."""
    reward = 0.0
    
    # Root cause match: +0.30
    if agent_root_cause == true_root_cause:
        reward += 0.30
    
    # Severity match: +0.30 exact, +0.15 within 1 level
    try:
        agent_sev = Severity(agent_severity)
        true_sev = Severity(true_severity)
        dist = severity_distance(agent_sev, true_sev)
        if dist == 0:
            reward += 0.30
        elif dist == 1:
            reward += 0.15
    except ValueError:
        pass
    
    # Remediation match: +0.20
    if agent_remediation == true_remediation:
        reward += 0.20
    
    # Incident link bonus: +0.10 if alert was correctly linked
    for label, agent_group in linked_incidents.items():
        if alert_id in agent_group:
            # Check if this agent group matches any true incident
            for true_label, true_ids in true_incidents.items():
                true_set = set(true_ids)
                overlap = agent_group & true_set
                if len(overlap) >= 2 and alert_id in overlap:
                    reward += 0.10
                    break
            break
    
    return reward


def compute_link_reward(
    agent_alert_ids: List[str],
    true_incidents: Dict[str, List[str]],
) -> float:
    """Compute reward for a link_alerts action."""
    reward = 0.0
    agent_set = set(agent_alert_ids)
    
    # Generate all pairs from agent grouping
    agent_pairs = set()
    ids = sorted(agent_alert_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            agent_pairs.add((ids[i], ids[j]))
    
    # Generate all true pairs
    true_pairs = set()
    for label, true_ids in true_incidents.items():
        sids = sorted(true_ids)
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                true_pairs.add((sids[i], sids[j]))
    
    for pair in agent_pairs:
        if pair in true_pairs:
            reward += 0.15  # correct pair
        else:
            reward -= 0.10  # incorrect pair
    
    return reward


def compute_skip_reward(is_true_false_alarm: bool) -> float:
    """Compute reward for a skip action."""
    return 0.20 if is_true_false_alarm else -0.30


def compute_step_penalty(step: int, max_steps: int) -> float:
    """Budget pressure penalty."""
    if step >= int(max_steps * 0.8):
        return -0.05
    return 0.0
