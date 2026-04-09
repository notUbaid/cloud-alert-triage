"""End-of-episode deterministic grader.

CRITICAL: All scores are clamped to the strictly open interval (0.001, 0.999).
Never returns exactly 0.0 or 1.0.
"""
from typing import Any, Dict, List, Set

from .config import (
    GRADER_WEIGHTS, SCORE_MIN, SCORE_MAX,
    Severity, severity_distance,
)


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (SCORE_MIN, SCORE_MAX)."""
    if score <= 0.0:
        return SCORE_MIN
    if score >= 1.0:
        return SCORE_MAX
    # Also clamp values very close to boundaries
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _pair_set_f1(
    agent_incidents: Dict[str, Set[str]],
    true_incidents: Dict[str, List[str]],
) -> float:
    """Compute pair-set F1 over incident groupings."""
    if not true_incidents:
        return 1.0  # Vacuously correct
    
    # True pairs
    true_pairs = set()
    for label, ids in true_incidents.items():
        sids = sorted(ids)
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                true_pairs.add((sids[i], sids[j]))
    
    if not true_pairs:
        return 1.0
    
    # Agent pairs
    agent_pairs = set()
    for label, ids in agent_incidents.items():
        sids = sorted(ids)
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                agent_pairs.add((sids[i], sids[j]))
    
    if not agent_pairs:
        return 0.0
    
    tp = len(true_pairs & agent_pairs)
    precision = tp / len(agent_pairs) if agent_pairs else 0.0
    recall = tp / len(true_pairs) if true_pairs else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_grader_score(
    task_id: str,
    ground_truth: Dict[str, Dict],
    agent_decisions: Dict[str, Dict],
    agent_incidents: Dict[str, Set[str]],
    true_incidents: Dict[str, List[str]],
    skipped_alerts: Set[str],
    stealth_root_service: str = None,
    original_alert_ids: Set[str] = None,
) -> float:
    """Compute the final grader score for an episode.
    
    Returns a score in the strictly open interval (0.001, 0.999).
    """
    weights = GRADER_WEIGHTS[task_id]
    
    # Only score original alerts (exclude cascade-spawned alerts)
    if original_alert_ids:
        scoreable_ids = original_alert_ids
    else:
        scoreable_ids = set(ground_truth.keys())
    
    total_alerts = len(scoreable_ids)
    if total_alerts == 0:
        return SCORE_MIN
    
    # Separate real alerts and false alarms
    real_alert_ids = set()
    false_alarm_ids = set()
    for aid in scoreable_ids:
        gt = ground_truth.get(aid, {})
        if gt.get("is_false_alarm", False):
            false_alarm_ids.add(aid)
        else:
            real_alert_ids.add(aid)
    
    # ── Root cause accuracy ──
    rc_correct = 0
    for aid in scoreable_ids:
        gt = ground_truth[aid]
        if aid in skipped_alerts:
            # Skipped = agent says false alarm
            if gt["is_false_alarm"]:
                rc_correct += 1
        elif aid in agent_decisions:
            dec = agent_decisions[aid]
            if dec.get("root_cause") == gt["root_cause"]:
                rc_correct += 1
        # else: untriaged = incorrect
    root_cause_acc = rc_correct / total_alerts
    
    # ── Severity accuracy ──
    sev_score_sum = 0.0
    for aid in scoreable_ids:
        gt = ground_truth[aid]
        if aid in skipped_alerts:
            if gt["is_false_alarm"]:
                sev_score_sum += 1.0  # Correct skip
            # else: wrong skip = 0
        elif aid in agent_decisions:
            dec = agent_decisions[aid]
            try:
                agent_sev = Severity(dec.get("severity", ""))
                true_sev = Severity(gt["severity"])
                dist = severity_distance(agent_sev, true_sev)
                if dist == 0:
                    sev_score_sum += 1.0
                elif dist == 1:
                    sev_score_sum += 0.15
            except ValueError:
                pass
        # else: untriaged = 0
    severity_acc = sev_score_sum / total_alerts
    
    # ── Remediation accuracy ──
    rem_correct = 0
    for aid in scoreable_ids:
        gt = ground_truth[aid]
        if aid in skipped_alerts:
            if gt["is_false_alarm"]:
                rem_correct += 1  # dismiss is correct for FA
        elif aid in agent_decisions:
            dec = agent_decisions[aid]
            if dec.get("remediation") == gt["remediation"]:
                rem_correct += 1
    remediation_acc = rem_correct / total_alerts
    
    # ── Incident link F1 ──
    incident_f1 = _pair_set_f1(agent_incidents, true_incidents)
    
    # ── False alarm accuracy ──
    # FA accuracy = correctly identified as FA (skipped) + correctly identified as real (triaged)
    if false_alarm_ids or real_alert_ids:
        fa_correct = 0
        # Correctly skipped false alarms (agent correctly identified it as FA)
        fa_correct += len(false_alarm_ids & skipped_alerts)
        # Correctly triaged real alerts (agent correctly identified it as real, not skipped)
        for aid in real_alert_ids:
            if aid in agent_decisions:
                fa_correct += 1
        fa_total = len(false_alarm_ids) + len(real_alert_ids)
        false_alarm_acc = fa_correct / fa_total if fa_total > 0 else 1.0
    else:
        false_alarm_acc = 1.0  # Vacuously correct
    
    # ── Coverage multiplier ──
    covered = (set(agent_decisions.keys()) | skipped_alerts) & scoreable_ids
    coverage_count = len(covered)
    coverage = coverage_count / total_alerts if total_alerts > 0 else 0.0
    coverage_mult = coverage ** 1.5
    
    # ── Weighted score ──
    base_score = (
        weights["root_cause_accuracy"] * root_cause_acc +
        weights["severity_accuracy"] * severity_acc +
        weights["remediation_accuracy"] * remediation_acc +
        weights["incident_link_f1"] * incident_f1 +
        weights["false_alarm_accuracy"] * false_alarm_acc
    )
    
    score = base_score * coverage_mult
    
    # ── Stealth bonus (hard only) ──
    if task_id == "hard" and stealth_root_service:
        # Check if agent correctly identified the stealth root service
        stealth_bonus_count = 0
        for aid, gt in ground_truth.items():
            if gt.get("is_stealth_root", False) and aid in agent_decisions:
                dec = agent_decisions[aid]
                if dec.get("root_cause") == gt["root_cause"]:
                    stealth_bonus_count += 1
        # Apply bonus per correct stealth root (capped at 0.05)
        score += min(stealth_bonus_count * 0.05, 0.05)
    
    # ── CRITICAL: Clamp to strictly open interval (0.001, 0.999) ──
    return _clamp(score)
