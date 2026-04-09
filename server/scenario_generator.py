"""Deterministic scenario generation for all tasks."""
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    TASK_CONFIGS, RootCause, Severity, Remediation,
    ROOT_CAUSE_REMEDIATION, SEVERITY_ORDER,
)
from .service_graph import SERVICE_MAP, ALL_SERVICES, get_dependents, TIERS

# ── Metric templates per root cause ──
METRIC_TEMPLATES = {
    RootCause.RESOURCE_EXHAUSTION: [
        ("cpu_usage_percent", 95.0, 80.0, "CPU usage critically high at {val:.1f}% (threshold: {thr:.1f}%)"),
        ("memory_usage_percent", 92.0, 85.0, "Memory usage at {val:.1f}%, approaching OOM (threshold: {thr:.1f}%)"),
        ("disk_io_percent", 88.0, 75.0, "Disk I/O saturated at {val:.1f}% (threshold: {thr:.1f}%)"),
    ],
    RootCause.NETWORK_FAILURE: [
        ("error_rate_percent", 12.0, 5.0, "Error rate spiked to {val:.1f}% (threshold: {thr:.1f}%)"),
        ("latency_p99_ms", 3500.0, 1000.0, "P99 latency at {val:.0f}ms, well above {thr:.0f}ms threshold"),
        ("connection_pool_exhausted", 0.0, 1.0, "Connection pool exhausted, dropping requests"),
    ],
    RootCause.DEPLOYMENT_BUG: [
        ("error_rate_percent", 25.0, 5.0, "Error rate surged to {val:.1f}% after recent deployment (threshold: {thr:.1f}%)"),
        ("crash_loop_count", 5.0, 1.0, "Service crash-looping: {val:.0f} restarts in 10 minutes"),
        ("http_5xx_rate", 18.0, 2.0, "HTTP 5xx rate at {val:.1f}% post-deploy (threshold: {thr:.1f}%)"),
    ],
    RootCause.CONFIG_ERROR: [
        ("failed_health_checks", 3.0, 1.0, "Health check failures: {val:.0f} consecutive failures"),
        ("config_drift_score", 0.8, 0.3, "Configuration drift detected: score {val:.2f} (threshold: {thr:.2f})"),
        ("tls_cert_days_remaining", 0.0, 7.0, "TLS certificate expired or expiring in {val:.0f} days"),
    ],
    RootCause.DEPENDENCY_OUTAGE: [
        ("upstream_error_rate", 45.0, 10.0, "Upstream dependency error rate at {val:.1f}% (threshold: {thr:.1f}%)"),
        ("dependency_timeout_rate", 30.0, 5.0, "Dependency timeouts at {val:.1f}% (threshold: {thr:.1f}%)"),
        ("circuit_breaker_open", 1.0, 0.0, "Circuit breaker OPEN for upstream dependency"),
    ],
}

FALSE_ALARM_TEMPLATES = [
    ("cpu_usage_percent", 81.0, 80.0, "CPU briefly touched {val:.1f}% (threshold: {thr:.1f}%) — likely transient spike"),
    ("latency_p99_ms", 1020.0, 1000.0, "P99 latency marginally exceeded threshold: {val:.0f}ms vs {thr:.0f}ms"),
    ("memory_usage_percent", 86.0, 85.0, "Memory at {val:.1f}%, fractionally above threshold — GC expected"),
    ("error_rate_percent", 5.2, 5.0, "Error rate at {val:.1f}%, barely above {thr:.1f}% — likely noise"),
]

STEALTH_TEMPLATE = (
    "memory_usage_percent", 87.5, 85.0,
    "Memory usage slightly elevated at {val:.1f}% (threshold: {thr:.1f}%) — possible slow leak"
)


def _make_alert(
    rng: random.Random,
    alert_id: str,
    service: str,
    root_cause: RootCause,
    severity: Severity,
    base_time: datetime,
    time_offset_minutes: int,
    context: Optional[str] = None,
    metric_override: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Create a single alert dict."""
    if metric_override:
        metric, val, thr, msg_tpl = metric_override
    else:
        templates = METRIC_TEMPLATES[root_cause]
        metric, val, thr, msg_tpl = rng.choice(sorted(templates, key=lambda t: t[0]))
    
    # Add noise to values
    val = val + rng.uniform(-2.0, 5.0)
    
    ts = base_time + timedelta(minutes=time_offset_minutes, seconds=rng.randint(0, 59))
    
    return {
        "alert_id": alert_id,
        "timestamp": ts.isoformat() + "Z",
        "service": service,
        "metric": metric,
        "metric_value": round(val, 2),
        "threshold": thr,
        "message": msg_tpl.format(val=val, thr=thr),
        "context": context,
        "triaged": False,
        "agent_decision": None,
    }


def _generate_incident_chain(
    rng: random.Random,
    root_service: str,
    root_cause: RootCause,
    chain_length: int,
    alert_id_start: int,
    base_time: datetime,
    time_offset: int,
    incident_label: str,
    is_stealth: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Generate a chain of correlated alerts forming an incident."""
    alerts = []
    ground_truth = {}
    
    severity_map = {
        0: Severity.CRITICAL,
        1: Severity.HIGH,
        2: Severity.MEDIUM,
    }
    
    # Root alert
    root_sev = Severity.MEDIUM if is_stealth else Severity.CRITICAL
    ctx = f"Root cause service in cascade group '{incident_label}'"
    if is_stealth:
        ctx = f"Subtle degradation — possible root cause for cascade group '{incident_label}'"
    
    metric_ov = STEALTH_TEMPLATE if is_stealth else None
    
    alert = _make_alert(
        rng, f"alert-{alert_id_start:03d}", root_service, root_cause,
        root_sev, base_time, time_offset, context=ctx, metric_override=metric_ov,
    )
    alerts.append(alert)
    ground_truth[alert["alert_id"]] = {
        "root_cause": root_cause.value,
        "severity": root_sev.value,
        "remediation": ROOT_CAUSE_REMEDIATION[root_cause].value,
        "is_false_alarm": False,
        "incident_label": incident_label,
        "is_stealth_root": is_stealth,
    }
    
    # Dependent alerts propagating upstream
    dependents = get_dependents(root_service)
    if not dependents:
        # If root is a leaf, pick services that depend on it indirectly
        # by checking all services
        dependents = [s for s in sorted(SERVICE_MAP.keys()) if root_service in SERVICE_MAP.get(s, [])]
    
    selected = sorted(dependents)
    if len(selected) > chain_length - 1:
        selected = sorted(rng.sample(selected, chain_length - 1))
    
    for i, dep_service in enumerate(selected):
        dep_id = alert_id_start + i + 1
        dep_sev = severity_map.get(i, Severity.HIGH)
        # Dependents show dependency_outage since their upstream is failing
        dep_cause = RootCause.DEPENDENCY_OUTAGE
        dep_ctx = f"Downstream of {root_service} in cascade group '{incident_label}'"
        
        dep_alert = _make_alert(
            rng, f"alert-{dep_id:03d}", dep_service, dep_cause,
            dep_sev, base_time, time_offset + i + 1, context=dep_ctx,
        )
        alerts.append(dep_alert)
        ground_truth[dep_alert["alert_id"]] = {
            "root_cause": RootCause.DEPENDENCY_OUTAGE.value,
            "severity": dep_sev.value,
            "remediation": ROOT_CAUSE_REMEDIATION[RootCause.DEPENDENCY_OUTAGE].value,
            "is_false_alarm": False,
            "incident_label": incident_label,
            "is_stealth_root": False,
        }
    
    return alerts, ground_truth


def generate_scenario(task_id: str, seed: int) -> Dict[str, Any]:
    """Generate a complete deterministic scenario for a task.
    
    Returns dict with keys: alerts, ground_truth, incidents, task_config.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    cfg = TASK_CONFIGS[task_id]
    rng = random.Random(seed)
    base_time = datetime(2026, 4, 10, 3, 0, 0)
    
    all_alerts: List[Dict] = []
    ground_truth: Dict[str, Dict] = {}
    incidents: Dict[str, List[str]] = {}  # incident_label → [alert_ids]
    
    alert_counter = 1
    time_offset = 0
    
    # Available services pool
    available_services = sorted(list(ALL_SERVICES))
    
    if task_id == "easy":
        # 5 independent alerts, one per root cause type (excluding false_alarm)
        causes = sorted([
            RootCause.RESOURCE_EXHAUSTION,
            RootCause.NETWORK_FAILURE,
            RootCause.DEPLOYMENT_BUG,
            RootCause.CONFIG_ERROR,
            RootCause.DEPENDENCY_OUTAGE,
        ], key=lambda c: c.value)
        
        services = sorted(rng.sample(available_services, 5))
        severities = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        
        for i, (cause, service) in enumerate(zip(causes, services)):
            alert = _make_alert(
                rng, f"alert-{alert_counter:03d}", service, cause,
                severities[i], base_time, time_offset + i * 2,
            )
            all_alerts.append(alert)
            ground_truth[alert["alert_id"]] = {
                "root_cause": cause.value,
                "severity": severities[i].value,
                "remediation": ROOT_CAUSE_REMEDIATION[cause].value,
                "is_false_alarm": False,
                "incident_label": None,
                "is_stealth_root": False,
            }
            alert_counter += 1
    
    elif task_id == "medium":
        # 2 incidents (chain of 3-4 alerts each)
        data_layer = sorted(["redis-cache", "postgres-primary"])
        incident_roots = sorted(rng.sample(data_layer, min(2, len(data_layer))))
        incident_causes = sorted([RootCause.RESOURCE_EXHAUSTION, RootCause.NETWORK_FAILURE], key=lambda c: c.value)
        
        for i, (root_svc, cause) in enumerate(zip(incident_roots, incident_causes)):
            label = f"incident-{i+1}"
            chain_len = 3 + rng.randint(0, 1)
            chain_alerts, chain_gt = _generate_incident_chain(
                rng, root_svc, cause, chain_len, alert_counter,
                base_time, time_offset, label,
            )
            all_alerts.extend(chain_alerts)
            ground_truth.update(chain_gt)
            incidents[label] = [a["alert_id"] for a in chain_alerts]
            alert_counter += len(chain_alerts)
            time_offset += chain_len + 2
        
        # Fill remaining with independent alerts
        used_services = {a["service"] for a in all_alerts}
        remaining_services = sorted([s for s in available_services if s not in used_services])
        independent_causes = sorted([
            RootCause.DEPLOYMENT_BUG, RootCause.CONFIG_ERROR,
            RootCause.RESOURCE_EXHAUSTION, RootCause.DEPENDENCY_OUTAGE,
        ], key=lambda c: c.value)
        
        while len(all_alerts) < cfg["alert_count"] - cfg["false_alarm_count"]:
            if remaining_services:
                svc = remaining_services.pop(0)
            else:
                svc = rng.choice(sorted(available_services))
            cause = rng.choice(sorted(independent_causes, key=lambda c: c.value))
            sev = rng.choice(sorted(SEVERITY_ORDER, key=lambda s: s.value))
            
            alert = _make_alert(
                rng, f"alert-{alert_counter:03d}", svc, cause,
                sev, base_time, time_offset,
            )
            all_alerts.append(alert)
            ground_truth[alert["alert_id"]] = {
                "root_cause": cause.value,
                "severity": sev.value,
                "remediation": ROOT_CAUSE_REMEDIATION[cause].value,
                "is_false_alarm": False,
                "incident_label": None,
                "is_stealth_root": False,
            }
            alert_counter += 1
            time_offset += 1
        
        # 2 false alarms
        fa_services = sorted(rng.sample(sorted([s for s in available_services]), min(cfg["false_alarm_count"], len(available_services))))
        for i, svc in enumerate(fa_services):
            tpl = sorted(FALSE_ALARM_TEMPLATES, key=lambda t: t[0])[i % len(FALSE_ALARM_TEMPLATES)]
            alert = _make_alert(
                rng, f"alert-{alert_counter:03d}", svc,
                RootCause.FALSE_ALARM, Severity.LOW,
                base_time, time_offset, metric_override=tpl,
            )
            all_alerts.append(alert)
            ground_truth[alert["alert_id"]] = {
                "root_cause": RootCause.FALSE_ALARM.value,
                "severity": Severity.LOW.value,
                "remediation": Remediation.DISMISS.value,
                "is_false_alarm": True,
                "incident_label": None,
                "is_stealth_root": False,
            }
            alert_counter += 1
            time_offset += 1
    
    elif task_id == "hard":
        # 5 incidents with 3-5 alerts each, one stealth
        data_layer = sorted(["redis-cache", "postgres-primary", "kafka-broker", "elasticsearch", "object-storage"])
        incident_roots = sorted(rng.sample(data_layer, 5))
        incident_causes = sorted([
            RootCause.RESOURCE_EXHAUSTION,
            RootCause.NETWORK_FAILURE,
            RootCause.DEPLOYMENT_BUG,
            RootCause.CONFIG_ERROR,
            RootCause.RESOURCE_EXHAUSTION,
        ], key=lambda c: c.value)
        
        stealth_idx = 0  # First incident is stealth (redis-cache typically)
        
        for i, (root_svc, cause) in enumerate(zip(incident_roots, incident_causes)):
            label = f"incident-{i+1}"
            chain_len = 3 + rng.randint(0, 2)
            is_stealth = (i == stealth_idx)
            
            chain_alerts, chain_gt = _generate_incident_chain(
                rng, root_svc, cause, chain_len, alert_counter,
                base_time, time_offset, label, is_stealth=is_stealth,
            )
            all_alerts.extend(chain_alerts)
            ground_truth.update(chain_gt)
            incidents[label] = [a["alert_id"] for a in chain_alerts]
            alert_counter += len(chain_alerts)
            time_offset += chain_len + 1
        
        # Fill with independent alerts
        used_services = {a["service"] for a in all_alerts}
        remaining_services = sorted([s for s in available_services if s not in used_services])
        independent_causes = sorted([
            RootCause.DEPLOYMENT_BUG, RootCause.CONFIG_ERROR,
            RootCause.NETWORK_FAILURE, RootCause.RESOURCE_EXHAUSTION,
        ], key=lambda c: c.value)
        
        while len(all_alerts) < cfg["alert_count"] - cfg["false_alarm_count"]:
            if remaining_services:
                svc = remaining_services.pop(0)
            else:
                svc = rng.choice(sorted(available_services))
            cause = rng.choice(sorted(independent_causes, key=lambda c: c.value))
            sev = rng.choice(sorted(SEVERITY_ORDER, key=lambda s: s.value))
            
            alert = _make_alert(
                rng, f"alert-{alert_counter:03d}", svc, cause,
                sev, base_time, time_offset,
            )
            all_alerts.append(alert)
            ground_truth[alert["alert_id"]] = {
                "root_cause": cause.value,
                "severity": sev.value,
                "remediation": ROOT_CAUSE_REMEDIATION[cause].value,
                "is_false_alarm": False,
                "incident_label": None,
                "is_stealth_root": False,
            }
            alert_counter += 1
            time_offset += 1
        
        # 6 false alarms — one mislabeled CRITICAL
        fa_services = sorted(rng.sample(sorted(available_services), min(cfg["false_alarm_count"], len(available_services))))
        for i, svc in enumerate(fa_services):
            tpl = sorted(FALSE_ALARM_TEMPLATES, key=lambda t: t[0])[i % len(FALSE_ALARM_TEMPLATES)]
            # First false alarm is mislabeled CRITICAL
            fa_sev = Severity.CRITICAL if i == 0 else Severity.LOW
            
            alert = _make_alert(
                rng, f"alert-{alert_counter:03d}", svc,
                RootCause.FALSE_ALARM, fa_sev,
                base_time, time_offset, metric_override=tpl,
            )
            # For the mislabeled critical FA, add misleading context
            if i == 0:
                alert["context"] = "CRITICAL severity assigned by monitoring system — verify before dismissing"
            
            all_alerts.append(alert)
            ground_truth[alert["alert_id"]] = {
                "root_cause": RootCause.FALSE_ALARM.value,
                "severity": Severity.LOW.value,  # True severity is always LOW for FA
                "remediation": Remediation.DISMISS.value,
                "is_false_alarm": True,
                "incident_label": None,
                "is_stealth_root": False,
            }
            alert_counter += 1
            time_offset += 1
    
    # Sort alerts by timestamp for temporal interleaving
    all_alerts.sort(key=lambda a: a["timestamp"])
    
    return {
        "alerts": all_alerts,
        "ground_truth": ground_truth,
        "incidents": incidents,
        "task_config": cfg,
        "stealth_root_service": incident_roots[stealth_idx] if task_id == "hard" else None,
    }
