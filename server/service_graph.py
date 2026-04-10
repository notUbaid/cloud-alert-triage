"""17-service microservice dependency DAG across 5 tiers.

Models a realistic e-commerce/SaaS production environment with
typed dependencies, ownership, SLA tiers, and operational metadata.
"""
from typing import Dict, List

# service_name → list of services it DEPENDS ON (downstream)
SERVICE_MAP: Dict[str, List[str]] = {
    # Tier 1 – Client
    "web-frontend": ["api-gateway"],
    # Tier 2 – Gateway
    "api-gateway": ["auth-service", "user-service", "order-service", "search-service", "notification-service"],
    # Tier 3 – Core APIs
    "auth-service": ["redis-cache", "postgres-primary"],
    "user-service": ["postgres-primary", "redis-cache"],
    "order-service": ["postgres-primary", "payment-gateway", "inventory-service", "kafka-broker"],
    "search-service": ["elasticsearch", "redis-cache"],
    "notification-service": ["email-worker", "sms-worker", "kafka-broker"],
    # Tier 4 – Workers
    "payment-gateway": ["postgres-primary"],
    "inventory-service": ["postgres-primary", "redis-cache"],
    "recommendation-engine": ["elasticsearch", "redis-cache"],
    "email-worker": ["kafka-broker"],
    "sms-worker": ["kafka-broker"],
    # Tier 5 – Data Layer (leaf nodes)
    "postgres-primary": [],
    "redis-cache": [],
    "kafka-broker": [],
    "elasticsearch": [],
    "object-storage": [],
}

ALL_SERVICES = sorted(SERVICE_MAP.keys())

# ── Tier assignments ──
TIERS: Dict[str, int] = {
    "web-frontend": 1,
    "api-gateway": 2,
    "auth-service": 3, "user-service": 3, "order-service": 3,
    "search-service": 3, "notification-service": 3,
    "payment-gateway": 4, "inventory-service": 4,
    "recommendation-engine": 4, "email-worker": 4, "sms-worker": 4,
    "postgres-primary": 5, "redis-cache": 5, "kafka-broker": 5,
    "elasticsearch": 5, "object-storage": 5,
}

# ── Operational metadata — makes scenarios feel real ──
SERVICE_META: Dict[str, Dict] = {
    "web-frontend":          {"team": "frontend-platform",  "sla": "99.95%", "oncall": "frontend-oncall",    "deploy_freq": "daily"},
    "api-gateway":           {"team": "platform-infra",     "sla": "99.99%", "oncall": "gateway-oncall",     "deploy_freq": "weekly"},
    "auth-service":          {"team": "identity",           "sla": "99.99%", "oncall": "auth-oncall",        "deploy_freq": "biweekly"},
    "user-service":          {"team": "user-platform",      "sla": "99.95%", "oncall": "user-oncall",        "deploy_freq": "weekly"},
    "order-service":         {"team": "commerce",           "sla": "99.99%", "oncall": "commerce-oncall",    "deploy_freq": "weekly"},
    "search-service":        {"team": "discovery",          "sla": "99.90%", "oncall": "search-oncall",      "deploy_freq": "daily"},
    "notification-service":  {"team": "messaging",          "sla": "99.90%", "oncall": "notif-oncall",       "deploy_freq": "weekly"},
    "payment-gateway":       {"team": "payments",           "sla": "99.99%", "oncall": "payments-oncall",    "deploy_freq": "biweekly"},
    "inventory-service":     {"team": "commerce",           "sla": "99.95%", "oncall": "commerce-oncall",    "deploy_freq": "weekly"},
    "recommendation-engine": {"team": "ml-platform",       "sla": "99.90%", "oncall": "ml-oncall",          "deploy_freq": "weekly"},
    "email-worker":          {"team": "messaging",          "sla": "99.90%", "oncall": "notif-oncall",       "deploy_freq": "weekly"},
    "sms-worker":            {"team": "messaging",          "sla": "99.90%", "oncall": "notif-oncall",       "deploy_freq": "weekly"},
    "postgres-primary":      {"team": "data-infra",         "sla": "99.999%","oncall": "dba-oncall",         "deploy_freq": "monthly"},
    "redis-cache":           {"team": "data-infra",         "sla": "99.99%", "oncall": "cache-oncall",       "deploy_freq": "monthly"},
    "kafka-broker":          {"team": "data-infra",         "sla": "99.99%", "oncall": "streaming-oncall",   "deploy_freq": "monthly"},
    "elasticsearch":         {"team": "data-infra",         "sla": "99.95%", "oncall": "search-infra-oncall","deploy_freq": "monthly"},
    "object-storage":        {"team": "data-infra",         "sla": "99.999%","oncall": "storage-oncall",     "deploy_freq": "quarterly"},
}


def get_dependents(service: str) -> List[str]:
    """Return services that depend ON the given service (upstream propagation)."""
    return sorted([s for s, deps in SERVICE_MAP.items() if service in deps])


def get_all_upstream(service: str) -> List[str]:
    """BFS to find all services transitively depending on the given service."""
    visited = set()
    queue = [service]
    while queue:
        current = queue.pop(0)
        for dependent in get_dependents(current):
            if dependent not in visited:
                visited.add(dependent)
                queue.append(dependent)
    return sorted(visited)
