"""17-service microservice dependency DAG across 5 tiers."""
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

# Tier assignments
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
