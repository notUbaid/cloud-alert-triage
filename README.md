---
title: CloudAlert Triage AI
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Cloud Alert Triage — OpenEnv RL Environment

**Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026 · Better Call Coders**

An SRE alert-triage environment where an AI agent must classify, correlate, and remediate cloud infrastructure alerts across a realistic **17-service microservice dependency graph** — under time pressure, with injected noise, stealth failures, and a live cascade mechanic that punishes delay.

---

## The Problem We're Solving

Infrastructure alert fatigue is one of the most expensive unsolved problems in modern engineering.

- **$5,600/minute** — average cost of unplanned downtime (Gartner, 2024)
- **30–40%** of critical alerts are missed or misclassified by on-call engineers (PagerDuty / Atlassian)
- **174 alerts/week** — average volume for a mid-size engineering org; >80% are noise
- **38 minutes** — mean time to acknowledge a real incident buried in false alarms (Splunk State of Observability 2025)

Current LLMs handle isolated, obvious alerts well. What breaks them — and what this environment specifically targets — is the **stealth cascade failure**: a data-layer service silently degrading while its dependent services emit loud, misleading alarms that send naive triage agents in the wrong direction. This is exactly the failure mode that causes real outages.

**No existing OpenEnv environment models multi-step, graph-aware, real-time incident triage with cascading world state.** This environment fills that gap.

---

## What Makes This Different

| Mechanic | What It Does | Why It Matters |
|---|---|---|
| **Live cascade** | Un-triaged critical/high alerts spawn new dependent alerts after step 5 | The world state changes based on agent behavior — a genuine sequential decision problem, not static classification |
| **Stealth incident** | Root service shows subtle metric elevation while dependents fail loudly | Exposes agents that follow severity labels instead of reasoning about causality |
| **Incident linking** | Agent must group correlated alerts before triaging — scored via pair-set F1 | Rewards causal reasoning, not just per-alert pattern matching |
| **Noise discrimination** | False alarms include one mislabeled `CRITICAL` by the monitoring system | Tests whether agents blindly trust severity labels or verify against metrics |
| **Deterministic grading** | Same `(task_id, seed)` always produces identical scenario + ground truth | Fair, reproducible, directly comparable across agent implementations |
| **Dense reward signal** | Per-step rewards for each decision component + end-of-episode grader score | Agents can learn from every action, not just sparse episode-end feedback |
| **Realistic context** | Alerts include deploy diffs, runbook hints, team/oncall info, config changes | Mirrors what a real SRE sees in PagerDuty/Datadog, not synthetic toy data |

---

## Quick Start

### Docker (recommended)

```bash
docker build -t cloud-alert-triage .
docker run -p 7860:7860 cloud-alert-triage

# Verify
curl http://localhost:7860/health
# {"status": "ok"}

# List tasks
curl http://localhost:7860/tasks | python -m json.tool

# Start an episode
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","seed":42}' | python -m json.tool
```

### Local (Python 3.10+)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline agent (separate terminal)
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

---

## How an Episode Works

```
1. RESET     POST /reset {"task_id": "hard", "seed": 42}
             → Returns full observation: all alerts + service dependency map

2. PLAN      Agent analyzes dependency graph + alert metrics
             → Identifies cascade roots, groups incidents, spots false alarms

3. LINK      POST /step {"action_type": "link_alerts", "alert_ids": [...], "incident_label": "..."}
             → Groups correlated alerts into named incidents (scored via pair-set F1)

4. TRIAGE    POST /step {"action_type": "triage", "alert_id": "...", "root_cause": "...", ...}
             → Classifies root cause + severity + remediation (per-step reward)

5. SKIP      POST /step {"action_type": "skip", "alert_id": "..."}
             → Dismisses false alarms (+0.20 correct, -0.30 wrong)

6. CASCADE   After step 5, untriaged critical/high alerts spawn new dependent alerts
             → Models how real incidents escalate without intervention

7. DONE      When all alerts handled or max_steps reached
             → Grader returns deterministic score in (0.001, 0.999)
```

---

## Service Dependency Graph

17 microservices across 5 tiers, modelling a realistic e-commerce backend:

```
Tier 1 (Client)     ┌─────────────────┐
                     │  web-frontend   │
                     └────────┬────────┘
                              │
Tier 2 (Gateway)     ┌────────▼────────┐
                     │   api-gateway   │
                     └──┬──┬──┬──┬──┬──┘
                        │  │  │  │  │
Tier 3 (Core)  ┌───────┘  │  │  │  └────────┐
         ┌─────▼──┐ ┌─────▼┐ ▼ ┌▼────────┐ ┌▼───────────────┐
         │  auth  │ │ user │ │ │ search   │ │ notification   │
         │service │ │ svc  │ │ │ service  │ │ service        │
         └──┬──┬──┘ └─┬──┬┘ │ └──┬────┬──┘ └──┬─────┬───┬───┘
            │  │      │  │  │    │    │        │     │   │
Tier 4     ─┼──┼──────┼──┼──┼────┼────┼────────┼─────┼───┼────
(Workers)   │  │      │  │  │    │    │        │     │   │
            │  │      │  │ ┌▼────────────┐     │  ┌──▼┐ ┌▼──┐
            │  │      │  │ │order-service│     │  │eml│ │sms│
            │  │      │  │ └┬───┬───┬──┬─┘     │  │wkr│ │wkr│
            │  │      │  │  │   │   │  │       │  └─┬─┘ └─┬─┘
            │  │      │  │ ┌▼──┐│┌──▼┐ │  ┌────▼──┐ │     │
            │  │      │  │ │pay││││inv│ │  │recmnd │ │     │
            │  │      │  │ │gwy│││svc │ │  │engine │ │     │
            │  │      │  │ └─┬─┘│└┬──┘ │  └──┬──┬─┘ │     │
            │  │      │  │   │  │ │    │     │  │   │     │
Tier 5     ─┼──┼──────┼──┼───┼──┼─┼────┼─────┼──┼───┼─────┼──
(Data)      │  │      │  │   │  │ │    │     │  │   │     │
         ┌──▼──▼──────▼──▼───▼──┘ │    │  ┌──▼──▼┐ ┌▼─────▼┐
         │  postgres-primary  │   │    │  │redis │ │kafka  │
         └────────────────────┘   │    │  │cache │ │broker │
                            ┌─────▼┐   │  └──────┘ └───────┘
                            │ elstc│   │
                            │search│   ┌────────────┐
                            └──────┘   │obj-storage │
                                       └────────────┘
```

Failures propagate **upward** through the graph. A `redis-cache` outage surfaces as auth failures, slow search, broken recommendations — 6+ services screaming. Naive agents chase the loudest alert. Smart agents trace the dependency graph to the silent root.

---

## Tasks

| ID | Title | Alerts | Steps | Incidents | FAs | Cascade | Stealth | Score Range |
|---|---|---|---|---|---|---|---|---|
| `easy` | Basic Alert Classification | 5 | 10 | 0 | 0 | No | No | 0.85–0.999 |
| `medium` | Correlated Incident Response | 15 | 25 | 2 | 2 | No | No | 0.65–0.85 |
| `hard` | Cascading Failure Under Noise | 30 | 45 | 5 | 6 | Yes | Yes | 0.40–0.70 |

### easy — "Can the agent classify at all?"
5 independent alerts, one per root-cause type, each with unambiguous metrics and context. No incidents, no noise. Any capable agent should score ≥0.85.

### medium — "Can it reason about causality?"
15 alerts with two multi-hop cascading incidents (e.g., `postgres-primary` resource exhaustion → `auth-service` + `user-service` + `payment-gateway` dependency outages). Two false alarms with borderline metrics test noise discrimination.

### hard — "Can it think like an SRE?"
30 alerts, five cascading incidents, six false alarms (one mislabeled `CRITICAL`), and a **stealth incident** where `redis-cache` shows only subtle memory elevation while its 4 dependent services emit critical/high alerts. The cascade mechanic spawns new alerts at step 5, making delay directly costly. Alerts are temporally interleaved across incidents.

---

## Observation Space

Returned by `POST /reset` and inside every `POST /step` response.

| Field | Type | Description |
|---|---|---|
| `alerts` | `list[Alert]` | All alerts for the episode |
| `service_map` | `dict[str, list[str]]` | Dependency graph: service → its dependencies |
| `pending_count` | `int` | Un-triaged alerts remaining |
| `step_number` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Step budget for this task |
| `feedback` | `str` | Feedback from last action |

**Alert fields:** `alert_id`, `timestamp`, `service`, `metric`, `metric_value`, `threshold`, `message`, `context` (operational context: deploy info, oncall team, SLA tier, runbook hints), `triaged`, `agent_decision`.

---

## Action Space

### `triage` — classify one alert
```json
{"action_type": "triage", "alert_id": "alert-001", "root_cause": "deployment_bug", "severity": "high", "remediation": "rollback_deploy"}
```

### `link_alerts` — group correlated alerts
```json
{"action_type": "link_alerts", "alert_ids": ["alert-003", "alert-007"], "incident_label": "redis-cascade"}
```

### `skip` — dismiss a false alarm
```json
{"action_type": "skip", "alert_id": "alert-005"}
```

**Valid values:**
- `root_cause`: `resource_exhaustion` · `network_failure` · `deployment_bug` · `config_error` · `dependency_outage` · `false_alarm`
- `severity`: `critical` · `high` · `medium` · `low`
- `remediation`: `restart_service` · `scale_up` · `rollback_deploy` · `fix_config` · `escalate_to_team` · `acknowledge_and_monitor` · `dismiss`

---

## Reward Function

**Per-step rewards** provide dense learning signal:

| Action | Condition | Reward |
|---|---|---|
| `triage` | `root_cause` exact match | +0.30 |
| `triage` | `severity` exact match | +0.30 |
| `triage` | `severity` within 1 level | +0.15 |
| `triage` | `remediation` exact match | +0.20 |
| `triage` | correctly linked to incident before triaging | +0.10 bonus |
| `link_alerts` | each correct pair | +0.15 |
| `link_alerts` | each incorrect pair | −0.10 |
| `skip` | true false alarm | +0.20 |
| `skip` | real alert | −0.30 |
| any | step ≥ 80% of budget | −0.05/step |

**Design rationale:** The multi-dimensional reward ensures signal on each decision component. The budget-pressure penalty incentivises efficient ordering: link first, triage in causal order, skip false alarms early. The incident-link bonus creates a positive feedback loop — agents that reason causally before triaging are doubly rewarded.

---

## Grader

End-of-episode deterministic score in the strictly open interval **(0.001, 0.999)** — never exactly 0 or 1.

| Component | Easy | Medium | Hard |
|---|---|---|---|
| `root_cause_accuracy` | 0.40 | 0.30 | 0.25 |
| `severity_accuracy` | 0.30 | 0.20 | 0.20 |
| `remediation_accuracy` | 0.30 | 0.20 | 0.15 |
| `incident_link_f1` | — | 0.20 | 0.25 |
| `false_alarm_accuracy` | — | 0.10 | 0.10 |
| stealth bonus (hard) | — | — | +0.05 |

Coverage multiplier: `coverage^1.5` penalises agents that triage few alerts. Dynamic cascade alerts are excluded from grader scoring.

---

## Baseline Agent

**Strategy:** Plan-then-execute — a single LLM call receives all alerts + dependency graph, plans all actions, then executes deterministically.

| Task | Model | Score | Steps |
|---|---|---|---|
| easy | llama-3.3-70b-versatile | 0.999 | 5 |
| medium | llama-3.3-70b-versatile | 0.999 | 25 |
| hard | llama-3.3-70b-versatile | 0.999 | 45 |

The expected range for `hard` (0.40–0.70) describes what a reactive, alert-by-alert agent achieves without global context. The plan-then-execute baseline with full-context planning and deterministic severity/remediation inference is what a well-designed reasoning agent looks like.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check → `{"status": "ok"}` |
| `/tasks` | GET | List available tasks with metadata and valid action values |
| `/reset` | POST | Start episode → `{"task_id": "easy", "seed": 42}` |
| `/step` | POST | Submit action → returns `{observation, reward, done, info}` |
| `/state` | GET | Debug: full internal state including ground truth |

When `done=true`, `info` contains `grader_score`, `score`, `task_score` — all clamped to (0.001, 0.999).

---

## Reproducibility

All scenario generation is deterministic: `generate_scenario(task_id, seed)` uses a `random.Random(seed)` instance exclusively. Global `random` is never touched. All list operations sort inputs before sampling. Given the same `(task_id, seed)` pair, the alert set, ground truth, and grader output are byte-for-byte identical across Python versions and operating systems.

---

## Project Structure

```
cloud-alert-triage/
├── inference.py              # Baseline LLM agent (plan-then-execute)
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile
├── requirements.txt
├── server/
│   ├── app.py                # FastAPI: /health, /tasks, /reset, /step, /state
│   ├── environment.py        # Episode state machine + cascade mechanic
│   ├── scenario_generator.py # Deterministic scenario generation
│   ├── rewards.py            # Per-step reward calculation
│   ├── grading.py            # End-of-episode grader (scores ∈ (0, 1))
│   ├── service_graph.py      # 17-service dependency DAG + operational metadata
│   ├── models.py             # Pydantic v2 models (Observation, Action, StepResponse)
│   └── config.py             # Enums, weights, task configs
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   └── task_hard.json
└── tests/                    # 54 tests, all passing
    └── test_environment.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Server | FastAPI + Uvicorn |
| Data Models | Pydantic v2 |
| Container | Docker (python:3.11-slim) |
| LLM Client | OpenAI SDK (compatible) |
| Testing | pytest |
| Deployment | Hugging Face Spaces (Docker) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | **Required.** API key for LLM calls |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

---

## Contributors

**Better Call Coders** — OpenEnv Hackathon 2026

| Contributor | GitHub |
|---|---|
| Bhavesh Kumar | [@Sam-bot-dev](https://github.com/Sam-bot-dev) |
| Ubaid Khan | [@notUbaid](https://github.com/notUbaid) |
| Ved Sharma | [@Destroyerved](https://github.com/Destroyerved) |

---

## License

MIT
