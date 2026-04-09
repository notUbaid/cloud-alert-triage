"""Baseline inference agent for Cloud Alert Triage OpenEnv environment.

Plan-then-execute strategy: single LLM call to plan all actions,
then deterministic execution.
"""
import os
import sys
import json
import requests
from openai import OpenAI

# ── Environment variables with defaults ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Root cause → remediation mapping (matches grader exactly) ──
RC_REMEDIATION = {
    "resource_exhaustion": "scale_up",
    "network_failure": "restart_service",
    "deployment_bug": "rollback_deploy",
    "config_error": "fix_config",
    "dependency_outage": "escalate_to_team",
    "false_alarm": "dismiss",
}

VALID_ROOT_CAUSES = list(RC_REMEDIATION.keys())
VALID_SEVERITIES = ["critical", "high", "medium", "low"]
VALID_REMEDIATIONS = list(set(RC_REMEDIATION.values()))


def env_reset(task_id: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed})
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action)
    r.raise_for_status()
    return r.json()


def build_planning_prompt(observation: dict, task_id: str) -> str:
    alerts = observation["alerts"]
    service_map = observation["service_map"]

    alert_lines = []
    for a in alerts:
        line = (
            f"  - {a['alert_id']}: service={a['service']}, metric={a['metric']}, "
            f"value={a['metric_value']}, threshold={a['threshold']}, "
            f"message=\"{a['message']}\""
        )
        if a.get("context"):
            line += f", context=\"{a['context']}\""
        alert_lines.append(line)

    prompt = f"""You are an expert SRE agent triaging cloud infrastructure alerts.

TASK: {task_id}
ALERTS ({len(alerts)} total):
{chr(10).join(alert_lines)}

SERVICE DEPENDENCY MAP:
{json.dumps(service_map, indent=2)}

RULES:
1. Analyze all alerts and the dependency graph to identify root causes.
2. Group correlated alerts into incidents (alerts caused by the same root failure).
3. Identify false alarms — alerts with borderline metrics barely exceeding thresholds.
4. For each real alert, determine: root_cause, severity, remediation.

VALID VALUES:
- root_cause: resource_exhaustion, network_failure, deployment_bug, config_error, dependency_outage, false_alarm
- severity: critical, high, medium, low
- remediation: restart_service, scale_up, rollback_deploy, fix_config, escalate_to_team, acknowledge_and_monitor, dismiss

IMPORTANT RULES FOR SEVERITY:
- If metric value is >> threshold (2x+), severity is "critical"
- If metric value is > threshold significantly, severity is "high"
- If metric value is moderately above threshold, severity is "medium"
- If metric barely exceeds threshold, severity is "low"
- For false alarms (barely exceeding threshold), use root_cause="false_alarm"

IMPORTANT RULES FOR REMEDIATION:
- resource_exhaustion → scale_up
- network_failure → restart_service
- deployment_bug → rollback_deploy
- config_error → fix_config
- dependency_outage → escalate_to_team
- false_alarm → dismiss

IMPORTANT: Look for stealth incidents — a data-layer service with subtle degradation
whose dependents show loud failures. The root cause is the data-layer service.

Look for context strings mentioning "cascade group" to identify incident membership.

Respond with ONLY valid JSON (no markdown), structured as:
{{
  "incidents": [
    {{"label": "incident-name", "alert_ids": ["alert-001", "alert-002"]}}
  ],
  "triage": [
    {{"alert_id": "alert-001", "root_cause": "...", "severity": "...", "remediation": "..."}}
  ],
  "skip": ["alert-005"]
}}
"""
    return prompt


def parse_plan(response_text: str) -> dict:
    """Parse LLM response into action plan."""
    text = response_text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            plan = json.loads(text[start:end])
        else:
            plan = {"incidents": [], "triage": [], "skip": []}

    # Validate and fix
    for t in plan.get("triage", []):
        if t.get("root_cause") not in VALID_ROOT_CAUSES:
            t["root_cause"] = "resource_exhaustion"
        t["remediation"] = RC_REMEDIATION.get(t["root_cause"], "acknowledge_and_monitor")
        if t.get("severity") not in VALID_SEVERITIES:
            t["severity"] = "medium"

    return plan


def run_task(task_id: str, seed: int = 42):
    """Run a single task episode."""
    # Reset
    resp = env_reset(task_id, seed)
    obs = resp["observation"]

    print(f"[START] task={task_id} env=cloud-alert-triage model={MODEL_NAME}")

    # Phase 1: Plan via LLM
    prompt = build_planning_prompt(obs, task_id)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
        )
        plan_text = completion.choices[0].message.content
        plan = parse_plan(plan_text)
    except Exception as e:
        print(f"[STEP] step=1 action=plan_failed reward=0.00 done=true error={str(e)}")
        print(f"[END] success=false steps=1 rewards=0.00")
        return

    # Phase 2: Execute — link first, then triage, then skip
    step_num = 0
    rewards = []
    done = False

    # Execute link actions
    for inc in plan.get("incidents", []):
        if done:
            break
        if len(inc.get("alert_ids", [])) < 2:
            continue
        action = {
            "action_type": "link_alerts",
            "alert_ids": inc["alert_ids"],
            "incident_label": inc.get("label", "incident"),
        }
        result = env_step(action)
        step_num += 1
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        rewards.append(reward)
        error = "null"
        print(f"[STEP] step={step_num} action=link_alerts('{inc.get('label', '')}') reward={reward:.2f} done={'true' if done else 'false'} error={error}")

    # Execute triage actions
    for t in plan.get("triage", []):
        if done:
            break
        action = {
            "action_type": "triage",
            "alert_id": t["alert_id"],
            "root_cause": t["root_cause"],
            "severity": t["severity"],
            "remediation": t["remediation"],
        }
        result = env_step(action)
        step_num += 1
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        rewards.append(reward)
        error_msg = result.get("observation", {}).get("feedback", "null")
        if "error" not in error_msg.lower():
            error_msg = "null"
        print(f"[STEP] step={step_num} action=triage('{t['alert_id']}') reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}")

    # Execute skip actions
    for alert_id in plan.get("skip", []):
        if done:
            break
        action = {
            "action_type": "skip",
            "alert_id": alert_id,
        }
        result = env_step(action)
        step_num += 1
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        rewards.append(reward)
        print(f"[STEP] step={step_num} action=skip('{alert_id}') reward={reward:.2f} done={'true' if done else 'false'} error=null")

    # If not done yet, check final info
    if done and result.get("info", {}).get("grader_score"):
        grader = result["info"]["grader_score"]
    else:
        grader = 0.0

    success = done and grader > 0.5
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={step_num} rewards={reward_str}")


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Running task: {task}")
        print(f"{'='*60}")
        run_task(task, seed=42)
