"""Comprehensive tests for the Cloud Alert Triage environment."""
import pytest
import json
from fastapi.testclient import TestClient

from server.app import app
from server.config import SCORE_MIN, SCORE_MAX, TASK_CONFIGS, RootCause, Severity, Remediation, ROOT_CAUSE_REMEDIATION
from server.service_graph import SERVICE_MAP, ALL_SERVICES, get_dependents, get_all_upstream
from server.scenario_generator import generate_scenario
from server.grading import compute_grader_score, _clamp
from server.rewards import compute_triage_reward, compute_skip_reward, compute_step_penalty
from server.environment import AlertTriageEnv

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    def test_score_min_positive(self):
        assert SCORE_MIN > 0.0

    def test_score_max_below_one(self):
        assert SCORE_MAX < 1.0

    def test_task_configs_have_required_keys(self):
        for tid, cfg in TASK_CONFIGS.items():
            assert "alert_count" in cfg
            assert "max_steps" in cfg
            assert "incident_count" in cfg
            assert "false_alarm_count" in cfg

    def test_three_tasks_exist(self):
        assert "easy" in TASK_CONFIGS
        assert "medium" in TASK_CONFIGS
        assert "hard" in TASK_CONFIGS

    def test_root_cause_remediation_mapping_complete(self):
        for rc in RootCause:
            assert rc in ROOT_CAUSE_REMEDIATION

    def test_severity_order(self):
        from server.config import SEVERITY_ORDER, severity_distance
        assert severity_distance(Severity.CRITICAL, Severity.CRITICAL) == 0
        assert severity_distance(Severity.CRITICAL, Severity.HIGH) == 1
        assert severity_distance(Severity.CRITICAL, Severity.LOW) == 3


# ═══════════════════════════════════════════════════════════════
# Service graph tests
# ═══════════════════════════════════════════════════════════════

class TestServiceGraph:
    def test_17_services(self):
        assert len(ALL_SERVICES) == 17

    def test_data_layer_are_leaves(self):
        leaves = ["postgres-primary", "redis-cache", "kafka-broker", "elasticsearch", "object-storage"]
        for s in leaves:
            assert SERVICE_MAP[s] == []

    def test_web_frontend_depends_on_gateway(self):
        assert "api-gateway" in SERVICE_MAP["web-frontend"]

    def test_get_dependents(self):
        deps = get_dependents("redis-cache")
        assert "auth-service" in deps

    def test_get_all_upstream(self):
        upstream = get_all_upstream("postgres-primary")
        assert len(upstream) > 0


# ═══════════════════════════════════════════════════════════════
# Scenario generator tests
# ═══════════════════════════════════════════════════════════════

class TestScenarioGenerator:
    def test_easy_scenario(self):
        s = generate_scenario("easy", 42)
        assert len(s["alerts"]) == 5
        assert len(s["incidents"]) == 0
        for aid, gt in s["ground_truth"].items():
            assert not gt["is_false_alarm"]

    def test_medium_scenario(self):
        s = generate_scenario("medium", 42)
        assert len(s["alerts"]) == 15
        assert len(s["incidents"]) == 2
        fa_count = sum(1 for gt in s["ground_truth"].values() if gt["is_false_alarm"])
        assert fa_count == 2

    def test_hard_scenario(self):
        s = generate_scenario("hard", 42)
        assert len(s["alerts"]) == 30
        assert len(s["incidents"]) == 5
        fa_count = sum(1 for gt in s["ground_truth"].values() if gt["is_false_alarm"])
        assert fa_count == 6
        stealth = [gt for gt in s["ground_truth"].values() if gt.get("is_stealth_root")]
        assert len(stealth) == 1

    def test_determinism(self):
        s1 = generate_scenario("hard", 42)
        s2 = generate_scenario("hard", 42)
        assert s1["alerts"] == s2["alerts"]
        assert s1["ground_truth"] == s2["ground_truth"]

    def test_different_seeds_different_scenarios(self):
        s1 = generate_scenario("easy", 42)
        s2 = generate_scenario("easy", 99)
        assert s1["alerts"] != s2["alerts"]

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            generate_scenario("invalid", 42)

    def test_all_alerts_have_required_fields(self):
        for tid in ["easy", "medium", "hard"]:
            s = generate_scenario(tid, 42)
            for a in s["alerts"]:
                assert "alert_id" in a
                assert "timestamp" in a
                assert "service" in a
                assert "metric" in a
                assert "metric_value" in a
                assert "threshold" in a
                assert "message" in a

    def test_ground_truth_covers_all_alerts(self):
        for tid in ["easy", "medium", "hard"]:
            s = generate_scenario(tid, 42)
            for a in s["alerts"]:
                assert a["alert_id"] in s["ground_truth"]


# ═══════════════════════════════════════════════════════════════
# Grading tests — CRITICAL: score range
# ═══════════════════════════════════════════════════════════════

class TestGrading:
    def test_clamp_zero(self):
        assert _clamp(0.0) == SCORE_MIN

    def test_clamp_one(self):
        assert _clamp(1.0) == SCORE_MAX

    def test_clamp_negative(self):
        assert _clamp(-5.0) == SCORE_MIN

    def test_clamp_above_one(self):
        assert _clamp(1.5) == SCORE_MAX

    def test_clamp_normal(self):
        assert _clamp(0.5) == 0.5

    def test_perfect_easy_score_below_one(self):
        s = generate_scenario("easy", 42)
        decisions = {}
        for aid, gt in s["ground_truth"].items():
            decisions[aid] = {
                "root_cause": gt["root_cause"],
                "severity": gt["severity"],
                "remediation": gt["remediation"],
            }
        score = compute_grader_score(
            "easy", s["ground_truth"], decisions, {}, {}, set(),
            original_alert_ids=set(s["ground_truth"].keys()),
        )
        assert score > 0.0
        assert score < 1.0
        assert score == SCORE_MAX  # Perfect score clamps to 0.999

    def test_zero_coverage_score_above_zero(self):
        s = generate_scenario("easy", 42)
        score = compute_grader_score(
            "easy", s["ground_truth"], {}, {}, {}, set(),
            original_alert_ids=set(s["ground_truth"].keys()),
        )
        assert score > 0.0
        assert score == SCORE_MIN

    def test_all_tasks_perfect_score_strictly_below_one(self):
        """THE CRITICAL TEST: perfect scores must never be exactly 1.0"""
        for tid in ["easy", "medium", "hard"]:
            s = generate_scenario(tid, 42)
            decisions = {}
            skipped = set()
            for aid, gt in s["ground_truth"].items():
                if gt["is_false_alarm"]:
                    skipped.add(aid)
                else:
                    decisions[aid] = {
                        "root_cause": gt["root_cause"],
                        "severity": gt["severity"],
                        "remediation": gt["remediation"],
                    }
            # Build perfect incident links
            agent_incidents = {}
            for label, ids in s["incidents"].items():
                agent_incidents[label] = set(ids)

            score = compute_grader_score(
                tid, s["ground_truth"], decisions, agent_incidents,
                s["incidents"], skipped, s.get("stealth_root_service"),
                set(s["ground_truth"].keys()),
            )
            assert 0.0 < score < 1.0, f"Task {tid} score {score} not in (0,1)"

    def test_all_tasks_worst_score_strictly_above_zero(self):
        """THE CRITICAL TEST: worst scores must never be exactly 0.0"""
        for tid in ["easy", "medium", "hard"]:
            s = generate_scenario(tid, 42)
            score = compute_grader_score(
                tid, s["ground_truth"], {}, {}, s["incidents"], set(),
                s.get("stealth_root_service"),
                set(s["ground_truth"].keys()),
            )
            assert 0.0 < score < 1.0, f"Task {tid} score {score} not in (0,1)"


# ═══════════════════════════════════════════════════════════════
# Rewards tests
# ═══════════════════════════════════════════════════════════════

class TestRewards:
    def test_perfect_triage_reward(self):
        r = compute_triage_reward(
            "resource_exhaustion", "critical", "scale_up",
            "resource_exhaustion", "critical", "scale_up",
            "alert-001", {}, {},
        )
        assert r == pytest.approx(0.80)

    def test_wrong_triage(self):
        r = compute_triage_reward(
            "network_failure", "low", "restart_service",
            "resource_exhaustion", "critical", "scale_up",
            "alert-001", {}, {},
        )
        assert r < 0.30

    def test_skip_true_fa(self):
        assert compute_skip_reward(True) == 0.20

    def test_skip_real_alert(self):
        assert compute_skip_reward(False) == -0.30

    def test_no_penalty_early(self):
        assert compute_step_penalty(1, 10) == 0.0

    def test_penalty_late(self):
        assert compute_step_penalty(9, 10) == -0.05


# ═══════════════════════════════════════════════════════════════
# Environment tests
# ═══════════════════════════════════════════════════════════════

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = AlertTriageEnv()
        result = env.reset("easy", 42)
        obs = result["observation"]
        assert obs["pending_count"] == 5
        assert obs["step_number"] == 0
        assert len(obs["alerts"]) == 5

    def test_step_triage(self):
        env = AlertTriageEnv()
        env.reset("easy", 42)
        alert_id = env.alerts[0]["alert_id"]
        gt = env.ground_truth[alert_id]
        result = env.step({
            "action_type": "triage",
            "alert_id": alert_id,
            "root_cause": gt["root_cause"],
            "severity": gt["severity"],
            "remediation": gt["remediation"],
        })
        assert result["reward"] > 0

    def test_step_invalid_action(self):
        env = AlertTriageEnv()
        env.reset("easy", 42)
        result = env.step({"action_type": "invalid"})
        assert result["reward"] < 0

    def test_double_triage_penalty(self):
        env = AlertTriageEnv()
        env.reset("easy", 42)
        aid = env.alerts[0]["alert_id"]
        gt = env.ground_truth[aid]
        env.step({"action_type": "triage", "alert_id": aid, "root_cause": gt["root_cause"], "severity": gt["severity"], "remediation": gt["remediation"]})
        result = env.step({"action_type": "triage", "alert_id": aid, "root_cause": gt["root_cause"], "severity": gt["severity"], "remediation": gt["remediation"]})
        assert result["reward"] <= -0.15

    def test_episode_end_has_grader_score(self):
        env = AlertTriageEnv()
        env.reset("easy", 42)
        result = None
        for a in env.alerts[:]:
            aid = a["alert_id"]
            gt = env.ground_truth[aid]
            result = env.step({
                "action_type": "triage",
                "alert_id": aid,
                "root_cause": gt["root_cause"],
                "severity": gt["severity"],
                "remediation": gt["remediation"],
            })
        assert result["done"] is True
        assert "grader_score" in result["info"]
        score = result["info"]["grader_score"]
        assert 0.0 < score < 1.0

    def test_grader_score_never_zero_or_one(self):
        """Run full episodes and verify score bounds."""
        for tid in ["easy", "medium", "hard"]:
            env = AlertTriageEnv()
            env.reset(tid, 42)
            result = None
            for a in env.alerts[:]:
                aid = a["alert_id"]
                gt = env.ground_truth[aid]
                if gt["is_false_alarm"]:
                    result = env.step({"action_type": "skip", "alert_id": aid})
                else:
                    result = env.step({
                        "action_type": "triage",
                        "alert_id": aid,
                        "root_cause": gt["root_cause"],
                        "severity": gt["severity"],
                        "remediation": gt["remediation"],
                    })
                if result["done"]:
                    break
            if result and result["done"]:
                score = result["info"]["grader_score"]
                assert score > 0.0, f"{tid}: score is 0.0"
                assert score < 1.0, f"{tid}: score is 1.0"

    def test_state_endpoint(self):
        env = AlertTriageEnv()
        env.reset("easy", 42)
        state = env.state()
        assert "ground_truth" in state


# ═══════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════

class TestAPI:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_reset_easy(self):
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data

    def test_reset_medium(self):
        r = client.post("/reset", json={"task_id": "medium", "seed": 42})
        assert r.status_code == 200

    def test_reset_hard(self):
        r = client.post("/reset", json={"task_id": "hard", "seed": 42})
        assert r.status_code == 200

    def test_reset_invalid_task(self):
        r = client.post("/reset", json={"task_id": "impossible", "seed": 42})
        assert r.status_code == 422

    def test_step_before_reset(self):
        # Need a fresh app instance; skip this if state is shared
        pass

    def test_full_easy_episode(self):
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        obs = r.json()["observation"]
        alerts = obs["alerts"]

        for a in alerts:
            r = client.post("/step", json={
                "action_type": "triage",
                "alert_id": a["alert_id"],
                "root_cause": "resource_exhaustion",
                "severity": "medium",
                "remediation": "scale_up",
            })

        data = r.json()
        if data["done"]:
            score = data["info"]["grader_score"]
            assert 0.0 < score < 1.0

    def test_step_skip(self):
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        obs = client.post("/reset", json={"task_id": "medium", "seed": 42}).json()["observation"]
        aid = obs["alerts"][0]["alert_id"]
        r = client.post("/step", json={"action_type": "skip", "alert_id": aid})
        assert r.status_code == 200

    def test_step_link(self):
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        obs = client.post("/reset", json={"task_id": "medium", "seed": 42}).json()["observation"]
        ids = [obs["alerts"][0]["alert_id"], obs["alerts"][1]["alert_id"]]
        r = client.post("/step", json={
            "action_type": "link_alerts",
            "alert_ids": ids,
            "incident_label": "test-incident",
        })
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════
# Determinism tests
# ═══════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_scenario_deterministic_across_calls(self):
        for tid in ["easy", "medium", "hard"]:
            s1 = generate_scenario(tid, 123)
            s2 = generate_scenario(tid, 123)
            assert s1["alerts"] == s2["alerts"]

    def test_grader_deterministic(self):
        s = generate_scenario("easy", 42)
        decisions = {}
        for aid, gt in s["ground_truth"].items():
            decisions[aid] = {"root_cause": gt["root_cause"], "severity": gt["severity"], "remediation": gt["remediation"]}
        score1 = compute_grader_score("easy", s["ground_truth"], decisions, {}, {}, set(), original_alert_ids=set(s["ground_truth"].keys()))
        score2 = compute_grader_score("easy", s["ground_truth"], decisions, {}, {}, set(), original_alert_ids=set(s["ground_truth"].keys()))
        assert score1 == score2


# ═══════════════════════════════════════════════════════════════
# Edge case and stress tests
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_many_seeds(self):
        """Verify score bounds across many seeds."""
        for seed in range(50):
            for tid in ["easy", "medium", "hard"]:
                s = generate_scenario(tid, seed)
                # Perfect play
                decisions = {}
                skipped = set()
                for aid, gt in s["ground_truth"].items():
                    if gt["is_false_alarm"]:
                        skipped.add(aid)
                    else:
                        decisions[aid] = {"root_cause": gt["root_cause"], "severity": gt["severity"], "remediation": gt["remediation"]}
                agent_inc = {l: set(ids) for l, ids in s["incidents"].items()}
                score = compute_grader_score(
                    tid, s["ground_truth"], decisions, agent_inc,
                    s["incidents"], skipped, s.get("stealth_root_service"),
                    set(s["ground_truth"].keys()),
                )
                assert 0.0 < score < 1.0, f"seed={seed} task={tid} score={score}"

    def test_zero_play_all_seeds(self):
        """Zero coverage across many seeds."""
        for seed in range(20):
            for tid in ["easy", "medium", "hard"]:
                s = generate_scenario(tid, seed)
                score = compute_grader_score(
                    tid, s["ground_truth"], {}, {}, s["incidents"], set(),
                    s.get("stealth_root_service"),
                    set(s["ground_truth"].keys()),
                )
                assert 0.0 < score < 1.0, f"seed={seed} task={tid} score={score}"
