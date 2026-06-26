from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agents.diagnosis import DiagnosisAgent
from .agents.execution import ExecutionAgent
from .agents.memory import MemoryAgent
from .agents.monitoring import MonitoringAgent
from .agents.planning import PlanningAgent
from .config import Settings, load_settings
#from .simulator import generate_events
from .types import AnomalyAlert, Diagnosis, LogEvent, Plan


@dataclass
class EpisodeResult:
    scenario: str
    incidents_handled: int
    success_count: int
    failure_count: int
    mean_steps_to_recover: float
    mean_mttr_seconds: float
    tool_selection_accuracy: float


def _reset_sqlite(path: Path | str) -> None:
    if str(path) == ":memory:":
        return

    p = Path(str(path))

    if p.exists():
        p.unlink()


def _baseline_plan(diagnosis: Diagnosis) -> Plan:

    if diagnosis["incident"] == "intrusion":
        return {
            "action": "block_ip",
            "target": "attacker_ip"
        }

    if diagnosis["incident"] == "crash":
        return {
            "action": "restart_service",
            "target": "compute"
        }

    if diagnosis["incident"] == "overload":
        return {
            "action": "restart_service",
            "target": "lb"
        }

    return {
        "action": "restart_service",
        "target": "compute"
    }
def run_episode(
    *,
    monitoring: MonitoringAgent,
    diagnosis_agent: DiagnosisAgent,
    planner: PlanningAgent,
    execution: ExecutionAgent,
    memory: MemoryAgent,
    settings: Settings,
    scenario: str,
    max_incidents: int,
    disable_memory: bool,
    disable_planning: bool,
    seed: Optional[int] = None,
) -> EpisodeResult:
    rng = random.Random(seed)
    events = []

    with open("agentcloud/data/processed/hdfs.jsonl", "r") as f:

        for line in f:

            try:

                event = json.loads(line)

                events.append(event)
                if len(events) >= 100:
                    break
                

            except Exception:
                continue

    events = iter(events)


    if not disable_memory:
        # start clean for each episode so results are comparable
        _reset_sqlite(settings.sqlite_path)

    recent: list[LogEvent] = []
    cooldown_until_by_type: dict[str, float] = {}

    incidents_handled = 0
    successes = 0
    failures = 0
    recovery_steps: list[int] = []
    mttr_seconds: list[float] = []
    tool_correct: int = 0
    tool_total: int = 0

    step = 0
    start = time.time()
    while incidents_handled < max_incidents and (time.time() - start) < 20:
        step += 1
        try:
            event = next(events)
        except StopIteration:
            break
        recent.append(event)
        if len(recent) > 120:
            recent = recent[-120:]

        alert: Optional[AnomalyAlert] = monitoring.observe(event)
        if not alert:
            continue

        alert_type = alert["type"]
        now = time.time()
        if cooldown_until_by_type.get(alert_type, 0.0) > now:
            continue

        diag = diagnosis_agent.diagnose(alert, recent)
        hint = None
        #if not disable_memory:
        #   hint = memory.recall_plan_hint(diag)

        oracle = _baseline_plan(diag)["action"]

        attempts = 0
        result = None
        incident_start = time.time()
        last_failed_action: Optional[str] = None
        while attempts < 3:
            attempts += 1
            if disable_planning:
                plan = _baseline_plan(diag)
            else:
                plan = planner.plan(diag, memory_hint=hint, memory_agent=memory)

            # Failure-aware adjustment (same logic as CLI)
            if last_failed_action and plan["action"] == last_failed_action:
                if diag["incident"] == "overload":
                    plan = {"action": "restart_service", "target": "lb"}
                elif diag["incident"] == "intrusion":
                    plan = {"action": "isolate_server", "target": "auth"}
                else:
                    plan = {"action": "restart_service", "target": "compute"}

            if attempts == 1:
                tool_total += 1
                if plan["action"] == oracle:
                    tool_correct += 1

            result = execution.execute(plan, recent)
            if not disable_memory:
                memory.remember(diag, plan, success=result.success)

            if result.success:
                break
            last_failed_action = plan["action"]

        mttr_seconds.append(max(0.0, time.time() - incident_start))

        incidents_handled += 1
        recovery_steps.append(attempts)
        if result.success:
            successes += 1
        else:
            failures += 1

        cooldown_until_by_type[alert_type] = time.time() + settings.incident_cooldown_seconds

        # add tiny randomness so different seeds sample different points in mixed mode

    mean_steps = (sum(recovery_steps) / len(recovery_steps)) if recovery_steps else 0.0
    mean_mttr = (sum(mttr_seconds) / len(mttr_seconds)) if mttr_seconds else 0.0
    tool_acc = (tool_correct / tool_total) if tool_total else 0.0
    return EpisodeResult(
        scenario=scenario,
        incidents_handled=incidents_handled,
        success_count=successes,
        failure_count=failures,
        mean_steps_to_recover=mean_steps,
        mean_mttr_seconds=mean_mttr,
        tool_selection_accuracy=tool_acc,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentCloud evaluation / ablation runner")
    p.add_argument("--scenario", default="mixed", help="normal|overload|crash|intrusion|mixed")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-incidents", type=int, default=5)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--no-planning", action="store_true")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--ablation", action="store_true", help="Also print no-memory / no-planning comparison")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = load_settings()
    monitoring = MonitoringAgent(
    cpu_high_threshold=settings.cpu_high_threshold,
    failed_login_burst=settings.failed_login_burst,
    failed_login_window_seconds=settings.failed_login_window_seconds,
    )

    diagnosis_agent = DiagnosisAgent(verbose=False)

    planner = PlanningAgent()

    execution = ExecutionAgent()

    memory = MemoryAgent(
        sqlite_path=settings.sqlite_path
    )
    # In-memory DB keeps evaluation fast and avoids filesystem issues.
    settings = Settings(**{**settings.__dict__, "sqlite_path": ":memory:"})

    def _run(disable_memory: bool, disable_planning: bool) -> list[EpisodeResult]:
        rs: list[EpisodeResult] = []
        for i in range(args.episodes):
            rs.append(
               run_episode(
                    monitoring=monitoring,
                    diagnosis_agent=diagnosis_agent,
                    planner=planner,
                    execution=execution,
                    memory=memory,

                    settings=settings,
                    scenario=str(args.scenario),
                    max_incidents=int(args.max_incidents),
                    disable_memory=disable_memory,
                    disable_planning=disable_planning,
                    seed=i,
                )
            )
        return rs

    results = _run(bool(args.no_memory), bool(args.no_planning))

    total_incidents = sum(r.incidents_handled for r in results)
    total_success = sum(r.success_count for r in results)
    total_failure = sum(r.failure_count for r in results)
    mean_steps = (
        sum(r.mean_steps_to_recover for r in results) / len(results) if results else 0.0
    )
    mean_mttr = sum(r.mean_mttr_seconds for r in results) / len(results) if results else 0.0
    mean_tool_acc = sum(r.tool_selection_accuracy for r in results) / len(results) if results else 0.0

    summary = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "max_incidents": args.max_incidents,

        "disable_memory": bool(args.no_memory),
        "disable_planning": bool(args.no_planning),

        "memory_enabled": not bool(args.no_memory),
        "planning_enabled": not bool(args.no_planning),

        "total_incidents": total_incidents,

        "success_rate": (
            total_success / total_incidents
        ) if total_incidents else 0.0,

        "failure_rate": (
            total_failure / total_incidents
        ) if total_incidents else 0.0,

        "autonomous_recovery_rate": (
            total_success / total_incidents
        ) if total_incidents else 0.0,

        "incidents_processed_per_episode": (
            total_incidents / args.episodes
        ) if args.episodes else 0.0,

        "avg_recovery_steps": mean_steps,

        "mttr_seconds": mean_mttr,

        "tool_selection_accuracy": mean_tool_acc,

        "success_count": total_success,

        "failure_count": total_failure,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("===== EVALUATION RESULTS =====")
        print(f"Episodes: {args.episodes}")
        print(f"Scenario: {args.scenario}")
        print(f"Success Rate: {round(summary['success_rate'] * 100)}%")
        print(f"Avg Recovery Steps: {summary['avg_recovery_steps']:.2f}")
        print(f"MTTR: {summary['mttr_seconds']:.3f} sec")
        print(f"Tool Selection Accuracy: {round(summary['tool_selection_accuracy'] * 100)}%")
        if args.ablation and not args.json:

            print("\n--- ABLATION ---")

            base = summary["success_rate"]

            nm = _run(True, bool(args.no_planning))

            nm_succ = (
                sum(r.success_count for r in nm)
                / max(1, sum(r.incidents_handled for r in nm))
            )

            np = _run(bool(args.no_memory), True)

            np_succ = (
                sum(r.success_count for r in np)
                / max(1, sum(r.incidents_handled for r in np))
            )
            full_steps = mean_steps

            nm_mean_steps = (
                sum(r.mean_steps_to_recover for r in nm)
                / max(1, len(nm))
            )

            np_mean_steps = (
                sum(r.mean_steps_to_recover for r in np)
                / max(1, len(np))
            )

            print(
                f"Full System Avg Steps: "
                f"{full_steps:.2f}"
            )

            print(
                f"No Memory Avg Steps: "
                f"{nm_mean_steps:.2f}"
            )

            print(
                f"No Planning Avg Steps: "
                f"{np_mean_steps:.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

