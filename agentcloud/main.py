from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Optional
import time

from .config import Settings, load_settings
from .io_utils import append_jsonl
from .simulator import generate_events
from .types import AnomalyAlert, LogEvent
from .agents.diagnosis import DiagnosisAgent
from .agents.execution import ExecutionAgent
from .agents.memory import MemoryAgent
from .agents.monitoring import MonitoringAgent
from .agents.planning import PlanningAgent
from .validation import is_diagnosis, is_plan


@dataclass
class Pipeline:
    settings: Settings
    monitoring: MonitoringAgent
    diagnosis: DiagnosisAgent
    planning: PlanningAgent
    execution: ExecutionAgent
    memory: MemoryAgent


def _start_log_generator(settings: Settings, q: "queue.Queue[LogEvent]", stop: threading.Event) -> threading.Thread:
    def _run() -> None:
        for event in generate_events(settings.scenario, settings.tick_seconds):
            if stop.is_set():
                return
            append_jsonl(settings.log_file, event)
            q.put(event)

    t = threading.Thread(target=_run, name="agentcloud-log-generator", daemon=True)
    t.start()
    return t


def _print_event(event: dict, *, enabled: bool) -> None:
    # Keep output compact and demo-friendly.
    if not enabled:
        return
    level = event.get("level", "INFO")
    msg = event.get("message", "")
    print(f"[{level}] {msg}")


def _ts_now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _write_trace_line(trace_file: Optional[str], obj: dict) -> None:
    if not trace_file:
        return
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _start_file_reader(log_file, q: "queue.Queue[LogEvent]", stop: threading.Event,delay=1.0):
    def _run():
        with open(log_file, "r") as f:
            for line in f:
                if stop.is_set():
                    return
                event = json.loads(line)
                q.put(event)
                time.sleep(delay) 

    t = threading.Thread(target=_run, name="agentcloud-file-reader", daemon=True)
    t.start()
    return t

def _run_agentic_loop(p: Pipeline) -> int:
    p.settings.logs_dir.mkdir(parents=True, exist_ok=True)

    q: "queue.Queue[LogEvent]" = queue.Queue()
    stop = threading.Event()
    _start_file_reader(p.settings.log_file, q, stop)

    recent: Deque[LogEvent] = deque(maxlen=80)
    handled = 0
    cooldown_until_by_type: dict[str, float] = {}

    try:
        while handled < p.settings.max_events:
            try:
                event = q.get(timeout=2.0)
            except queue.Empty:
                continue

            recent.append(event)


            _print_event(event, enabled=p.settings.show_logs)
            _write_trace_line(
                p.settings.trace_file,
                {"timestamp": datetime.now(timezone.utc).isoformat(), "agent": "simulator", "kind": "log", "event": event},
            )

            alert: Optional[AnomalyAlert] = p.monitoring.observe(event)
            if alert:
                print("[ALERT DETECTED]", alert)
            if not alert:
                continue

            alert_type = alert["type"]
            now = _ts_now()
            if cooldown_until_by_type.get(alert_type, 0.0) > now:
                continue

            print(f"[MONITOR] Anomaly detected: {alert_type}")
            _write_trace_line(
                p.settings.trace_file,
                {"timestamp": datetime.now(timezone.utc).isoformat(), "agent": "monitor", "source": "rule", "kind": "alert", "alert": alert},
            )

            diagnosis = p.diagnosis.diagnose(alert, list(recent))
            if not is_diagnosis(diagnosis):
                print("[DIAGNOSIS] invalid output; skipping")
                continue

            print(f"[DIAGNOSIS] {diagnosis['incident']} ({diagnosis['severity']}) - {diagnosis['cause']}")
            _write_trace_line(
                p.settings.trace_file,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "diagnosis",
                    "source": "llm" if p.diagnosis.llm is not None else "fallback",
                    "kind": "diagnosis",
                    "diagnosis": diagnosis,
                },
            )

            hint = None
            if not p.settings.disable_memory:
                hint = p.memory.get_similar_incident(diagnosis["incident"])

            recent_failed_action = None
            if not p.settings.disable_memory:
                recent_failed_action = p.memory.get_recent_failure_action(diagnosis["incident"])

            if p.settings.disable_planning:
                if diagnosis["incident"] == "intrusion":
                    plan = {"action": "block_ip", "target": "attacker_ip"}
                elif diagnosis["incident"] == "crash":
                    plan = {"action": "restart_service", "target": "compute"}
                else:
                    plan = {"action": "reroute_traffic", "target": "backup_server"}
            else:
                plan = p.planning.plan(diagnosis, memory_hint=hint, memory_agent=p.memory)
            if not is_plan(plan):
                print("[PLAN] invalid output; skipping")
                continue

            source = "planning"
            if hint is not None:
                source = "memory"

            # Failure-aware adjustment (demonstrates learning behavior)
            if recent_failed_action and plan["action"] == recent_failed_action:
                print("[PLANNING] Adjusted using memory (avoiding known failed action)")
                if diagnosis["incident"] == "overload":
                    plan = {"action": "restart_service", "target": "lb"}
                elif diagnosis["incident"] == "intrusion":
                    plan = {"action": "isolate_server", "target": "auth"}
                else:
                    plan = {"action": "restart_service", "target": "compute"}
                source = "memory_adjusted"

            print(f"[PLAN] {plan['action']} (target={plan['target']})")
            _write_trace_line(
                p.settings.trace_file,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "planning",
                    "source": source,
                    "kind": "plan",
                    "plan": plan,
                },
            )

            result = p.execution.execute(plan, list(recent))
            status = "Resolved" if result.success else "Failed"
            if result.success:
                print(f"[EXECUTE] {result.details}")
            else:
                print("[EXECUTION] Failed")
                print("[EXECUTE] " + result.details)
            print(f"[STATUS] {status}")
            _write_trace_line(
                p.settings.trace_file,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "execution",
                    "source": "tool",
                    "kind": "execution",
                    "result": {
                        "success": result.success,
                        "details": result.details,
                        "new_state": result.new_state,
                    },
                },
            )

            if not p.settings.disable_memory:
                p.memory.remember(diagnosis, plan, success=result.success)
                if result.success:
                    print("[MEMORY] Stored success")
                else:
                    print("[MEMORY] Stored failure")
            handled += 1
            cooldown_until_by_type[alert_type] = _ts_now() + p.settings.incident_cooldown_seconds

        return 0
    except KeyboardInterrupt:
        print("\nStopping.")
        return 130
    finally:
        stop.set()


def build_pipeline(settings: Settings) -> Pipeline:



    memory = MemoryAgent(
        sqlite_path=settings.sqlite_path
    )

   

    monitoring = MonitoringAgent(
        cpu_high_threshold=settings.cpu_high_threshold,
        failed_login_burst=settings.failed_login_burst,
        failed_login_window_seconds=settings.failed_login_window_seconds,
    )



    diagnosis = DiagnosisAgent(
        verbose=False
    )

   

    planning = PlanningAgent(
        memory=memory
    )

    

    execution = ExecutionAgent(
        demo_fail_first_reroute=settings.demo_learning
    )



    return Pipeline(

        settings=settings,

        monitoring=monitoring,

        diagnosis=diagnosis,

        planning=planning,

        execution=execution,

        memory=memory,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentCloud CLI demo (CPU-only, simulated).")
    parser.add_argument("--scenario", default=None, help="normal|overload|crash|intrusion|mixed")
    parser.add_argument("--max-events", type=int, default=None, help="Max incidents to handle before exit")
    parser.add_argument("--tick", type=float, default=None, help="Seconds between simulated events")
    parser.add_argument("--no-memory", action="store_true", help="Ablation: disable memory agent read/write")
    parser.add_argument("--no-planning", action="store_true", help="Ablation: disable planning agent")
    parser.add_argument("--trace-file", default=None, help="Write JSONL trace of pipeline decisions")
    parser.add_argument("--demo-learning", action="store_true", help="Demo: force one failure then memory-adjusted recovery")
    parser.add_argument("--show-logs", action="store_true", help="Print raw simulator log lines")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = load_settings()
    print("1 settings loaded")
    if args.scenario:
        settings = Settings(**{**settings.__dict__, "scenario": str(args.scenario).lower()})
    if args.max_events is not None:
        settings = Settings(**{**settings.__dict__, "max_events": int(args.max_events)})
    if args.tick is not None:
        settings = Settings(**{**settings.__dict__, "tick_seconds": float(args.tick)})
    if args.no_memory:
        settings = Settings(**{**settings.__dict__, "disable_memory": True})
    if args.no_planning:
        settings = Settings(**{**settings.__dict__, "disable_planning": True})
    if args.trace_file:
        settings = Settings(**{**settings.__dict__, "trace_file": str(args.trace_file)})
    if args.show_logs:
        settings = Settings(**{**settings.__dict__, "show_logs": True})
    if args.demo_learning:
        demo_db = settings.project_root / "demo.db"
        try:
            demo_db.unlink()
        except FileNotFoundError:
            pass
        settings = Settings(
            **{
                **settings.__dict__,
                "demo_learning": True,
                "scenario": "overload",
                "max_events": 2,
                "sqlite_path": demo_db,
            }
        )

    pipeline = build_pipeline(settings)
    return _run_agentic_loop(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())

