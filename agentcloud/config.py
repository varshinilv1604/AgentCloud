from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Paths
    project_root: Path
    logs_dir: Path
    log_file: Path
    sqlite_path: Path

    # Simulator
    tick_seconds: float = 0.25
    scenario: str = "mixed"  # mixed | normal | overload | crash | intrusion

    # Monitoring thresholds
    cpu_high_threshold: int = 85
    failed_login_burst: int = 5
    failed_login_window_seconds: int = 20

    # Agentic loop
    max_events: int = 200
    incident_cooldown_seconds: float = 8.0

    # Ablations / tracing
    disable_memory: bool = False
    disable_planning: bool = False
    trace_file: str | None = None

    # Demo mode
    demo_learning: bool = False

    # CLI output
    show_logs: bool = False


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "agentcloud" / "data" / "processed"
    log_file = root / "agentcloud" / "data" / "processed" / "hdfs.jsonl"
    sqlite_path = root / "database.db"

    tick = float(os.getenv("AGENTCLOUD_TICK_SECONDS", "0.25"))
    scenario = os.getenv("AGENTCLOUD_SCENARIO", "mixed").strip().lower()
    max_events = int(os.getenv("AGENTCLOUD_MAX_EVENTS", "200"))
    cooldown = float(os.getenv("AGENTCLOUD_INCIDENT_COOLDOWN_SECONDS", "8.0"))
    trace_file = os.getenv("AGENTCLOUD_TRACE_FILE", "").strip() or None

    cpu_high = int(os.getenv("AGENTCLOUD_CPU_HIGH_THRESHOLD", "85"))
    failed_burst = int(os.getenv("AGENTCLOUD_FAILED_LOGIN_BURST", "5"))
    failed_window = int(os.getenv("AGENTCLOUD_FAILED_LOGIN_WINDOW_SECONDS", "20"))

    return Settings(
        project_root=root,
        logs_dir=logs_dir,
        log_file=log_file,
        sqlite_path=sqlite_path,
        tick_seconds=tick,
        scenario=scenario,
        cpu_high_threshold=cpu_high,
        failed_login_burst=failed_burst,
        failed_login_window_seconds=failed_window,
        max_events=max_events,
        incident_cooldown_seconds=cooldown,
        trace_file=trace_file,
    )

