from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator

from .io_utils import now_iso
from .types import LogEvent


@dataclass
class SimState:
    active_server: str = "primary"
    primary_healthy: bool = True
    backup_healthy: bool = True
    isolated: bool = False
    blocked_ips: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blocked_ips is None:
            self.blocked_ips = set()


def _base_event(level: str, service: str, event: str, message: str) -> LogEvent:
    return {
        "ts": now_iso(),
        "level": level,  # type: ignore[typeddict-item]
        "service": service,
        "event": event,
        "message": message,
        "host": "agentcloud-1",
    }


def generate_events(scenario: str, tick_seconds: float) -> Iterator[LogEvent]:
    """
    Yields JSON-serializable log events forever.

    scenario:
      - normal | overload | crash | intrusion | mixed
    """
    rng = random.Random()
    scenario = scenario.lower().strip()
    step = 0

    while True:
        step += 1
        if scenario == "mixed":
            # Mostly normal; occasionally inject an incident burst.
            if step % 40 == 0:
                scenario_choice = rng.choice(["overload", "intrusion", "crash"])
            else:
                scenario_choice = "normal"
        else:
            scenario_choice = scenario

        if scenario_choice == "normal":
            cpu = rng.randint(10, 60)
            yield {
                **_base_event("INFO", "compute", "cpu", f"CPU usage steady at {cpu}%"),
                "cpu": cpu,
            }
            if step % 6 == 0:
                yield _base_event("INFO", "lb", "traffic", "Traffic within normal limits")
            if step % 10 == 0:
                yield _base_event("INFO", "system", "heartbeat", "All services healthy")

        elif scenario_choice == "overload":
            # Short spike burst; emits high CPU + traffic warnings.
            for _ in range(rng.randint(6, 12)):
                cpu = rng.randint(86, 99)
                yield {
                    **_base_event("WARN", "compute", "cpu", f"High CPU usage detected: {cpu}%"),
                    "cpu": cpu,
                }
                yield _base_event("WARN", "lb", "traffic", "Traffic spike observed")
                time.sleep(tick_seconds)

        elif scenario_choice == "intrusion":
            attacker_ip = f"203.0.113.{rng.randint(2, 250)}"
            # Burst of failed logins
            for i in range(rng.randint(5, 9)):
                yield {
                    **_base_event(
                        "ALERT",
                        "auth",
                        "auth",
                        f"Failed login attempt #{i+1} from {attacker_ip}",
                    ),
                    "ip": attacker_ip,
                    "user": rng.choice(["root", "admin", "ubuntu", "ec2-user"]),
                    "status": "failed",
                    "code": "AUTH_FAILED",
                }
                time.sleep(tick_seconds)

        elif scenario_choice == "crash":
            yield _base_event("ERROR", "compute", "error", "Service crashed unexpectedly (segfault)")
            yield {
                **_base_event("INFO", "compute", "cpu", "CPU usage unavailable (process down)"),
                "cpu": 0,
            }

        else:
            yield _base_event("INFO", "system", "heartbeat", "Unknown scenario; emitting heartbeat")

        time.sleep(tick_seconds)

