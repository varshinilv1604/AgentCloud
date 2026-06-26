from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class LogEvent(TypedDict, total=False):
    ts: str
    level: Literal["INFO", "WARN", "ERROR", "ALERT"]
    service: str
    event: str  # cpu | error | auth | traffic | heartbeat
    message: str
    cpu: int
    host: str
    ip: str
    user: str
    status: str
    code: str


class AnomalyAlert(TypedDict):
    status: Literal["anomaly_detected"]
    type: Literal["overload", "crash", "intrusion"]


class Diagnosis(TypedDict):
    incident: Literal["overload", "crash", "intrusion", "normal"]
    cause: str
    severity: Literal["low", "medium", "high"]


class Plan(TypedDict):
    action: Literal["restart_service", "isolate_server", "reroute_traffic", "block_ip"]
    target: str


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    details: str
    new_state: dict[str, Any]

