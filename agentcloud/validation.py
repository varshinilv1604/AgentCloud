from __future__ import annotations

from typing import Any, Literal, TypeGuard

from .types import Diagnosis, Plan


def is_diagnosis(obj: Any) -> TypeGuard[Diagnosis]:
    if not isinstance(obj, dict):
        return False
    if obj.get("incident") not in ("overload", "crash", "intrusion", "normal"):
        return False
    if not isinstance(obj.get("cause"), str) or not obj.get("cause"):
        return False
    if obj.get("severity") not in ("low", "medium", "high"):
        return False
    return True


def is_plan(obj: Any) -> TypeGuard[Plan]:
    if not isinstance(obj, dict):
        return False
    if obj.get("action") not in ("restart_service", "isolate_server", "reroute_traffic", "block_ip"):
        return False
    if not isinstance(obj.get("target"), str) or not obj.get("target"):
        return False
    return True

