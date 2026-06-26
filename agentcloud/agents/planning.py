from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import Diagnosis, Plan
from ..validation import is_plan
from .memory import MemoryAgent
from ..llm import generate_plan_with_qwen


@dataclass
class PlanningAgent:

    memory: Optional[MemoryAgent] = None

    def plan(
        self,
        diagnosis: Diagnosis,
        memory_hint: Optional[Plan] = None,
        memory_agent: Optional[object] = None,
    ) -> Plan:

        incident = diagnosis["incident"]

        severity = diagnosis["severity"]

        # ---------------------------------
        # 1. Semantic memory retrieval
        # ---------------------------------

        if self.memory is not None:

            semantic_query = (
                f"{diagnosis['incident']} "
                f"{diagnosis['cause']}"
            )

            semantic_plan = self.memory.semantic_recall(
                semantic_query
            )

            if semantic_plan:

                print(
                    "[PLANNING][SEMANTIC] "
                    "Using semantically similar incident"
                )

                memory_hint = semantic_plan

        # ---------------------------------
        # 2. Exact memory retrieval
        # ---------------------------------

        if memory_hint is None and memory_agent is not None:

            getter = getattr(
                memory_agent,
                "get_similar_incident",
                None
            )

            if callable(getter):

                memory_hint = getter(
                    diagnosis["incident"]
                )

        if memory_hint is not None:

            print(
                "[PLANNING][MEMORY] "
                "Using past successful strategy"
            )

            if is_plan(memory_hint):

                print(
                    "[PLANNING][MEMORY] "
                    "Memory hint available"
                )

        # ---------------------------------
        # 3. Qwen reasoning
        # ---------------------------------

        try:

            qwen_plan = (
                generate_plan_with_qwen(
                    incident=incident,
                    severity=severity,
                )
            )

            if qwen_plan:

                print(
                    "[PLANNING][QWEN] "
                    "Generated plan"
                )

                return qwen_plan

        except Exception as e:

            print(
                "[QWEN ERROR]",
                e
            )

        # ---------------------------------
        # 4. Symbolic fallback planning
        # ---------------------------------

        print(
            "[PLANNING][FALLBACK] "
            "Using symbolic planner"
        )

        if incident == "normal":

            return {
                "action": "restart_service",
                "target": "noop"
            }

        if incident == "intrusion":

            return {
                "action": "block_ip",
                "target": "attacker_ip"
            }

        if incident == "crash":

            return {
                "action": "restart_service",
                "target": "compute"
            }

        # overload

        if severity in ("high",):

            return {
                "action": "reroute_traffic",
                "target": "backup_server"
            }

        return {
            "action": "restart_service",
            "target": "lb"
        }