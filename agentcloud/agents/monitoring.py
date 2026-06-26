from __future__ import annotations
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Optional

from ..types import AnomalyAlert, LogEvent


def _parse_ts(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


@dataclass
class MonitoringAgent:
    cpu_high_threshold: int = 85
    failed_login_burst: int = 5
    failed_login_window_seconds: int = 20

    _failed_login_ts: Deque[float] = None  # type: ignore[assignment]
    _cpu_history: Deque[int] = None  # type: ignore[assignment]
    def __post_init__(self) -> None:
        if self._failed_login_ts is None:
            self._failed_login_ts = deque()
        if self._cpu_history is None:
            self._cpu_history = deque(maxlen=50)
    def observe(self, event: LogEvent) -> Optional[AnomalyAlert]:
        msg = (event.get("message") or "").lower()

        # HDFS detection (NEW)
        if "error" in msg or "exception" in msg or "fail" in msg:
            return {"status": "anomaly_detected", "type": "crash"}

        if "terminating" in msg:
            return {"status": "anomaly_detected", "type": "crash"}
        cpu = event.get("cpu")

        if isinstance(cpu, int):
            self._cpu_history.append(cpu)

            # Keep bounded history
            if len(self._cpu_history) > 100:
                self._cpu_history.popleft()

            if len(self._cpu_history) >= 10:

                mean_cpu = statistics.mean(
                    self._cpu_history
                )

                std_cpu = statistics.stdev(
                    self._cpu_history
                )

                if std_cpu > 0:

                    z_score = (
                        (cpu - mean_cpu)
                        / std_cpu
                    )

                    # Stronger threshold
                    if z_score > 3.0 and cpu >= 85:

                        # Reduce spam
                        if not hasattr(
                            self,
                            "_last_overload_alert"
                        ):
                            self._last_overload_alert = 0

                        current_ts = _parse_ts(
                            event.get("ts", "")
                        )

                        cooldown = (
                            current_ts
                            - self._last_overload_alert
                        )

                        if cooldown > 10:

                            self._last_overload_alert = (
                                current_ts
                            )

                            print(
                                f"[MONITOR][ANOMALY] "
                                f"CPU spike detected | "
                                f"cpu={cpu} "
                                f"mean={mean_cpu:.2f} "
                                f"z={z_score:.2f}"
                            )

                            return {
                                "status": "anomaly_detected",
                                "type": "overload"
                            }

        if event.get("level") in ("ERROR", "ALERT") and event.get("event") == "error":
            msg = (event.get("message") or "").lower()
            if "crash" in msg or "segfault" in msg or "process down" in msg:
                return {"status": "anomaly_detected", "type": "crash"}

        if event.get("event") == "auth" and event.get("status") == "failed":
            ts = _parse_ts(event.get("ts", ""))
            self._failed_login_ts.append(ts)
            cutoff = ts - self.failed_login_window_seconds
            while self._failed_login_ts and self._failed_login_ts[0] < cutoff:
                self._failed_login_ts.popleft()
            if len(self._failed_login_ts) >= self.failed_login_burst:
                return {"status": "anomaly_detected", "type": "intrusion"}

        return None

