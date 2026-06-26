from __future__ import annotations

import sqlite3
import faiss
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from ..io_utils import now_iso
from ..types import Diagnosis, Plan


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  incident TEXT NOT NULL,
  cause TEXT NOT NULL,
  severity TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  success INTEGER NOT NULL,
  signature TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'agent'
);
CREATE INDEX IF NOT EXISTS idx_incidents_signature ON incidents(signature);
CREATE INDEX IF NOT EXISTS idx_incidents_incident ON incidents(incident);
"""


def _signature(d: Diagnosis) -> str:
    return f"{d['incident']}|{d['severity']}"


@dataclass
class MemoryAgent:

    sqlite_path: Path | str

    def __post_init__(self):

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.dimension = 384

        self.index = faiss.IndexFlatL2(
            self.dimension
        )

        self.semantic_memory = []

    def _connect(self) -> sqlite3.Connection:

        sqlite_path = str(self.sqlite_path)

        if sqlite_path != ":memory:":
            Path(sqlite_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

        conn = sqlite3.connect(sqlite_path)

        if sqlite_path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL;")

        conn.executescript(SCHEMA_SQL)

        self._migrate(conn)

        return conn

    def _migrate(
        self,
        conn: sqlite3.Connection
    ) -> None:

        cols = [
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(incidents)"
            ).fetchall()
        ]

        if "severity" not in cols:
            conn.execute(
                "ALTER TABLE incidents "
                "ADD COLUMN severity TEXT "
                "NOT NULL DEFAULT 'medium'"
            )

        if "source" not in cols:
            conn.execute(
                "ALTER TABLE incidents "
                "ADD COLUMN source TEXT "
                "NOT NULL DEFAULT 'agent'"
            )

        conn.commit()

    def remember(
        self,
        diagnosis: Diagnosis,
        plan: Plan,
        success: bool
    ) -> None:

        sig = _signature(diagnosis)

        # Semantic embedding storage

        memory_text = (
            f"{diagnosis['incident']} "
            f"{diagnosis['cause']} "
            f"{plan['action']}"
        )

        embedding = self.embedding_model.encode(
            [memory_text]
        )

        embedding = np.array(
            embedding,
            dtype=np.float32
        )

        self.index.add(embedding)

        self.semantic_memory.append({
            "diagnosis": diagnosis,
            "plan": plan,
            "success": success
        })

        print("[MEMORY][SEMANTIC] Stored embedding")

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO incidents (
                    ts,
                    incident,
                    cause,
                    severity,
                    action,
                    target,
                    success,
                    signature,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso(),
                    diagnosis["incident"],
                    diagnosis["cause"],
                    diagnosis["severity"],
                    plan["action"],
                    plan["target"],
                    1 if success else 0,
                    sig,
                    "agent",
                ),
            )

    def recall_plan_hint(
        self,
        diagnosis: Diagnosis
    ) -> Optional[Plan]:

        sig = _signature(diagnosis)

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT action, target
                FROM incidents
                WHERE signature = ?
                AND success = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (sig,),
            ).fetchone()

        if not row:
            return None

        action, target = row

        return {
            "action": action,
            "target": target
        }

    def get_similar_incident(
        self,
        incident_type: str
    ) -> Optional[Plan]:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT action, target
                FROM incidents
                WHERE incident = ?
                AND success = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (incident_type,),
            ).fetchone()

        if not row:
            return None

        action, target = row

        return {
            "action": action,
            "target": target
        }

    def get_recent_failure_action(
        self,
        incident_type: str
    ) -> Optional[str]:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT action
                FROM incidents
                WHERE incident = ?
                AND success = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (incident_type,),
            ).fetchone()

        if not row:
            return None

        (action,) = row

        return str(action)

    def semantic_recall(
        self,
        query: str
    ) -> Optional[Plan]:

        if len(self.semantic_memory) == 0:
            return None

        query_embedding = self.embedding_model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            1
        )

        idx = indices[0][0]

        if idx < 0:
            return None

        memory = self.semantic_memory[idx]

        print(
            f"[MEMORY][SEMANTIC] "
            f"Retrieved similar incident "
            f"(distance={distances[0][0]:.2f})"
        )

        return memory["plan"]