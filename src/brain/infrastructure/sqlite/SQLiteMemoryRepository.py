import json
import dataclasses
from typing import List, Optional
from ...gate.interfaces.MemoryRepository import MemoryRepository
from ...gate.models.Memory import Memory
from ...emotionalHandlerAndStore.emotionalContract import EmotionDTO
from .SQLiteManager import SQLiteManager

class SQLiteMemoryRepository(MemoryRepository):

    def __init__(self, manager: SQLiteManager) -> None:
        self.manager = manager

    def _serialize_emotion(self, emotion: Optional[EmotionDTO]) -> Optional[str]:
        if emotion is None:
            return None
        return json.dumps(dataclasses.asdict(emotion), default=str)

    def _deserialize_emotion(self, emotion_str: Optional[str]) -> Optional[EmotionDTO]:
        if not emotion_str:
            return None
        try:
            data = json.loads(emotion_str)
            if "timestamp" in data and isinstance(data["timestamp"], str):
                from datetime import datetime
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    pass
            return EmotionDTO(**data)
        except Exception:
            return None

    def save(self, memory: Memory) -> None:
        conn = self.manager.get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories (id, content, confidence, created_at, promoted_at, emotion)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.confidence,
                memory.created_at,
                memory.promoted_at,
                self._serialize_emotion(memory.emotion)
            )
        )

    def load(self, memory_id: str) -> Optional[Memory]:
        conn = self.manager.get_connection()
        row = conn.execute(
            "SELECT id, content, confidence, created_at, promoted_at, emotion FROM memories WHERE id = ?",
            (memory_id,)
        ).fetchone()
        if not row:
            return None
        return Memory(
            id=row["id"],
            content=row["content"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            promoted_at=row["promoted_at"],
            emotion=self._deserialize_emotion(row["emotion"])
        )

    def update(self, memory: Memory) -> None:
        conn = self.manager.get_connection()
        conn.execute(
            """
            UPDATE memories
            SET content = ?, confidence = ?, promoted_at = ?, emotion = ?
            WHERE id = ?
            """,
            (
                memory.content,
                memory.confidence,
                memory.promoted_at,
                self._serialize_emotion(memory.emotion),
                memory.id
            )
        )

    def delete(self, memory_id: str) -> None:
        conn = self.manager.get_connection()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def load_all(self) -> List[Memory]:
        conn = self.manager.get_connection()
        rows = conn.execute("SELECT id, content, confidence, created_at, promoted_at, emotion FROM memories").fetchall()
        return [
            Memory(
                id=row["id"],
                content=row["content"],
                confidence=row["confidence"],
                created_at=row["created_at"],
                promoted_at=row["promoted_at"],
                emotion=self._deserialize_emotion(row["emotion"])
            )
            for row in rows
        ]
