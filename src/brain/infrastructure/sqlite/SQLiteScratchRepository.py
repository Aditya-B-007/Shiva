import json
import dataclasses
from datetime import datetime
from typing import List, Optional
from ...gate.interfaces.ScratchRepository import ScratchRepository
from ...gate.models.ScratchEntry import ScratchEntry
from ...emotionalHandlerAndStore.emotionalContract import EmotionDTO
from .SQLiteManager import SQLiteManager

class SQLiteScratchRepository(ScratchRepository):

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

    def _serialize_datetime(self, value: datetime | float | int | str | None) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(float(value)).isoformat()
        return str(value)

    def _deserialize_datetime(self, value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(float(value))
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.utcfromtimestamp(float(value))
                except ValueError:
                    return None
        return None

    def save(self, entry: ScratchEntry) -> None:
        conn = self.manager.get_connection()
        conn.execute(
            """
            INSERT INTO scratchpad (id, content, confidence, created_at, updated_at, emotion)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                entry.confidence,
                self._serialize_datetime(entry.created_at),
                self._serialize_datetime(entry.updated_at),
                self._serialize_emotion(entry.emotion)
            )
        )

    def load(self, entry_id: str) -> Optional[ScratchEntry]:
        conn = self.manager.get_connection()
        row = conn.execute(
            "SELECT id, content, confidence, created_at, updated_at, emotion FROM scratchpad WHERE id = ?",
            (entry_id,)
        ).fetchone()
        if not row:
            return None
        return ScratchEntry(
            id=row["id"],
            content=row["content"],
            confidence=row["confidence"],
            created_at=self._deserialize_datetime(row["created_at"]),
            updated_at=self._deserialize_datetime(row["updated_at"]),
            emotion=self._deserialize_emotion(row["emotion"])
        )

    def update(self, entry: ScratchEntry) -> None:
        conn = self.manager.get_connection()
        conn.execute(
            """
            UPDATE scratchpad
            SET content = ?, confidence = ?, updated_at = ?, emotion = ?
            WHERE id = ?
            """,
            (
                entry.content,
                entry.confidence,
                entry.updated_at,
                self._serialize_emotion(entry.emotion),
                entry.id
            )
        )

    def delete(self, entry_id: str) -> None:
        conn = self.manager.get_connection()
        conn.execute("DELETE FROM scratchpad WHERE id = ?", (entry_id,))

    def clear(self) -> None:
        conn = self.manager.get_connection()
        conn.execute("DELETE FROM scratchpad")

    def load_all(self) -> List[ScratchEntry]:
        conn = self.manager.get_connection()
        rows = conn.execute("SELECT id, content, confidence, created_at, updated_at, emotion FROM scratchpad").fetchall()
        return [
            ScratchEntry(
                id=row["id"],
                content=row["content"],
                confidence=row["confidence"],
                created_at=self._deserialize_datetime(row["created_at"]),
                updated_at=self._deserialize_datetime(row["updated_at"]),
                emotion=self._deserialize_emotion(row["emotion"])
            )
            for row in rows
        ]
