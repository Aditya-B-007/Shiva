import sqlite3
import re  # Fixed: Added missing import
from typing import List, Dict, Any

class Hippocampus:
    def __init__(self, db_path: str = "ai_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS long_term_memory USING fts5(
                user_id UNINDEXED,
                fact
            )
        """)
        
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")
        self.conn.commit()

    def save_chat_message(self, session_id: str, role: str, content: str):
        self.cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.conn.commit()

    def save_long_term_fact(self, user_id: str, fact_text: str):
        """Stores a permanent text fact directly into the FTS5 table without vectors."""
        self.cursor.execute(
            "INSERT INTO long_term_memory(user_id, fact) VALUES (?, ?)",
            (user_id, fact_text)
        )
        self.conn.commit()

    def retrieve_context(self, user_id: str, session_id: str, current_query: str, limit_chat: int = 5, limit_facts: int = 3) -> Dict[str, Any]:
        self.cursor.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit_chat)
        )
        recent_chat = [{"role": r, "content": c} for r, c in self.cursor.fetchall()][::-1]
        
        keywords = re.findall(r'\b\w+\b', current_query)
        fts_search_query = " OR ".join(keywords) if keywords else ""

        relevant_facts = []
        if fts_search_query:
            self.cursor.execute("""
                SELECT fact FROM long_term_memory 
                WHERE user_id = ? AND long_term_memory MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (user_id, fts_search_query, limit_facts))
        
            relevant_facts = [row[0] for row in self.cursor.fetchall()]
        return {
            "recent_chat_history": recent_chat,
            "relevant_facts": relevant_facts
        }

    def close(self):
        self.conn.close()
