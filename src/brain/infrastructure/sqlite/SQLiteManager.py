import sqlite3
import threading
from typing import Optional
from dotenv import load_dotenv
from ...gate.interfaces.TransactionManager import TransactionManager

class SQLiteManager(TransactionManager):

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()
        # Initialize dotenv just in case
        load_dotenv()

    def get_connection(self) -> sqlite3.Connection:
        """Returns the thread-local SQLite connection, creating it if necessary."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent reading/writing support
            conn.execute("PRAGMA journal_mode=WAL;")
            self._local.connection = conn
        return self._local.connection

    def close(self) -> None:
        """Closes the thread-local SQLite connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None

    def begin(self) -> None:
        """Starts an immediate write transaction on the current thread's connection."""
        conn = self.get_connection()
        conn.execute("BEGIN IMMEDIATE TRANSACTION;")

    def commit(self) -> None:
        """Commits the current transaction on the current thread's connection."""
        conn = self.get_connection()
        conn.commit()

    def rollback(self) -> None:
        """Rolls back the current transaction on the current thread's connection."""
        conn = self.get_connection()
        conn.rollback()
