"""The archive: every message of one conversation, in SQLite."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, message_to_dict, messages_from_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_turn ON messages (conversation, turn, id);
"""


class MessageStore:
    """Stores the messages of one conversation, numbered by turn. Never truncates content."""

    def __init__(self, path: Path | None, conversation: str) -> None:
        self._db = sqlite3.connect(":memory:" if path is None else path)
        self._db.executescript(_SCHEMA)
        self._conversation = conversation

    def last_turn(self) -> int:
        row = self._db.execute(
            "SELECT MAX(turn) FROM messages WHERE conversation = ?", (self._conversation,)
        ).fetchone()
        return row[0] or 0

    def add(self, message: BaseMessage) -> int:
        """Store the message and return its turn number.

        A user message opens a new turn; any other message joins the current one. Messages that
        arrive before the first user message open turn 1.
        """
        last = self.last_turn()
        turn = last + 1 if isinstance(message, HumanMessage) or last == 0 else last
        with self._db:
            self._db.execute(
                "INSERT INTO messages (conversation, turn, type, data, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    self._conversation,
                    turn,
                    message.type,
                    json.dumps(message_to_dict(message)),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return turn

    def turns(self, first: int, last: int) -> dict[int, list[BaseMessage]]:
        """The messages of turns `first` through `last`, in order, grouped by turn."""
        rows = self._db.execute(
            "SELECT turn, data FROM messages WHERE conversation = ? AND turn BETWEEN ? AND ?"
            " ORDER BY id",
            (self._conversation, first, last),
        ).fetchall()
        grouped: dict[int, list[BaseMessage]] = {}
        for turn, data in rows:
            grouped.setdefault(turn, []).extend(messages_from_dict([json.loads(data)]))
        return grouped
