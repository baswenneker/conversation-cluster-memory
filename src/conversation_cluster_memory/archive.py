"""The archive: every message of one conversation, in SQLite."""

import json
import re
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)

from conversation_cluster_memory.summaries import ClusterSummary, Summary

# System messages belong to no turn: they are never counted, embedded or summarized.
SYSTEM_TURN = 0

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
-- A level-1 cluster holds turns; a higher level (a period) holds the clusters whose `parent`
-- it is. A cluster with a parent is shown in the window through that parent's summary.
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation TEXT NOT NULL,
    oldest_at TEXT NOT NULL DEFAULT '',
    level INTEGER NOT NULL DEFAULT 1,
    parent INTEGER REFERENCES clusters (id)
);
-- The primary key makes re-assigning a clustered turn impossible.
CREATE TABLE IF NOT EXISTS cluster_turns (
    conversation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    cluster INTEGER NOT NULL REFERENCES clusters (id),
    PRIMARY KEY (conversation, turn)
);
-- Every summary ever made; a cluster's current summary is its newest.
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation TEXT NOT NULL,
    cluster INTEGER NOT NULL REFERENCES clusters (id),
    text TEXT NOT NULL,
    anchors TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- An archived turn that is summarized links to the summary that covers it now.
CREATE TABLE IF NOT EXISTS summarized_turns (
    conversation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    summary INTEGER NOT NULL REFERENCES summaries (id),
    PRIMARY KEY (conversation, turn)
);
-- Keyword search (BM25) over each cluster's current summary and its anchors. Superseded
-- summaries are removed, so a topic summarized many times does not make its own words common.
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5 (
    text, subject, period, names, conversation UNINDEXED, cluster UNINDEXED
);
CREATE TRIGGER IF NOT EXISTS summaries_fts_insert AFTER INSERT ON summaries BEGIN
    DELETE FROM summaries_fts WHERE conversation = new.conversation AND cluster = new.cluster;
    INSERT INTO summaries_fts (rowid, text, subject, period, names, conversation, cluster) VALUES (
        new.id, new.text, json_extract(new.anchors, '$.subject'),
        json_extract(new.anchors, '$.period'), json_extract(new.anchors, '$.names'),
        new.conversation, new.cluster
    );
END;
-- Summaries stored before keyword search existed.
INSERT INTO summaries_fts (rowid, text, subject, period, names, conversation, cluster)
    SELECT id, text, json_extract(anchors, '$.subject'), json_extract(anchors, '$.period'),
        json_extract(anchors, '$.names'), conversation, cluster
    FROM summaries
    WHERE id IN (SELECT MAX(id) FROM summaries GROUP BY conversation, cluster)
        AND id NOT IN (SELECT rowid FROM summaries_fts);
"""


_TABLES = (
    "messages",
    "clusters",
    "cluster_turns",
    "summaries",
    "summarized_turns",
    "summaries_fts",
)


def stored_conversations(path: Path) -> list[str]:
    """The id of every conversation with a row in the archive file. Every table counts: a
    round that commits after its conversation was deleted leaves rows without messages."""
    union = " UNION ".join(f"SELECT conversation FROM {table}" for table in _TABLES)
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute(union).fetchall()
    return [conversation for (conversation,) in rows]


def delete_conversations(path: Path, conversations: list[str]) -> None:
    """Remove every row of these conversations from the archive file, in one transaction."""
    marks = ",".join("?" * len(conversations))
    with closing(sqlite3.connect(path)) as db:
        # Overwrite the freed pages, so deleted text cannot be read back from the file.
        db.execute("PRAGMA secure_delete = ON")
        with db:
            for table in _TABLES:
                db.execute(f"DELETE FROM {table} WHERE conversation IN ({marks})", conversations)


class MessageStore:
    """Stores the messages of one conversation, numbered by turn. Never truncates content."""

    def __init__(self, path: Path | None, conversation: str) -> None:
        # Shared across threads; the lock serialises every use of the connection.
        # isolation_level=None: no implicit transactions; add() opens its own with BEGIN.
        self._db = sqlite3.connect(
            ":memory:" if path is None else path, check_same_thread=False, isolation_level=None
        )
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(clusters)")}
        if "level" not in columns:  # an archive stored before period summaries existed
            self._db.executescript(
                "ALTER TABLE clusters ADD COLUMN level INTEGER NOT NULL DEFAULT 1;"
                " ALTER TABLE clusters ADD COLUMN parent INTEGER REFERENCES clusters (id);"
            )
        self._conversation = conversation

    def last_turn(self) -> int:
        with self._lock:
            return self._last_turn()

    def _last_turn(self) -> int:
        row = self._db.execute(
            "SELECT MAX(turn) FROM messages WHERE conversation = ?", (self._conversation,)
        ).fetchone()
        return row[0] or 0

    def _has_user_message(self, turn: int) -> bool:
        return (
            self._db.execute(
                "SELECT 1 FROM messages WHERE conversation = ? AND turn = ? AND type = 'human'",
                (self._conversation, turn),
            ).fetchone()
            is not None
        )

    def add(self, message: BaseMessage) -> int:
        """Store the message and return its turn number.

        A user message opens a new turn; any other message joins the current one. A system
        message gets SYSTEM_TURN; other messages before the first user message join turn 1.
        """
        with self._lock, self._db:
            # Take the write lock before reading, so another connection to the same file
            # cannot compute the same turn number at the same time.
            self._db.execute("BEGIN IMMEDIATE")
            last = self._last_turn()
            opens = isinstance(message, HumanMessage) and self._has_user_message(last)
            turn = last + 1 if opens or last == 0 else last
            if isinstance(message, SystemMessage):
                turn = SYSTEM_TURN
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
        with self._lock:
            rows = self._db.execute(
                "SELECT turn, data FROM messages WHERE conversation = ? AND turn BETWEEN ? AND ?"
                " ORDER BY id",
                (self._conversation, first, last),
            ).fetchall()
        grouped: dict[int, list[BaseMessage]] = {}
        for turn, data in rows:
            grouped.setdefault(turn, []).extend(messages_from_dict([json.loads(data)]))
        return grouped

    def clustered_turns(self) -> set[int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT turn FROM cluster_turns WHERE conversation = ?", (self._conversation,)
            ).fetchall()
        return {turn for (turn,) in rows}

    def clusters(self) -> list[tuple[int, str, tuple[int, ...]]]:
        """Every cluster as (id, oldest_at, turns), oldest first; ties by id."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id, c.oldest_at, t.turn FROM clusters c"
                " JOIN cluster_turns t ON t.cluster = c.id AND t.conversation = c.conversation"
                " WHERE c.conversation = ? ORDER BY c.oldest_at, c.id, t.turn",
                (self._conversation,),
            ).fetchall()
        grouped: dict[int, tuple[str, list[int]]] = {}
        for cluster, oldest_at, turn in rows:
            grouped.setdefault(cluster, (oldest_at, []))[1].append(turn)
        return [(c, oldest_at, tuple(turns)) for c, (oldest_at, turns) in grouped.items()]

    def commit_round(
        self,
        groups: list[tuple[int | None, list[int]]],
        summaries: list[tuple[int | None, int | None, Summary, tuple[int, ...]]],
    ) -> list[int]:
        """Commit a round in one transaction: add each group of turns to its cluster (None: a
        new one), then store each summary with the turns it covers.

        A summary names its committed cluster, or else (cluster None) the index of the group
        that creates its cluster. Returns the cluster id per group.
        """
        ids = []
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            for cluster, turns in groups:
                if cluster is None:
                    cluster = self._db.execute(
                        "INSERT INTO clusters (conversation) VALUES (?)", (self._conversation,)
                    ).lastrowid
                    assert cluster is not None
                self._db.executemany(
                    "INSERT INTO cluster_turns (conversation, turn, cluster) VALUES (?, ?, ?)",
                    [(self._conversation, turn, cluster) for turn in turns],
                )
                # A backfilled turn may be older than every current member.
                self._db.execute(
                    "UPDATE clusters SET oldest_at = (SELECT MIN(m.created_at) FROM messages m"
                    " JOIN cluster_turns t ON t.conversation = m.conversation AND t.turn = m.turn"
                    " WHERE t.conversation = ? AND t.cluster = ?) WHERE id = ?",
                    (self._conversation, cluster, cluster),
                )
                ids.append(cluster)
            for cluster, group, summary, turns in summaries:
                if cluster is None:
                    assert group is not None
                    cluster = ids[group]
                # A grown cluster shows its own new summary again; the period summary that
                # merged its older one keeps that text. Re-merging the period is the upgrade
                # if returning topics make the window repeat itself.
                self._db.execute("UPDATE clusters SET parent = NULL WHERE id = ?", (cluster,))
                self._save_summary(cluster, summary, turns)
        return ids

    def periods(self) -> list[tuple[int, str, tuple[int, ...]]]:
        """Every period (a cluster above level 1) as (id, oldest_at, turns), oldest first; its
        turns are those of the level-1 clusters below it."""
        with self._lock:
            rows = self._db.execute(
                "WITH RECURSIVE tree (top, id) AS ("
                " SELECT id, id FROM clusters WHERE conversation = ? AND level > 1"
                " UNION ALL SELECT tree.top, c.id FROM clusters c JOIN tree ON c.parent = tree.id)"
                " SELECT p.id, p.oldest_at, t.turn FROM tree"
                " JOIN clusters p ON p.id = tree.top"
                " JOIN cluster_turns t ON t.cluster = tree.id AND t.conversation = p.conversation"
                " ORDER BY p.oldest_at, p.id, t.turn",
                (self._conversation,),
            ).fetchall()
        grouped: dict[int, tuple[str, list[int]]] = {}
        for period, oldest_at, turn in rows:
            grouped.setdefault(period, (oldest_at, []))[1].append(turn)
        return [(p, oldest_at, tuple(turns)) for p, (oldest_at, turns) in grouped.items()]

    def merge(self, clusters: list[int], summary: Summary) -> int:
        """Store a period: one cluster a level above the highest of `clusters`, with `summary`,
        as their parent. Returns its id."""
        marks = ",".join("?" * len(clusters))
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            period = self._db.execute(
                "INSERT INTO clusters (conversation, oldest_at, level)"
                f" SELECT ?, MIN(oldest_at), MAX(level) + 1 FROM clusters WHERE id IN ({marks})",
                (self._conversation, *clusters),
            ).lastrowid
            assert period is not None
            self._db.execute(
                f"UPDATE clusters SET parent = ? WHERE id IN ({marks})", (period, *clusters)
            )
            self._save_summary(period, summary, ())
        return period

    def period(self, turns: list[int]) -> str:
        """The time span of these turns' messages, as "first to last" ISO times."""
        first, last = self.time_range(turns)
        return f"{first} to {last}"

    def time_range(self, turns: list[int]) -> tuple[str, str]:
        """The ISO times of the first and the last message of these turns."""
        marks = ",".join("?" * len(turns))
        with self._lock:
            first, last = self._db.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM messages"
                f" WHERE conversation = ? AND turn IN ({marks})",
                (self._conversation, *turns),
            ).fetchone()
        return first, last

    def keyword_search(self, query: str, min_score: float) -> list[tuple[int, float]]:
        """Clusters whose current summary or anchors contain a word of the query, as
        (cluster, BM25 score), best first; only scores of at least `min_score`.

        BM25 statistics span the current summaries of every conversation in the file, so a
        word common in other conversations scores lower here too.
        """
        words = re.findall(r"\w+", query)
        if not words:
            return []
        match = " OR ".join(f'"{word}"' for word in words)
        with self._lock:
            rows = self._db.execute(
                "SELECT cluster, -bm25(summaries_fts) AS score FROM summaries_fts"
                " WHERE summaries_fts MATCH ? AND conversation = ? ORDER BY score DESC, cluster",
                (match, self._conversation),
            ).fetchall()
        return [(cluster, score) for cluster, score in rows if score >= min_score]

    def _save_summary(self, cluster: int, summary: Summary, turns: tuple[int, ...]) -> None:
        """Store the cluster's new summary and link its turns to it (inside a transaction)."""
        anchors = {"subject": summary.subject, "period": summary.period, "names": summary.names}
        summary_id = self._db.execute(
            "INSERT INTO summaries (conversation, cluster, text, anchors, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                self._conversation,
                cluster,
                summary.text,
                json.dumps(anchors),
                datetime.now(UTC).isoformat(),
            ),
        ).lastrowid
        self._db.executemany(
            "INSERT OR REPLACE INTO summarized_turns (conversation, turn, summary)"
            " VALUES (?, ?, ?)",
            [(self._conversation, turn, summary_id) for turn in turns],
        )

    def summaries(self, in_window: bool = False) -> dict[int, ClusterSummary]:
        """Each summarized cluster's current summary, in time order (by oldest member), of
        every level; with `in_window` only those the window shows (no parent)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT s.cluster, s.id, s.text, s.anchors FROM summaries s"
                " JOIN clusters c ON c.id = s.cluster"
                " WHERE s.id IN (SELECT MAX(id) FROM summaries WHERE conversation = ?"
                " GROUP BY cluster)"
                + (" AND c.parent IS NULL" if in_window else "")
                + " ORDER BY c.oldest_at, c.id",
                (self._conversation,),
            ).fetchall()
        return {
            cluster: ClusterSummary(summary_id, text, **json.loads(anchors))
            for cluster, summary_id, text, anchors in rows
        }

    def summarized_turns(self) -> dict[int, int]:
        """Each summarized turn with the id of the summary that covers it."""
        with self._lock:
            rows = self._db.execute(
                "SELECT turn, summary FROM summarized_turns WHERE conversation = ?",
                (self._conversation,),
            ).fetchall()
        return dict(rows)
