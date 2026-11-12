"""The public entry point: ConversationClusterMemory."""

import logging
import uuid
import weakref
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, MessageLikeRepresentation, convert_to_messages

from conversation_cluster_memory.archive import MessageStore
from conversation_cluster_memory.embeddings import (
    EmbeddingText,
    default_embedding_text,
    default_embeddings,
    embedding_model_id,
)
from conversation_cluster_memory.keys import KeyLike, collection_name, storage_id
from conversation_cluster_memory.vectors import VectorStore, chroma_client

logger = logging.getLogger(__name__)


class ConversationClusterMemory:
    """Memory for one conversation: stores every message and returns the window to send."""

    def __init__(
        self,
        key: KeyLike,
        *,
        path: str | Path | None = None,
        embeddings: Embeddings | None = None,
        keep_recent_turns: int = 10,
        embedding_text: EmbeddingText = default_embedding_text,
    ) -> None:
        self.keep_recent_turns = keep_recent_turns
        self.embeddings = embeddings or default_embeddings()
        self.embedding_text = embedding_text
        name = collection_name(key)
        if path is None:
            self.archive = MessageStore(None, storage_id(key))
            # In-process Chroma clients share their collections, so an in-memory memory gets a
            # name of its own: two in-memory memories for one key must not see each other.
            name = f"{name}-{uuid.uuid4().hex[:8]}"
            client = chroma_client(None)
            weakref.finalize(self, client.delete_collection, name)
        else:
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            self.archive = MessageStore(directory / "archive.sqlite3", storage_id(key))
            client = chroma_client(str(directory / "chroma"))
        self.vectors = VectorStore(client, name, embedding_model_id(self.embeddings))

    def add_message(self, message: MessageLikeRepresentation) -> None:
        (converted,) = convert_to_messages([message])
        self.archive.add(converted)
        self._embed_turns_outside_window()

    def _embed_turns_outside_window(self) -> None:
        """Embed every turn outside the recent window that has no vector yet (incl. backfill)."""
        last_outside = self.archive.last_turn() - self.keep_recent_turns
        if last_outside < 1:
            return
        missing = set(range(1, last_outside + 1)) - self.vectors.turns()
        if not missing:
            return
        turns = self.archive.turns(min(missing), max(missing))
        numbers = sorted(missing & turns.keys())
        texts = [self.embedding_text(turns[n]) for n in numbers]
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception:
            # The messages are already stored; the next add_message backfills these turns.
            logger.exception("embedding turns %s failed; retrying on the next message", numbers)
            return
        self.vectors.add(dict(zip(numbers, vectors, strict=True)))

    def window(self) -> list[BaseMessage]:
        last = self.archive.last_turn()
        recent = self.archive.turns(last - self.keep_recent_turns + 1, last)
        return [message for turn in sorted(recent) for message in recent[turn]]
