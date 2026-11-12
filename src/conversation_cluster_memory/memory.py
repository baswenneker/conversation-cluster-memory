"""The public entry point: ConversationClusterMemory."""

import logging
import threading
import uuid
import weakref
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    convert_to_messages,
)

from conversation_cluster_memory.archive import SYSTEM_TURN, MessageStore
from conversation_cluster_memory.clustering import (
    AgglomerativeClustering,
    Cluster,
    ClusteringStrategy,
    centroid,
    cosine,
)
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
        min_batch_turns: int = 8,
        cluster_every_n_turns: int = 5,
        similarity_threshold: float = 0.8,
        clustering: ClusteringStrategy | None = None,
    ) -> None:
        if keep_recent_turns < 1:
            # The newest turn is still open; embedding it would freeze an incomplete vector.
            raise ValueError(f"keep_recent_turns must be at least 1; got {keep_recent_turns}")
        if min_batch_turns < 1 or cluster_every_n_turns < 1:
            raise ValueError("min_batch_turns and cluster_every_n_turns must be at least 1")
        if not -1 <= similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold is a cosine similarity; got {similarity_threshold}"
            )
        self.keep_recent_turns = keep_recent_turns
        self.min_batch_turns = min_batch_turns
        self.cluster_every_n_turns = cluster_every_n_turns
        self.similarity_threshold = similarity_threshold
        self.clustering = clustering or AgglomerativeClustering()
        # One embedding and clustering pass at a time per memory object, so a turn is embedded
        # and clustered once. Memory objects in other processes sharing the path are not
        # covered; a lock in SQLite would be the upgrade if that ever happens in practice.
        self._lock = threading.Lock()
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
            weakref.finalize(self, client.delete_collection, f"{name}-centroids")
        else:
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            self.archive = MessageStore(directory / "archive.sqlite3", storage_id(key))
            client = chroma_client(str(directory / "chroma"))
        model_id = embedding_model_id(self.embeddings)
        self.vectors = VectorStore(client, name, model_id)
        self.centroids = VectorStore(client, f"{name}-centroids", model_id)

    def add_message(self, message: MessageLikeRepresentation) -> None:
        (converted,) = convert_to_messages([message])
        turn = self.archive.add(converted)
        with self._lock:
            self._embed_turns_outside_window()
            if isinstance(converted, HumanMessage) and turn % self.cluster_every_n_turns == 0:
                self._cluster_batch()

    def clusters(self) -> list[Cluster]:
        """Every cluster, oldest first."""
        stored = self.archive.clusters()
        centroids = self.centroids.get([cluster for cluster, _, _ in stored])
        return [
            Cluster(c, turns, oldest_at, centroids.get(c) or self._centroid(turns))
            for c, oldest_at, turns in stored
        ]

    def _centroid(self, turns: tuple[int, ...] | list[int]) -> list[float]:
        vectors = self.vectors.get(list(turns))
        return centroid([vectors[t] for t in turns])

    def _cluster_batch(self) -> None:
        """Cluster the unclustered embedded turns once there are `min_batch_turns` of them.

        Only turns outside the recent window have a vector, so the batch lies outside it.
        """
        batch = sorted(self.vectors.turns() - self.archive.clustered_turns())
        if len(batch) < self.min_batch_turns:
            return
        vectors = self.vectors.get(batch)
        groups = self.clustering.cluster(
            batch, [vectors[t] for t in batch], self.similarity_threshold
        )
        existing = self.clusters()  # oldest first, so on a tie the oldest wins
        plan: list[tuple[int | None, list[int]]] = []
        for group in groups:
            batch_centroid = centroid([vectors[t] for t in group])
            best, best_similarity = None, self.similarity_threshold
            for cluster in existing:
                similarity = cosine(batch_centroid, cluster.centroid)
                if similarity > best_similarity or (best is None and similarity == best_similarity):
                    best, best_similarity = cluster.id, similarity
            plan.append((best, group))
        ids = self.archive.assign(plan)
        # Written after the SQLite commit; a missing centroid is recomputed from its turns.
        members = {cluster: turns for cluster, _, turns in self.archive.clusters()}
        self.centroids.add({c: self._centroid(members[c]) for c in set(ids)})

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
            self.vectors.add(dict(zip(numbers, vectors, strict=True)))
        except Exception:
            # The message is already stored; the next add_message backfills these turns.
            logger.exception("embedding turns %s failed; retrying on the next message", numbers)

    def window(self) -> list[BaseMessage]:
        """System messages first, then the last `keep_recent_turns` turns verbatim."""
        last = self.archive.last_turn()
        system = self.archive.turns(SYSTEM_TURN, SYSTEM_TURN).get(SYSTEM_TURN, [])
        recent = self.archive.turns(max(1, last - self.keep_recent_turns + 1), last)
        return system + [message for turn in sorted(recent) for message in recent[turn]]
