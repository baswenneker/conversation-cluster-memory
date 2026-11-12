"""The public entry point: ConversationClusterMemory."""

import asyncio
import logging
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    convert_to_messages,
)

from conversation_cluster_memory.archive import SYSTEM_TURN, MessageStore
from conversation_cluster_memory.clustering import (
    EPSILON,
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
    openai_api_key,
    tool_result_headers,
)
from conversation_cluster_memory.keys import KeyLike, collection_name, storage_id
from conversation_cluster_memory.summaries import (
    DEFAULT_SUMMARY_MODEL,
    POOR_COMPRESSION_RATIO,
    CompactionReport,
    SummaryReport,
    memory_text,
    summarize,
    token_counter,
    transcript,
)
from conversation_cluster_memory.vectors import VectorStore, chroma_client

logger = logging.getLogger(__name__)

type AutoCompact = Literal["background", "inline", "off"]
type ResummarizeStrategy = Literal["incremental", "full"]


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
        summary_model: str | BaseChatModel | None = None,
        resummarize_strategy: ResummarizeStrategy = "incremental",
        max_tool_result_tokens: int = 2000,
        auto_compact: AutoCompact = "background",
        on_compact: Callable[[CompactionReport], object] | None = None,
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
        if resummarize_strategy not in ("incremental", "full"):
            raise ValueError(
                f"resummarize_strategy is incremental or full; got {resummarize_strategy!r}"
            )
        if auto_compact not in ("background", "inline", "off"):
            raise ValueError(f"auto_compact is background, inline or off; got {auto_compact!r}")
        self.clustering = clustering or AgglomerativeClustering()
        self.resummarize_strategy = resummarize_strategy
        self.max_tool_result_tokens = max_tool_result_tokens
        self.auto_compact = auto_compact
        self.on_compact = on_compact
        if isinstance(summary_model, BaseChatModel):
            self._summary_model: BaseChatModel | None = summary_model
            self.summary_model_name = embedding_model_id(summary_model)
        else:
            # Built on first use, so a memory that never summarizes needs no OpenAI key.
            self._summary_model = None
            self.summary_model_name = (
                summary_model or os.environ.get("OPENAI_SUMMARY_MODEL") or DEFAULT_SUMMARY_MODEL
            )
        # One compaction round at a time per memory object, so a turn is embedded, clustered
        # and summarized once. Memory objects in other processes sharing the path are not
        # covered; a lock in SQLite would be the upgrade if that ever happens in practice.
        self._round_lock = threading.Lock()
        # Guards the background worker's state: whether it runs, whether work came in meanwhile,
        # and whether a cluster check is due. True at start: a round interrupted before a
        # restart is redone on the first add_message.
        self._state = threading.Condition()
        self._running = False
        self._again = False
        self._check_due = True
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
        """Store the message and, unless `auto_compact` is off, start a round if there is work.

        `background` returns before the round ends; `inline` returns after it.
        """
        (converted,) = convert_to_messages([message])
        turn = self.archive.add(converted)
        if self.auto_compact == "off":
            return
        with self._state:
            if isinstance(converted, HumanMessage) and turn % self.cluster_every_n_turns == 0:
                self._check_due = True
            if self._running:
                self._again = True  # the worker starts another round when this one ends
                return
            if not self._has_work():
                self._check_due = False  # the check found too few turns
                return
            if self.auto_compact == "background":
                self._running = True
        if self.auto_compact == "inline":
            self._compact(check=None)
            return
        # A daemon thread: a process that exits mid-round leaves the previous state, and the
        # round is redone after a restart.
        threading.Thread(target=self._work, daemon=True).start()

    def compact(self) -> CompactionReport:
        """Run one compaction round now (with a cluster check), after any running round."""
        return self._compact(check=True)

    async def acompact(self) -> CompactionReport:
        return await asyncio.to_thread(self.compact)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no round is running; False if `timeout` seconds passed first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state:
            if not self._state.wait_for(lambda: not self._running, timeout):
                return False
        left = -1 if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._round_lock.acquire(timeout=left):
            return False
        self._round_lock.release()
        return True

    def _work(self) -> None:
        """The background worker: rounds until no work came in during the last one."""
        idle = False
        try:
            while not idle:
                self._compact(check=None)
                with self._state:
                    # Deciding to stop and marking the worker idle happen under one lock, so
                    # an add_message in between cannot set `_again` for a worker that is gone.
                    idle = not (self._again and self._has_work())
                    if idle:
                        self._check_due = False  # a check that found too few turns
                    self._again = False
                    if idle:
                        self._running = False
                        self._state.notify_all()
        finally:
            if not idle:  # the round raised: let the next add_message start a new worker
                with self._state:
                    self._running = False
                    self._state.notify_all()

    def _has_work(self) -> bool:
        if self._missing_vectors():
            return True
        if self._check_due and self._unclustered_count() >= self.min_batch_turns:
            return True
        return bool(self._unsummarized())

    def _compact(self, check: bool | None) -> CompactionReport:
        """One round: embed, cluster when a check is due (or `check`), summarize, report."""
        report = CompactionReport()
        with self._round_lock:
            if check is None:
                with self._state:
                    check, self._check_due = self._check_due, False
            report.window_tokens_before = self._tokens(self.window())
            try:
                self._embed_turns_outside_window()
                if check:
                    report.clustered_turns = self._cluster_batch()
                self._summarize(report)
            except Exception as error:
                # Committed work stays; what is left is retried by the next round.
                logger.exception("compaction round failed; retrying in the next round")
                report.errors.append(repr(error))
            report.window_tokens_after = self._tokens(self.window())
        logger.info("compaction round: %s", report)
        if self.on_compact is not None:
            try:
                self.on_compact(report)
            except Exception:
                logger.exception("on_compact callback failed")
        return report

    def _unsummarized(self) -> dict[int, tuple[list[int], tuple[int, ...]]]:
        """Clusters with turns no summary covers yet, as (new turns, all turns), oldest first."""
        summarized = self.archive.summarized_turns()
        result = {}
        for cluster, _, turns in self.archive.clusters():
            if new := [t for t in turns if t not in summarized]:
                result[cluster] = (new, turns)
        return result

    def _summarize(self, report: CompactionReport) -> None:
        """Summarize every new or grown cluster; a failed one stays unsummarized for a retry."""
        current = self.archive.summaries()
        for cluster, (new, turns) in self._unsummarized().items():
            previous = current.get(cluster) if self.resummarize_strategy == "incremental" else None
            inputs = new if previous is not None else list(turns)
            stored = self.archive.turns(min(turns), max(turns))
            # The summary model reads tool results in full, so their facts reach the summary;
            # the token counts measure the turns as the window would hold them.
            shortened = {t: self._shorten(stored[t]) for t in turns}
            try:
                summary, usage = summarize(
                    self._model(),
                    transcript([stored[t] for t in inputs]),
                    previous.text if previous is not None else None,
                    self.archive.period(list(turns)),
                )
            except Exception as error:
                logger.exception("summarizing cluster %s failed; retrying next round", cluster)
                report.errors.append(f"cluster {cluster}: {error!r}")
                continue
            report.input_tokens += usage.get("input_tokens", 0)
            report.output_tokens += usage.get("output_tokens", 0)
            self.archive.save_summary(cluster, summary, turns)
            original = self._tokens([m for t in turns for m in shortened[t]])
            replaced = self._tokens([m for t in new for m in shortened[t]])
            report.window_tokens_before += replaced
            summary_tokens = self._count(summary.text)
            report.summaries.append(
                SummaryReport(
                    cluster,
                    turns,
                    original,
                    summary_tokens,
                    summary_tokens > POOR_COMPRESSION_RATIO * original,
                )
            )

    def _model(self) -> BaseChatModel:
        if self._summary_model is None:
            from langchain_openai import ChatOpenAI

            self._summary_model = ChatOpenAI(
                model=self.summary_model_name, api_key=openai_api_key()
            )
        return self._summary_model

    def _count(self, text: str) -> int:
        # tiktoken loads its encoding on first use (a download if not cached), so a memory
        # that never counts tokens never needs it.
        return token_counter(self.summary_model_name)(text)

    def _tokens(self, messages: Sequence[BaseMessage]) -> int:
        return sum(self._count(message.text) for message in messages)

    def _shorten(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """The turn with each tool result above `max_tool_result_tokens` replaced by its header."""
        headers = tool_result_headers(messages)
        return [
            message.model_copy(update={"content": headers[i]})
            if i in headers and self._count(message.text) > self.max_tool_result_tokens
            else message
            for i, message in enumerate(messages)
        ]

    def clusters(self) -> list[Cluster]:
        """Every cluster, oldest first."""
        stored = self.archive.clusters()
        centroids = self.centroids.get([cluster for cluster, _, _ in stored])
        summaries = self.archive.summaries()
        return [
            Cluster(
                c, turns, oldest_at, centroids.get(c) or self._centroid(turns), summaries.get(c)
            )
            for c, oldest_at, turns in stored
        ]

    def _centroid(self, turns: tuple[int, ...] | list[int]) -> list[float]:
        vectors = self.vectors.get(list(turns))
        return centroid([vectors[t] for t in turns])

    def _unclustered_count(self) -> int:
        return len(self.vectors.turns() - self.archive.clustered_turns())

    def _cluster_batch(self) -> list[int]:
        """Cluster the unclustered embedded turns once there are `min_batch_turns` of them.

        Only turns outside the recent window have a vector, so the batch lies outside it.
        """
        stored = {c: turns for c, _, turns in self.archive.clusters()}
        if missing := stored.keys() - self.centroids.turns():  # write failed after a commit
            self.centroids.add({c: self._centroid(stored[c]) for c in missing})
        batch = sorted(self.vectors.turns() - self.archive.clustered_turns())
        if len(batch) < self.min_batch_turns:
            return []
        vectors = self.vectors.get(batch)
        groups = self.clustering.cluster(
            batch, [vectors[t] for t in batch], self.similarity_threshold
        )
        existing = self.clusters()  # oldest first, so on a tie the oldest wins
        plan: list[tuple[int | None, list[int]]] = []
        for group in groups:
            batch_centroid = centroid([vectors[t] for t in group])
            best, best_similarity = None, self.similarity_threshold - EPSILON
            for cluster in existing:
                similarity = cosine(batch_centroid, cluster.centroid)
                if similarity >= best_similarity + (EPSILON if best is not None else 0):
                    best, best_similarity = cluster.id, similarity
            plan.append((best, group))
        ids = self.archive.assign(plan)
        # Written after the SQLite commit; a missing centroid is backfilled by the next check.
        members = {cluster: turns for cluster, _, turns in self.archive.clusters()}
        self.centroids.add({c: self._centroid(members[c]) for c in set(ids)})
        return batch

    def _missing_vectors(self) -> set[int]:
        """Turns outside the recent window without a vector."""
        last_outside = self.archive.last_turn() - self.keep_recent_turns
        return set(range(1, last_outside + 1)) - self.vectors.turns()

    def _embed_turns_outside_window(self) -> None:
        """Embed every turn outside the recent window that has no vector yet (incl. backfill)."""
        missing = self._missing_vectors()
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
        """What to send to the model, from the last committed state.

        System messages first, then one "Conversation memory" system message with the cluster
        summaries in time order, then the last `keep_recent_turns` turns verbatim. In every
        turn but the current one, tool results above `max_tool_result_tokens` are replaced by
        their header.
        """
        last = self.archive.last_turn()
        system = self.archive.turns(SYSTEM_TURN, SYSTEM_TURN).get(SYSTEM_TURN, [])
        # One query: a summary and the turns it covers are committed together.
        if summaries := list(self.archive.summaries().values()):
            system = [*system, SystemMessage(memory_text(summaries))]
        recent = self.archive.turns(max(1, last - self.keep_recent_turns + 1), last)
        return system + [
            message
            for turn in sorted(recent)
            for message in (recent[turn] if turn == last else self._shorten(recent[turn]))
        ]
