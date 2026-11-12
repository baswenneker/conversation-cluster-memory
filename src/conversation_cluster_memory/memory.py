"""The public entry point: ConversationClusterMemory."""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.tools import BaseTool, StructuredTool

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
from conversation_cluster_memory.retrieval import REWRITE_PROMPT, RetrievalHit, fuse, hit_text
from conversation_cluster_memory.summaries import (
    DEFAULT_SUMMARY_MODEL,
    POOR_COMPRESSION_RATIO,
    CompactionReport,
    Summary,
    SummaryReport,
    first_tokens,
    memory_text,
    previous_text,
    summarize,
    token_counter,
    transcript,
)
from conversation_cluster_memory.vectors import VectorStore, chroma_client

logger = logging.getLogger(__name__)

type AutoCompact = Literal["background", "inline", "off"]


class _Pending(NamedTuple):
    """A cluster to summarize: committed (`cluster`) or planned this round (`plan_index`)."""

    cluster: int | None
    plan_index: int | None
    new: list[int]  # turns no summary covers yet
    turns: tuple[int, ...]  # all its turns


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
        auto_retrieval: bool = True,
        retrieval_k: int = 3,
        retrieval_token_budget: int = 1500,
        retrieval_min_similarity: float = 0.5,
        retrieval_min_keyword_score: float = 0.0,
        retrieval_rewrite_query: bool = False,
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
        if not -1 <= retrieval_min_similarity <= 1:
            raise ValueError(
                f"retrieval_min_similarity is a cosine similarity; got {retrieval_min_similarity}"
            )
        self.auto_retrieval = auto_retrieval
        self.retrieval_k = retrieval_k
        self.retrieval_token_budget = retrieval_token_budget
        self.retrieval_min_similarity = retrieval_min_similarity
        # Any matching word scores above 0. BM25 scores depend on how many clusters a
        # conversation has; raise this when common words add noise.
        self.retrieval_min_keyword_score = retrieval_min_keyword_score
        self.retrieval_rewrite_query = retrieval_rewrite_query
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
            weakref.finalize(self, client.delete_collection, f"{name}-summaries")
        else:
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            self.archive = MessageStore(directory / "archive.sqlite3", storage_id(key))
            client = chroma_client(str(directory / "chroma"))
        model_id = embedding_model_id(self.embeddings)
        self.vectors = VectorStore(client, name, model_id)
        self.centroids = VectorStore(client, f"{name}-centroids", model_id)
        # Keyed by summary id, so a cluster's newer summary never reuses an older one's vector.
        self.summary_vectors = VectorStore(client, f"{name}-summaries", model_id)

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
        return bool(self._pending([]))

    def _compact(self, check: bool | None) -> CompactionReport:
        """One round: embed, cluster when a check is due (or `check`), summarize, report."""
        report = CompactionReport()
        with self._round_lock:
            if check is None:
                with self._state:
                    check, self._check_due = self._check_due, False
            # The recent turns counted here are not added again when this round summarizes them.
            first_recent = self.archive.last_turn() - self.keep_recent_turns + 1
            # Retrieval hits are left out: they depend on the question, not on the round.
            report.window_tokens_before = self._tokens(self._window(retrieval=False))
            try:
                self._embed_turns_outside_window()
                plan = self._cluster_batch() if check else []
                done = self._summarize(self._pending(plan), report)
                if plan or done:
                    self._commit(plan, done, report, first_recent)
            except Exception as error:
                # Nothing of this round is committed; the next round does it again.
                logger.exception("compaction round failed; retrying in the next round")
                report.errors.append(repr(error))
            self._embed_summaries()
            report.window_tokens_after = self._tokens(self._window(retrieval=False))
        logger.info("compaction round: %s", report)
        if self.on_compact is not None:
            try:
                self.on_compact(report)
            except Exception:
                logger.exception("on_compact callback failed")
        return report

    def _pending(self, plan: list[tuple[int | None, list[int]]]) -> list[_Pending]:
        """Every cluster this round must summarize: committed clusters with turns no summary
        covers yet, and the clusters `plan` grows or creates, oldest first."""
        summarized = self.archive.summarized_turns()
        joining: dict[int, list[int]] = {}
        for cluster, group in plan:
            if cluster is not None:
                joining.setdefault(cluster, []).extend(group)
        pending = []
        for cluster, _, members in self.archive.clusters():
            turns = tuple(sorted([*members, *joining.get(cluster, [])]))
            if new := [t for t in turns if t not in summarized]:
                pending.append(_Pending(cluster, None, new, turns))
        pending += [
            _Pending(None, i, group, tuple(group))
            for i, (cluster, group) in enumerate(plan)
            if cluster is None
        ]
        return pending

    def _summarize(
        self, pending: list[_Pending], report: CompactionReport
    ) -> list[tuple[_Pending, Summary]]:
        """Summarize each pending cluster; a failed one stays unsummarized for a retry."""
        current = self.archive.summaries()
        done = []
        for item in pending:
            previous = current.get(item.cluster) if item.cluster is not None else None
            if self.resummarize_strategy == "full":
                previous = None
            inputs = item.new if previous is not None else list(item.turns)
            stored = self.archive.turns(min(item.turns), max(item.turns))
            try:
                summary, usage = summarize(
                    self._model(),
                    transcript([self._clip(stored[t]) for t in inputs]),
                    previous_text(previous) if previous is not None else None,
                    self.archive.period(list(item.turns)),
                )
            except Exception as error:
                name = item.cluster if item.cluster is not None else f"new {item.turns}"
                logger.exception("summarizing cluster %s failed; retrying next round", name)
                report.errors.append(f"cluster {name}: {error!r}")
                continue
            report.input_tokens += usage.get("input_tokens", 0)
            report.output_tokens += usage.get("output_tokens", 0)
            done.append((item, summary))
        return done

    def _commit(
        self,
        plan: list[tuple[int | None, list[int]]],
        done: list[tuple[_Pending, Summary]],
        report: CompactionReport,
        first_recent: int,
    ) -> None:
        """Commit the round's clusters and summaries in one transaction, then report them."""
        ids = self.archive.commit_round(
            plan,
            [(item.cluster, item.plan_index, summary, item.turns) for item, summary in done],
        )
        report.clustered_turns = sorted(t for _, group in plan for t in group)
        if plan:
            # Written after the SQLite commit; a missing centroid is backfilled by the next check.
            members = {cluster: turns for cluster, _, turns in self.archive.clusters()}
            self.centroids.add({c: self._centroid(members[c]) for c in set(ids)})
        for item, summary in done:
            stored = self.archive.turns(min(item.turns), max(item.turns))
            shortened = {t: self._shorten(stored[t]) for t in item.turns}
            # The token counts measure the turns as the window would hold them.
            original = self._tokens([m for t in item.turns for m in shortened[t]])
            report.window_tokens_before += self._tokens(
                [m for t in item.new if t < first_recent for m in shortened[t]]
            )
            summary_tokens = self._count(summary.text)
            cluster = item.cluster
            if cluster is None:
                assert item.plan_index is not None
                cluster = ids[item.plan_index]
            report.summaries.append(
                SummaryReport(
                    cluster,
                    item.turns,
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
        """The text of the messages plus the names and arguments of their tool calls."""
        return sum(
            self._count(message.text)
            + sum(
                self._count(call["name"] + json.dumps(call["args"]))
                for call in getattr(message, "tool_calls", [])
            )
            for message in messages
        )

    def _clip(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """The turn as the summary model reads it: a tool result above `max_tool_result_tokens`
        becomes its header plus its first `max_tool_result_tokens` tokens.

        So the facts at the start of a large result reach the summary, and one result never
        outgrows the model's context. Facts further down are lost; summarizing a large result
        in chunks is the upgrade if the evaluation shows that matters.
        """
        headers = tool_result_headers(messages)
        limit = self.max_tool_result_tokens
        return [
            message.model_copy(
                update={
                    "content": f"{headers[i]}, first {limit} tokens:\n"
                    + first_tokens(self.summary_model_name, message.text, limit)
                }
            )
            if i in headers and self._count(message.text) > limit
            else message
            for i, message in enumerate(messages)
        ]

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

    def _cluster_batch(self) -> list[tuple[int | None, list[int]]]:
        """Plan the clustering of the unclustered embedded turns once there are
        `min_batch_turns` of them: per batch cluster, the cluster it joins (None: a new one).

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
        return plan

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

    def _embed_summaries(self) -> None:
        """Embed every current summary without a vector (incl. backfill after a failure)."""
        missing = {s.id: s.text for s in self.archive.summaries().values()}
        for done in self.summary_vectors.turns() & missing.keys():
            del missing[done]
        if not missing:
            return
        try:
            vectors = self.embeddings.embed_documents(list(missing.values()))
            self.summary_vectors.add(dict(zip(missing, vectors, strict=True)))
        except Exception:
            # Until then retrieval falls back to the cluster's centroid.
            logger.exception("embedding summaries %s failed; retrying next round", list(missing))

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """The clusters of this conversation that match the query, best first: at most `k`
        (default `retrieval_k`) and at most `retrieval_token_budget` tokens together."""
        return self._search(query, self.retrieval_k if k is None else k, exclude=set())

    def as_tool(self) -> BaseTool:
        """A LangChain tool `recall_memory(query)` that returns the hits of `retrieve()`."""

        def recall_memory(query: str) -> tuple[str, list[RetrievalHit]]:
            """Search earlier topics of this conversation that are no longer in view.

            Use it when the user refers to something discussed before, such as a name, an
            identifier, a number or a decision.
            """
            hits = self.retrieve(query)
            return "\n\n".join(hit_text(hit) for hit in hits) or "Nothing found.", hits

        return StructuredTool.from_function(recall_memory, response_format="content_and_artifact")

    def _search(self, query: str, k: int, exclude: set[int]) -> list[RetrievalHit]:
        """Hybrid search over the clusters not in `exclude`: vector and keyword hits, fused
        by rank, then cut at `k` and the token budget."""
        self.vectors.check(embedding_model_id(self.embeddings))
        # Every cluster of this conversation, compared one by one; a nearest-neighbour query
        # in Chroma is the upgrade when conversations reach thousands of clusters.
        candidates = {c.id: c for c in self.clusters() if c.id not in exclude}
        if not candidates or k < 1:
            return []
        if self.retrieval_rewrite_query:
            query = self._rewrite(query)
        vector = self.embeddings.embed_query(query)
        summary_vectors = self.summary_vectors.get(
            [c.summary.id for c in candidates.values() if c.summary is not None]
        )
        similarity = {
            c.id: cosine(
                vector,
                summary_vectors.get(c.summary.id, c.centroid) if c.summary else c.centroid,
            )
            for c in candidates.values()
        }
        by_vector = sorted(
            (c for c, s in similarity.items() if s >= self.retrieval_min_similarity - EPSILON),
            key=lambda c: (-similarity[c], c),
        )
        by_keyword = [
            c
            for c, _ in self.archive.keyword_search(query, self.retrieval_min_keyword_score)
            if c in candidates
        ]
        hits: list[RetrievalHit] = []
        budget = self.retrieval_token_budget
        for cluster in fuse(by_vector, by_keyword):
            if len(hits) == k:
                break
            hit = self._hit(candidates[cluster], similarity[cluster])
            tokens = self._count(hit_text(hit))
            if tokens > budget:
                continue  # dropped; a smaller hit further down may still fit
            budget -= tokens
            hits.append(hit)
        return hits

    def _hit(self, cluster: Cluster, similarity: float) -> RetrievalHit:
        start, end = self.archive.time_range(list(cluster.turns))
        if cluster.summary is not None:
            return RetrievalHit(cluster.id, similarity, start, end, True, cluster.summary.text)
        stored = self.archive.turns(min(cluster.turns), max(cluster.turns))
        content = transcript([self._shorten(stored[t]) for t in cluster.turns])
        return RetrievalHit(cluster.id, similarity, start, end, False, content)

    def _rewrite(self, query: str) -> str:
        """The chat model's search query for the message; the message itself if that fails."""
        try:
            rewritten = self._model().invoke([SystemMessage(REWRITE_PROMPT), HumanMessage(query)])
        except Exception:
            logger.exception("rewriting the query failed; searching with the message itself")
            return query
        return rewritten.text.strip() or query

    def window(self) -> list[BaseMessage]:
        """What to send to the model, from the last committed state.

        System messages first, then one "Conversation memory" system message with the cluster
        summaries in time order, then (with `auto_retrieval`) one system message per retrieval
        hit for the latest user message, then the last `keep_recent_turns` turns verbatim. In
        every turn but the current one, tool results above `max_tool_result_tokens` are
        replaced by their header.
        """
        return self._window(self.auto_retrieval)

    def _window(self, retrieval: bool) -> list[BaseMessage]:
        last = self.archive.last_turn()
        system = self.archive.turns(SYSTEM_TURN, SYSTEM_TURN).get(SYSTEM_TURN, [])
        # One query, and a round commits its clusters and summaries in one transaction.
        summaries = self.archive.summaries()
        if summaries:
            system = [*system, SystemMessage(memory_text(list(summaries.values())))]
        recent = self.archive.turns(max(1, last - self.keep_recent_turns + 1), last)
        question = next((m.text for m in recent.get(last, []) if isinstance(m, HumanMessage)), "")
        if retrieval and question:
            # Clusters whose summary is in the memory message are in the window already.
            hits = self._search(question, self.retrieval_k, exclude=set(summaries))
            system += [SystemMessage(hit_text(hit)) for hit in hits]
        return system + [
            message
            for turn in sorted(recent)
            for message in (recent[turn] if turn == last else self._shorten(recent[turn]))
        ]
