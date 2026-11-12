from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from conversation_cluster_memory import (
    ConversationClusterMemory,
    ConversationKey,
    EmbeddingModelMismatchError,
)
from conversation_cluster_memory.summaries import token_counter
from conversation_cluster_memory.vectors import chroma_client
from tests.conftest import FakeEmbeddings, FakeSummaryModel, TopicEmbeddings
from tests.test_clustering import question
from tests.test_retrieval import OTHER, TOPICS, TURNS, add

LONG = " and more about this topic" * 12
# Clusters a, b and c get a long summary each; the fourth call merges the oldest two.
SUMMARIES = (f"a: the first topic{LONG}", f"b: the second topic{LONG}", f"c: the third topic{LONG}")
PERIOD = "a: the first and second topic, briefly"
LIMIT = 150


def memory_for(path: Path | None = None, **options: Any) -> ConversationClusterMemory:
    settings: dict[str, Any] = {
        "path": path,
        "embeddings": TopicEmbeddings(TOPICS, default=OTHER),
        "embedding_text": question,
        "keep_recent_turns": 2,
        "min_batch_turns": 4,
        "cluster_every_n_turns": 5,
        "summary_model": FakeSummaryModel(texts=[*SUMMARIES, PERIOD]),
        "auto_compact": "inline",
        "max_summary_tokens": LIMIT,
    } | options
    return ConversationClusterMemory("c1", **settings)


def memory_message(memory: ConversationClusterMemory) -> str:
    return memory.window()[0].text


def test_above_the_limit_the_oldest_summaries_merge_into_a_level_2_summary() -> None:
    memory = memory_for()
    add(memory, TURNS)

    text = memory_message(memory)
    assert PERIOD in text
    assert SUMMARIES[0] not in text and SUMMARIES[1] not in text
    assert SUMMARIES[2] in text
    assert token_counter(memory.summary_model_name)(text) <= LIMIT


def test_under_the_limit_nothing_merges() -> None:
    memory = memory_for(max_summary_tokens=3000)
    add(memory, TURNS)

    text = memory_message(memory)
    assert all(summary in text for summary in SUMMARIES)
    assert PERIOD not in text


def test_merged_summaries_and_turns_stay_and_retrieval_finds_both_levels() -> None:
    memory = memory_for()
    add(memory, TURNS)

    first, second, third = memory.clusters()
    assert first.summary is not None and first.summary.text == SUMMARIES[0]
    assert memory.archive.turns(1, 1)[1][0].content == "a: question 1"
    found = {hit.cluster: hit.content for hit in memory.retrieve("a: what did we say?")}
    assert found.pop(first.id) == SUMMARIES[0]
    assert list(found.values()) == [PERIOD]  # the period, a cluster of its own


def test_a_reopened_memory_resumes_with_the_same_window(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    add(memory, TURNS)
    window = memory.window()

    reopened = memory_for(tmp_path)

    assert reopened.window() == window
    assert PERIOD in reopened.window()[0].text


class NamedTopicEmbeddings(TopicEmbeddings):
    def __init__(self, model: str, scale: float = 1.0) -> None:
        topics = {t: [scale * x for x in v] for t, v in TOPICS.items()}
        super().__init__(topics, default=[scale * x for x in OTHER])
        self.model = model


def test_another_embedding_model_needs_reindex_after_which_retrieval_works(
    tmp_path: Path,
) -> None:
    memory = memory_for(tmp_path, embeddings=NamedTopicEmbeddings("model-a"))
    add(memory, TURNS)
    with pytest.raises(EmbeddingModelMismatchError, match=r"reindex\(new_embeddings\)"):
        memory_for(tmp_path, embeddings=NamedTopicEmbeddings("model-b"))

    memory.reindex(NamedTopicEmbeddings("model-b", scale=2.0))

    reopened = memory_for(tmp_path, embeddings=NamedTopicEmbeddings("model-b", scale=2.0))
    for store in (reopened.vectors, reopened.centroids, reopened.summary_vectors):
        assert store.model_ids() == {"model-b"}
    assert reopened.vectors.turns() == set(range(1, 9))
    assert reopened.vectors.get([1])[1] == [2.0, 0.0, 0.0, 0.0]
    first = reopened.clusters()[0]
    assert {hit.cluster for hit in reopened.retrieve("a: what did we say?")} >= {first.id}


KEYS = {
    "alice-1": ConversationKey("c1", user="alice", tenant="t1"),
    "alice-2": ConversationKey("c2", user="alice", tenant="t1"),
    "bob": ConversationKey("c3", user="bob", tenant="t1"),
    "alice-t2": ConversationKey("c4", user="alice", tenant="t2"),
    "named-alice": ConversationKey("alice"),  # sets no user: never matched as user alice
    "plain": "t1",  # a plain string sets no tenant
}


def stored(tmp_path: Path) -> dict[str, ConversationClusterMemory]:
    memories = {}
    for name, key in KEYS.items():
        memory = ConversationClusterMemory(
            key, path=tmp_path, embeddings=FakeEmbeddings(), keep_recent_turns=1
        )
        memory.add_message(HumanMessage(f"{name}: question"))
        memory.add_message(HumanMessage(f"{name}: next question"))
        memory.wait_idle()
        memories[name] = memory
    return memories


def left(tmp_path: Path) -> set[str]:
    """The names of KEYS that still have messages or vectors in the archive."""
    names = set()
    for name, key in KEYS.items():
        memory = ConversationClusterMemory(key, path=tmp_path, embeddings=FakeEmbeddings())
        if memory.archive.last_turn() or memory.vectors.turns():
            names.add(name)
    return names


def collections(tmp_path: Path) -> set[str]:
    return {c.name for c in chroma_client(str(tmp_path / "chroma")).list_collections()}


@pytest.mark.parametrize(
    ("delete", "deleted"),
    [
        (lambda p: ConversationClusterMemory.delete_conversation(p, KEYS["bob"]), {"bob"}),
        (lambda p: ConversationClusterMemory.delete_conversation(p, "t1"), {"plain"}),
        (
            lambda p: ConversationClusterMemory.delete_user(p, "alice"),
            {"alice-1", "alice-2", "alice-t2"},
        ),
        (lambda p: ConversationClusterMemory.delete_user(p, "alice", tenant="t2"), {"alice-t2"}),
        (lambda p: ConversationClusterMemory.delete_tenant(p, "t1"), {"alice-1", "alice-2", "bob"}),
    ],
)
def test_a_delete_removes_exactly_its_scope(tmp_path: Path, delete, deleted: set[str]) -> None:
    memories = stored(tmp_path)
    before = collections(tmp_path)

    assert delete(tmp_path) == len(deleted)

    names = {m.vectors.name for n, m in memories.items() if n in deleted}
    assert names and not names & collections(tmp_path)
    assert before - collections(tmp_path) == {
        f"{n}{suffix}" for n in names for suffix in ("", "-centroids", "-summaries")
    }
    assert left(tmp_path) == set(KEYS) - deleted
