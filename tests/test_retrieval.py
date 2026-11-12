from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from conversation_cluster_memory import ConversationClusterMemory, EmbeddingModelMismatchError
from tests.conftest import FakeEmbeddings, FakeSummaryModel, TopicEmbeddings
from tests.test_clustering import question

# One-hot topics; a text without a known topic embeds as "other", far from every topic.
TOPICS = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0],
    "c": [0.0, 0.0, 1.0, 0.0],
}
OTHER = [0.0, 0.0, 0.0, 1.0]


def memory_for(*summaries: str, **options: Any) -> ConversationClusterMemory:
    """A memory whose first compaction summarizes its clusters as `summaries`, in order."""
    settings: dict[str, Any] = {
        "embeddings": TopicEmbeddings(TOPICS, default=OTHER),
        "embedding_text": question,
        "keep_recent_turns": 2,
        "min_batch_turns": 4,
        "cluster_every_n_turns": 5,
        "summary_model": FakeSummaryModel(texts=list(summaries)),
        "auto_compact": "inline",
    } | options
    return ConversationClusterMemory("c1", **settings)


def add(memory: ConversationClusterMemory, topics: str) -> None:
    for topic in topics:
        n = memory.archive.last_turn() + 1
        memory.add_message(HumanMessage(f"{topic}: question {n}"))
        memory.add_message(AIMessage(f"answer {n}"))


def cluster_ids(memory: ConversationClusterMemory) -> dict[str, int]:
    """Cluster id per topic: the cluster of turns 1-4 is "a", of turns 5-8 "b"."""
    return {"ab"[i]: cluster.id for i, cluster in enumerate(memory.clusters())}


def test_a_query_close_to_a_cluster_returns_it_and_a_distant_query_nothing() -> None:
    memory = memory_for("a: about the first topic", "b: about the second topic")
    add(memory, "aaaabbbbcc")  # turns 1-4 and 5-8 become two summarized clusters

    (hit,) = memory.retrieve("a: what did we say?")

    assert hit.cluster == cluster_ids(memory)["a"]
    assert abs(hit.similarity - 1.0) < 1e-6
    assert memory.retrieve("c: something new") == []


def test_an_exact_word_from_a_summary_or_anchor_finds_its_cluster_despite_low_similarity() -> None:
    memory = memory_for("a: the invoice ZX81 was approved", "b: about the second topic")
    add(memory, "aaaabbbbcc")

    (hit,) = memory.retrieve("c: what happened to ZX81?")
    assert hit.cluster == cluster_ids(memory)["a"]
    assert hit.similarity < 0.5

    # The fake summary model names each cluster "name <n>": an anchor, not the summary text.
    (anchor_hit,) = memory.retrieve("c: number 2")
    assert anchor_hit.cluster == cluster_ids(memory)["b"]


def test_do_you_remember_when_we_discussed_x_finds_the_cluster_about_x() -> None:
    memory = memory_for("a: planning the vegetable garden", "b: choosing running shoes")
    add(memory, "aaaabbbbcc")

    hits = memory.retrieve("Do you remember when we discussed the garden?")

    assert [hit.cluster for hit in hits] == [cluster_ids(memory)["a"]]


def test_a_hit_carries_the_summary_or_else_the_original_turns_in_time_order() -> None:
    memory = memory_for("a: about the first topic", "b: about the second topic")
    add(memory, "aaaabbbbcc")

    (summary_hit,) = memory.retrieve("a: what did we say?")
    assert summary_hit.is_summary
    assert summary_hit.content == "a: about the first topic"
    assert f"{summary_hit.start} to {summary_hit.end}" == memory.archive.period([1, 2, 3, 4])

    unsummarized = memory_for(summary_model=FakeSummaryModel(fail_times=100))
    add(unsummarized, "aaaabbbbcc")

    (turns_hit,) = unsummarized.retrieve("a: what did we say?")
    assert not turns_hit.is_summary
    assert turns_hit.content == (
        "User: a: question 1\nAssistant: answer 1\nUser: a: question 2\nAssistant: answer 2\n"
        "User: a: question 3\nAssistant: answer 3\nUser: a: question 4\nAssistant: answer 4"
    )
    assert abs(turns_hit.similarity - 1.0) < 1e-6
    assert f"{turns_hit.start} to {turns_hit.end}" == unsummarized.archive.period([1, 2, 3, 4])


def test_hits_stop_at_k_and_a_hit_that_would_exceed_the_token_budget_is_dropped() -> None:
    long_summary = "a: kiwi " + " ".join(f"detail{n}" for n in range(200))
    memory = memory_for(long_summary, "b: kiwi", retrieval_token_budget=100)
    add(memory, "aaaabbbbcc")

    assert [hit.cluster for hit in memory.retrieve("c: kiwi")] == [cluster_ids(memory)["b"]]
    assert len(memory.retrieve("c: kiwi", k=1)) == 1

    roomy = memory_for("a: kiwi", "b: kiwi")
    add(roomy, "aaaabbbbcc")
    assert len(roomy.retrieve("c: kiwi")) == 2
    assert len(roomy.retrieve("c: kiwi", k=1)) == 1


def retrieved(memory: ConversationClusterMemory) -> list[str]:
    return [m.text for m in memory.window() if m.text.startswith("Retrieved from memory")]


def test_window_adds_hits_between_the_memory_message_and_the_recent_turns() -> None:
    memory = memory_for(summary_model=FakeSummaryModel(fail_times=100))
    add(memory, "aaaabbbbca")  # the latest user message is about "a"

    window = [m.text for m in memory.window()]

    assert window[0].startswith("Retrieved from memory (original turns")
    assert "User: a: question 1" in window[0]
    assert window[1:] == ["c: question 9", "answer 9", "a: question 10", "answer 10"]


def test_clusters_already_in_the_window_are_not_added_again() -> None:
    memory = memory_for("a: about the first topic", "b: about the second topic")
    add(memory, "aaaabbbbca")

    assert "a: about the first topic" in memory.window()[0].text  # the memory message
    assert memory.retrieve("a: question 10") != []
    assert retrieved(memory) == []


def test_a_query_embedded_with_another_model_is_refused() -> None:
    memory = memory_for("a: about the first topic", "b: about the second topic")
    add(memory, "aaaabbbbcc")
    memory.embeddings = FakeEmbeddings("another-model", size=4)

    with pytest.raises(EmbeddingModelMismatchError):
        memory.retrieve("a: what did we say?")


def test_retrieval_never_returns_another_conversations_clusters(tmp_path: Path) -> None:
    other = memory_for("a: the invoice ZX81", "b: about b", path=tmp_path)
    add(other, "aaaabbbbcc")
    memory = ConversationClusterMemory(
        "c2",
        path=tmp_path,
        embeddings=TopicEmbeddings(TOPICS, default=OTHER),
        embedding_text=question,
        keep_recent_turns=2,
        min_batch_turns=4,
        summary_model=FakeSummaryModel(texts=["c: about c"]),
        auto_compact="inline",
    )
    add(memory, "ccccccbb")  # one cluster, about "c"

    assert other.retrieve("a: ZX81") != []
    assert memory.retrieve("a: ZX81") == []
    assert {hit.cluster for hit in memory.retrieve("c: about c")} <= {
        c.id for c in memory.clusters()
    }


def test_with_query_rewrite_the_chat_models_query_is_used() -> None:
    # This model cannot summarize, so the clusters stay unsummarized; it rewrites to "b".
    model = FakeListChatModel(responses=["b: the rewritten query"])
    memory = memory_for(summary_model=model, retrieval_rewrite_query=True)
    add(memory, "aaaabbbbcc")

    (hit,) = memory.retrieve("a: what did we say?")

    assert hit.cluster == cluster_ids(memory)["b"]


def test_the_tool_returns_the_same_hits_as_retrieve() -> None:
    memory = memory_for("a: about the first topic", "b: about the second topic")
    add(memory, "aaaabbbbcc")
    tool = memory.as_tool()

    call = {"type": "tool_call", "name": "recall_memory", "id": "1", "args": {"query": "a: hi"}}
    result = tool.invoke(call)

    assert tool.name == "recall_memory"
    assert result.artifact == memory.retrieve("a: hi") != []
    assert "about the first topic" in result.content


def test_with_auto_retrieval_off_window_adds_no_hits() -> None:
    memory = memory_for(summary_model=FakeSummaryModel(fail_times=100), auto_retrieval=False)
    add(memory, "aaaabbbbca")

    assert retrieved(memory) == []
    assert memory.retrieve("a: question 10") != []
