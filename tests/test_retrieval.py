from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from conversation_cluster_memory import ConversationClusterMemory, EmbeddingModelMismatchError
from tests.conftest import FakeEmbeddings, FakeSummaryModel, TopicEmbeddings
from tests.test_clustering import question

# One-hot topics; a text without a known topic (such as "x") embeds as OTHER, far from all.
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


# Turns 1-3, 4-6 and 7-8 become three clusters (a, b, c); turns 9-10 stay in the window.
TURNS = "aaabbbccxx"
SUMMARIES = ("a: about the first topic", "b: about the second topic", "c: about the third topic")


def cluster_ids(memory: ConversationClusterMemory) -> dict[str, int]:
    """Cluster id per topic, in the order of TURNS."""
    return {"abc"[i]: cluster.id for i, cluster in enumerate(memory.clusters())}


def test_a_query_close_to_a_cluster_returns_it_and_a_distant_query_nothing() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, TURNS)

    (hit,) = memory.retrieve("a: what did we say?")

    assert hit.cluster == cluster_ids(memory)["a"]
    assert abs(hit.similarity - 1.0) < 1e-6
    assert memory.retrieve("x: something new") == []


def test_a_word_shared_with_every_summary_does_not_make_a_hit() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, TURNS)

    assert memory.retrieve("x: what about the weather?") == []


def test_an_exact_word_from_a_summary_or_anchor_finds_its_cluster_despite_low_similarity() -> None:
    memory = memory_for("a: the invoice ZX81 was approved", *SUMMARIES[1:])
    add(memory, TURNS)

    (hit,) = memory.retrieve("x: what happened to ZX81?")
    assert hit.cluster == cluster_ids(memory)["a"]
    assert hit.similarity < 0.5

    # The fake summary model names each cluster "name <n>": an anchor, not the summary text.
    (anchor_hit,) = memory.retrieve("x: number 2")
    assert anchor_hit.cluster == cluster_ids(memory)["b"]


def test_do_you_remember_when_we_discussed_x_finds_the_cluster_about_x() -> None:
    memory = memory_for(
        "a: planning the vegetable garden", "b: choosing the running shoes", SUMMARIES[2]
    )
    add(memory, TURNS)

    hits = memory.retrieve("Do you remember when we discussed the garden?")

    assert [hit.cluster for hit in hits] == [cluster_ids(memory)["a"]]


def test_a_hit_carries_the_summary_or_else_the_original_turns_in_time_order() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, TURNS)

    (summary_hit,) = memory.retrieve("a: what did we say?")
    assert summary_hit.is_summary
    assert summary_hit.content == "a: about the first topic"
    assert f"{summary_hit.start} to {summary_hit.end}" == memory.archive.period([1, 2, 3])

    unsummarized = memory_for(summary_model=FakeSummaryModel(fail_times=100))
    add(unsummarized, TURNS)

    (turns_hit,) = unsummarized.retrieve("a: what did we say?")
    assert not turns_hit.is_summary
    assert turns_hit.content == (
        "User: a: question 1\nAssistant: answer 1\nUser: a: question 2\nAssistant: answer 2\n"
        "User: a: question 3\nAssistant: answer 3"
    )
    assert abs(turns_hit.similarity - 1.0) < 1e-6
    assert f"{turns_hit.start} to {turns_hit.end}" == unsummarized.archive.period([1, 2, 3])


def test_hits_stop_at_k_and_a_hit_that_would_exceed_the_token_budget_is_dropped() -> None:
    long_summary = "a: kiwi " + " ".join(f"detail{n}" for n in range(200))
    memory = memory_for(long_summary, "b: mango", SUMMARIES[2], retrieval_token_budget=100)
    add(memory, TURNS)

    assert [hit.cluster for hit in memory.retrieve("x: kiwi mango")] == [cluster_ids(memory)["b"]]

    roomy = memory_for("a: kiwi", "b: mango", SUMMARIES[2])
    add(roomy, TURNS)
    assert len(roomy.retrieve("x: kiwi mango")) == 2
    assert len(roomy.retrieve("x: kiwi mango", k=1)) == 1


def retrieved(memory: ConversationClusterMemory) -> list[str]:
    return [m.text for m in memory.window() if m.text.startswith("Retrieved from memory")]


def test_window_adds_hits_between_the_memory_message_and_the_recent_turns() -> None:
    memory = memory_for(summary_model=FakeSummaryModel(fail_times=100))
    add(memory, "aaabbbccxa")  # the latest user message is about "a"

    window = [m.text for m in memory.window()]

    assert window[0].startswith("Retrieved from memory (original turns")
    assert "User: a: question 1" in window[0]
    assert window[1:] == ["x: question 9", "answer 9", "a: question 10", "answer 10"]


def test_window_without_hits_when_the_query_cannot_be_embedded(caplog) -> None:
    memory = memory_for(summary_model=FakeSummaryModel(fail_times=100))
    add(memory, "aaabbbccxa")
    memory.embeddings.embed_query = Mock(side_effect=ConnectionError("embeddings down"))

    assert retrieved(memory) == []
    assert [m.text for m in memory.window()][-1] == "answer 10"
    assert "embeddings down" in caplog.text


def test_clusters_already_in_the_window_are_not_added_again() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, "aaabbbccxa")

    assert "a: about the first topic" in memory.window()[0].text  # the memory message
    assert memory.retrieve("a: question 10") != []
    assert retrieved(memory) == []


def test_a_query_embedded_with_another_model_is_refused() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, TURNS)
    memory.embeddings = FakeEmbeddings("another-model", size=4)

    with pytest.raises(EmbeddingModelMismatchError):
        memory.retrieve("a: what did we say?")
    with pytest.raises(EmbeddingModelMismatchError):
        memory.window()


def test_retrieval_never_returns_another_conversations_clusters(tmp_path: Path) -> None:
    other = memory_for("a: the invoice ZX81", *SUMMARIES[1:], path=tmp_path)
    add(other, TURNS)
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
    add(memory, "ccccccxx")  # one cluster, about "c"

    assert other.retrieve("a: ZX81") != []
    assert memory.retrieve("a: ZX81") == []
    assert {hit.cluster for hit in memory.retrieve("c: about c")} == {
        c.id for c in memory.clusters()
    }


def test_with_query_rewrite_the_chat_models_query_is_used() -> None:
    # This model cannot summarize, so the clusters stay unsummarized; it rewrites to "b".
    model = FakeListChatModel(responses=["b: the rewritten query"])
    memory = memory_for(summary_model=model, retrieval_rewrite_query=True)
    add(memory, TURNS)

    (hit,) = memory.retrieve("a: what did we say?")

    assert hit.cluster == cluster_ids(memory)["b"]


def test_the_tool_returns_the_same_hits_as_retrieve() -> None:
    memory = memory_for(*SUMMARIES)
    add(memory, TURNS)
    tool = memory.as_tool()

    call = {"type": "tool_call", "name": "recall_memory", "id": "1", "args": {"query": "a: hi"}}
    result = tool.invoke(call)

    assert tool.name == "recall_memory"
    assert result.artifact == memory.retrieve("a: hi") != []
    assert "about the first topic" in result.content


def test_with_auto_retrieval_off_window_adds_no_hits() -> None:
    memory = memory_for(summary_model=FakeSummaryModel(fail_times=100), auto_retrieval=False)
    add(memory, "aaabbbccxa")

    assert retrieved(memory) == []
    assert memory.retrieve("a: question 10") != []
