from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from conversation_cluster_memory import ConversationClusterMemory
from conversation_cluster_memory.clustering import AgglomerativeClustering
from tests.conftest import FakeSummaryModel, TopicEmbeddings

TOPICS = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}


def question(messages: Sequence[BaseMessage]) -> str:
    return messages[0].text


def memory_for(
    topics: dict[str, list[float]] = TOPICS, keep_recent_turns: int = 2, **options: Any
) -> ConversationClusterMemory:
    return ConversationClusterMemory(
        "c1",
        embeddings=TopicEmbeddings(topics),
        embedding_text=question,
        keep_recent_turns=keep_recent_turns,
        summary_model=FakeSummaryModel(),
        auto_compact="inline",
        **options,
    )


def add(memory: ConversationClusterMemory, *topics: str) -> None:
    """One turn per topic; the question text starts with the topic."""
    for topic in topics:
        n = memory.archive.last_turn() + 1
        memory.add_message(HumanMessage(f"{topic}: question {n}"))
        memory.add_message(AIMessage(f"answer {n}"))


def turns_per_cluster(memory: ConversationClusterMemory) -> list[list[int]]:
    return [list(cluster.turns) for cluster in memory.clusters()]


def test_nothing_is_clustered_before_min_batch_turns_lie_outside_the_window() -> None:
    memory = memory_for(min_batch_turns=8, cluster_every_n_turns=5)
    add(memory, *"a" * 9)  # turn 10 opens next: turns 1-8 lie outside the window

    assert turns_per_cluster(memory) == []

    add(memory, "a")  # turn 10 opened: the check ran with 8 turns outside the window

    assert turns_per_cluster(memory) == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_the_check_runs_only_every_cluster_every_n_turns() -> None:
    memory = memory_for(min_batch_turns=3, cluster_every_n_turns=5)
    add(memory, *"a" * 9)  # the check at turn 5 found 3 turns outside the window

    assert turns_per_cluster(memory) == [[1, 2, 3]]

    add(memory, "a")

    assert turns_per_cluster(memory) == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_turns_on_one_topic_share_a_cluster_and_a_topic_change_opens_a_new_one() -> None:
    memory = memory_for(min_batch_turns=8, cluster_every_n_turns=5)
    add(memory, *"aaaabbbb", "c", "c")

    assert turns_per_cluster(memory) == [[1, 2, 3, 4], [5, 6, 7, 8]]


def _close_topics(similarity: float) -> dict[str, list[float]]:
    return {"a": [1.0, 0.0], "b": [similarity, (1 - similarity**2) ** 0.5], "c": [0.0, 1.0]}


def test_raising_the_threshold_splits_a_cluster_a_lower_one_keeps_together() -> None:
    clusters = {}
    for threshold in (0.7, 0.9):
        memory = memory_for(
            _close_topics(0.8),
            min_batch_turns=8,
            cluster_every_n_turns=5,
            similarity_threshold=threshold,
            clustering=AgglomerativeClustering(proximity_bonus=0),
        )
        add(memory, *"aaaabbbb", "c", "c")
        clusters[threshold] = turns_per_cluster(memory)

    assert clusters[0.7] == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert clusters[0.9] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_the_proximity_bonus_keeps_neighbours_together_but_not_distant_turns() -> None:
    clustering = AgglomerativeClustering(proximity_bonus=0.1)
    a, b = _close_topics(0.75)["a"], _close_topics(0.75)["b"]

    assert clustering.cluster([1, 2], [a, b], threshold=0.8) == [[1, 2]]
    assert clustering.cluster([1, 20], [a, b], threshold=0.8) == [[1], [20]]


def test_a_batch_cluster_similar_to_an_existing_cluster_joins_it() -> None:
    memory = memory_for(min_batch_turns=8, cluster_every_n_turns=5)
    add(memory, *"aaaabbbb", *"a" * 10, "c", "c")

    assert turns_per_cluster(memory) == [[1, 2, 3, 4, *range(9, 19)], [5, 6, 7, 8]]


def test_on_a_tie_the_oldest_cluster_wins() -> None:
    topics = {**TOPICS, "d": [1.0, 1.0, 0.0]}  # equally similar to a and b
    memory = memory_for(
        topics, min_batch_turns=8, cluster_every_n_turns=5, similarity_threshold=0.7
    )
    add(memory, *"bbbbaaaa", *"d" * 10, "c", "c")

    assert turns_per_cluster(memory) == [[1, 2, 3, 4, *range(9, 19)], [5, 6, 7, 8]]


class RecordingClustering(AgglomerativeClustering):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[int]] = []

    def cluster(self, turns, vectors, threshold):
        self.batches.append(list(turns))
        return super().cluster(turns, vectors, threshold)


def test_clustered_turns_are_never_clustered_again() -> None:
    clustering = RecordingClustering()
    memory = memory_for(min_batch_turns=3, cluster_every_n_turns=5, clustering=clustering)
    add(memory, *"abcabcabcabcabcabcab")

    assert clustering.batches == [
        [1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18],
    ]
    assert sorted(t for c in turns_per_cluster(memory) for t in c) == list(range(1, 19))


def test_the_same_input_always_yields_the_same_clusters() -> None:
    topics = {"a": [1.0, 0.1, 0.2], "b": [0.2, 1.0, 0.1], "c": [0.6, 0.6, 0.5]}
    results = []
    for _ in range(3):
        memory = memory_for(topics, min_batch_turns=8, cluster_every_n_turns=5)
        add(memory, *"abcacbbacabcabcabcab")
        results.append([(c.id, c.turns) for c in memory.clusters()])

    assert results[0] == results[1] == results[2]
    assert results[0]
