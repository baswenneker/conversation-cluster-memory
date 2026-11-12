from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conversation_cluster_memory import (
    ConversationClusterMemory,
    ConversationKey,
    EmbeddingModelMismatchError,
)
from tests.conftest import FakeEmbeddings
from tests.test_turns_and_window import add_turns


def test_a_reopened_memory_has_the_same_messages_and_vectors(tmp_path: Path) -> None:
    memory = ConversationClusterMemory(
        "c1", path=tmp_path, embeddings=FakeEmbeddings(), keep_recent_turns=2, auto_compact="inline"
    )
    add_turns(memory, 4)
    window, vectors = memory.window(), memory.vectors.turns()

    reopened = ConversationClusterMemory(
        "c1", path=tmp_path, embeddings=FakeEmbeddings(), keep_recent_turns=2, auto_compact="inline"
    )

    assert reopened.window() == window
    assert reopened.vectors.turns() == vectors == {1, 2}


def test_without_a_path_nothing_touches_disk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    memory = ConversationClusterMemory(
        "c1", embeddings=FakeEmbeddings(), keep_recent_turns=1, auto_compact="inline"
    )
    add_turns(memory, 3)

    assert list(tmp_path.iterdir()) == []


def test_two_in_memory_memories_for_one_key_are_separate() -> None:
    first = ConversationClusterMemory(
        "c1", embeddings=FakeEmbeddings(), keep_recent_turns=1, auto_compact="inline"
    )
    add_turns(first, 3)
    second = ConversationClusterMemory(
        "c1", embeddings=FakeEmbeddings(), keep_recent_turns=1, auto_compact="inline"
    )

    assert second.window() == []
    assert second.vectors.turns() == set()


def test_opening_with_another_embedding_model_is_refused(tmp_path: Path) -> None:
    memory = ConversationClusterMemory(
        "c1", path=tmp_path, embeddings=FakeEmbeddings("model-a"), keep_recent_turns=1
    )
    add_turns(memory, 2)

    with pytest.raises(EmbeddingModelMismatchError, match="model-a"):
        ConversationClusterMemory("c1", path=tmp_path, embeddings=FakeEmbeddings("model-b"))


def test_querying_with_another_embedding_model_is_refused(embeddings) -> None:
    memory = ConversationClusterMemory(
        "c1", embeddings=embeddings, keep_recent_turns=1, auto_compact="inline"
    )
    add_turns(memory, 2)

    assert memory.vectors.query(embeddings.embed_query("answer 1"), 1, "fake-model")[0][0] == 1
    with pytest.raises(EmbeddingModelMismatchError):
        memory.vectors.query(embeddings.embed_query("answer 1"), 1, "other-model")


def test_conversations_never_see_each_others_messages_or_vectors(tmp_path: Path) -> None:
    alice = ConversationClusterMemory(
        ConversationKey("c1", user="alice"),
        path=tmp_path,
        embeddings=FakeEmbeddings(),
        keep_recent_turns=1,
        auto_compact="inline",
    )
    bob = ConversationClusterMemory(
        ConversationKey("c1", user="bob"),
        path=tmp_path,
        embeddings=FakeEmbeddings(),
        keep_recent_turns=1,
        auto_compact="inline",
    )
    add_turns(alice, 3)
    bob.add_message(HumanMessage("hello"))

    assert [m.content for m in bob.window()] == ["hello"]
    assert bob.vectors.turns() == set()
    assert alice.vectors.turns() == {1, 2}


@pytest.mark.parametrize(
    ("one", "other"),
    [
        (ConversationKey("c1"), ConversationKey("c1", user="u1")),
        (ConversationKey("c1", user="u1"), ConversationKey("c1", tenant="u1")),
        (ConversationKey("c1", tenant="t1"), ConversationKey("c1", user="u1", tenant="t1")),
        (ConversationKey("c1"), ConversationKey("c1", user="")),
    ],
)
def test_keys_that_differ_in_an_optional_field_are_different_conversations(
    tmp_path: Path, one, other
) -> None:
    first = ConversationClusterMemory(one, path=tmp_path, embeddings=FakeEmbeddings())
    first.add_message(HumanMessage("only in the first"))
    second = ConversationClusterMemory(other, path=tmp_path, embeddings=FakeEmbeddings())

    assert second.window() == []


def test_a_plain_string_is_a_key_with_only_a_conversation(tmp_path: Path) -> None:
    by_string = ConversationClusterMemory("c1", path=tmp_path, embeddings=FakeEmbeddings())
    by_string.add_message(HumanMessage("hi"))
    by_key = ConversationClusterMemory(
        ConversationKey("c1"), path=tmp_path, embeddings=FakeEmbeddings()
    )

    assert [m.content for m in by_key.window()] == ["hi"]


class AppKey:
    def __init__(self, workspace: str, thread: str) -> None:
        self.workspace, self.thread = workspace, thread

    def key_parts(self) -> tuple[str, ...]:
        return (self.workspace, self.thread)


def test_an_application_key_type_works(tmp_path: Path) -> None:
    memory = ConversationClusterMemory(
        AppKey("w1", "t1"), path=tmp_path, embeddings=FakeEmbeddings()
    )
    memory.add_message(HumanMessage("hi"))
    memory.add_message(AIMessage("hello"))
    same = ConversationClusterMemory(AppKey("w1", "t1"), path=tmp_path, embeddings=FakeEmbeddings())
    other = ConversationClusterMemory(
        AppKey("w1", "t2"), path=tmp_path, embeddings=FakeEmbeddings()
    )

    assert [m.content for m in same.window()] == ["hi", "hello"]
    assert other.window() == []
