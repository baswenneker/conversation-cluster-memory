from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation_cluster_memory import ConversationClusterMemory, default_embedding_text
from tests.conftest import FakeEmbeddings
from tests.test_turns_and_window import add_turns


def test_a_conversation_shorter_than_the_window_has_no_vectors(embeddings) -> None:
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=3)
    add_turns(memory, 3)

    assert memory.vectors.turns() == set()
    assert embeddings.texts == []


def test_a_turn_gets_one_vector_when_it_leaves_the_window(embeddings) -> None:
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=3)
    add_turns(memory, 3)
    memory.add_message(HumanMessage("question 4"))

    assert memory.vectors.turns() == {1}

    memory.add_message(AIMessage("answer 4"))
    add_turns(memory, 2, start=5)

    assert memory.vectors.turns() == {1, 2, 3}
    assert len(embeddings.texts) == 3


def test_turns_missing_a_vector_are_backfilled_on_the_next_message() -> None:
    embeddings = FakeEmbeddings(fail_times=1)
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=1)
    add_turns(memory, 1)
    memory.add_message(HumanMessage("question 2"))  # turn 1 leaves; embedding fails

    assert memory.vectors.turns() == set()

    memory.add_message(AIMessage("answer 2"))

    assert memory.vectors.turns() == {1}


def test_default_embedding_text_has_user_answer_and_only_a_tool_header() -> None:
    big_result = "row\n" * 5000
    turn = [
        HumanMessage("How many rows?"),
        AIMessage("", tool_calls=[{"name": "count_rows", "args": {}, "id": "t1"}]),
        ToolMessage(big_result, tool_call_id="t1"),
        AIMessage("There are 5000 rows."),
    ]

    text = default_embedding_text(turn)

    assert "How many rows?" in text
    assert "There are 5000 rows." in text
    assert "count_rows" in text
    assert "20000" in text
    assert "row\nrow" not in text


def test_a_custom_embedding_text_replaces_the_default(embeddings) -> None:
    memory = ConversationClusterMemory(
        "c1",
        embeddings=embeddings,
        keep_recent_turns=1,
        embedding_text=lambda messages: f"{len(messages)} messages",
    )
    add_turns(memory, 2)

    assert embeddings.texts == ["2 messages"]


def test_tool_results_are_stored_in_full(embeddings) -> None:
    big_result = "x" * 1_000_000
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=1)
    memory.add_message(HumanMessage("fetch"))
    memory.add_message(AIMessage("", tool_calls=[{"name": "fetch", "args": {}, "id": "t1"}]))
    memory.add_message(ToolMessage(big_result, tool_call_id="t1"))
    add_turns(memory, 1, start=2)

    assert memory.archive.turns(1, 1)[1][2].content == big_result
