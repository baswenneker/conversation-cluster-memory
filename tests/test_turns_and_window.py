from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation_cluster_memory import ConversationClusterMemory


def add_turns(memory: ConversationClusterMemory, count: int, start: int = 1) -> None:
    for n in range(start, start + count):
        memory.add_message(HumanMessage(f"question {n}"))
        memory.add_message(AIMessage(f"answer {n}"))


def test_window_keeps_the_last_turns_verbatim(embeddings) -> None:
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=2)
    add_turns(memory, 3)

    assert [m.content for m in memory.window()] == [
        "question 2",
        "answer 2",
        "question 3",
        "answer 3",
    ]


def test_a_turn_runs_until_the_next_user_message(embeddings) -> None:
    memory = ConversationClusterMemory("c1", embeddings=embeddings, keep_recent_turns=1)
    memory.add_message(HumanMessage("old question"))
    memory.add_message(HumanMessage("new question"))
    memory.add_message(AIMessage("", tool_calls=[{"name": "lookup", "args": {}, "id": "t1"}]))
    memory.add_message(ToolMessage("result", tool_call_id="t1"))
    memory.add_message(AIMessage("final answer"))

    assert [m.content for m in memory.window()] == [
        "new question",
        "",
        "result",
        "final answer",
    ]


def test_keep_recent_turns_defaults_to_ten(embeddings) -> None:
    memory = ConversationClusterMemory("c1", embeddings=embeddings)
    add_turns(memory, 12)

    assert memory.window()[0].content == "question 3"
