"""Demo: run a built-in conversation through an in-memory ConversationClusterMemory.

uv run python -m conversation_cluster_memory
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from conversation_cluster_memory import ConversationClusterMemory, ConversationKey

CONVERSATION: list[BaseMessage] = [
    HumanMessage("I want to start running. Where do I begin?"),
    AIMessage("Start with three short sessions a week, alternating running and walking."),
    HumanMessage("How long should each session be?"),
    AIMessage("About twenty minutes: one minute running, two minutes walking, repeated."),
    HumanMessage("Which shoes should I buy?"),
    AIMessage("Neutral running shoes that fit well; have your gait checked in a running shop."),
    HumanMessage("Let's switch topics. Can you plan a vegetable garden for me?"),
    AIMessage("Sure. How much space do you have, and how many hours of sun?"),
    HumanMessage("About six square metres, sunny for most of the day."),
    AIMessage("", tool_calls=[{"name": "planting_calendar", "args": {"m2": 6}, "id": "call-1"}]),
    ToolMessage(
        "\n".join(f"week {w}: sow or plant crop {w % 7}" for w in range(1, 53)),
        tool_call_id="call-1",
    ),
    AIMessage("Here is a planting calendar for the year: tomatoes, beans, lettuce and herbs."),
    HumanMessage("When do I plant the tomatoes?"),
    AIMessage("Outside after mid-May, once nights stay above ten degrees."),
    HumanMessage("And the beans?"),
    AIMessage("Sow beans directly from early May, five centimetres deep."),
    HumanMessage("Back to running: how do I avoid shin pain?"),
    AIMessage("Increase your weekly distance slowly, at most ten percent a week."),
    HumanMessage("Should I stretch before running?"),
    AIMessage("Warm up with five minutes of brisk walking; stretch after the run."),
    HumanMessage("What do I eat before a morning run?"),
    AIMessage("Something small and easy to digest, such as a banana, an hour before."),
    HumanMessage("Can I water the garden in the evening?"),
    AIMessage("Morning is better: leaves dry during the day, which prevents mildew."),
    HumanMessage("How often should I water?"),
    AIMessage("Deeply twice a week rather than a little every day."),
    HumanMessage("Do I need fertiliser?"),
    AIMessage("Work compost into the soil in spring; that is usually enough."),
    HumanMessage("Last question: how do I track my running progress?"),
    AIMessage("Log date, distance and how it felt; compare week by week."),
]


def main() -> None:
    memory = ConversationClusterMemory(ConversationKey("demo", user="reader"))
    print(f"Embedding model: {memory.vectors.model_id}")
    print(f"Recent window: {memory.keep_recent_turns} turns\n")
    for message in CONVERSATION:
        memory.add_message(message)

    with_vector = memory.vectors.turns()
    turns = memory.archive.turns(1, memory.archive.last_turn())
    for number, messages in turns.items():
        mark = "vector" if number in with_vector else "in window"
        print(f"turn {number:2}  [{mark:9}]  {messages[0].text}")
    window = memory.window()
    print(f"\n{len(with_vector)} turns have a vector; the window holds {len(window)} messages.")


if __name__ == "__main__":
    main()
