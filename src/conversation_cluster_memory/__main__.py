"""Demo: run a built-in conversation through an in-memory ConversationClusterMemory.

uv run python -m conversation_cluster_memory

Summaries need the summary model, so OPENAI_API_KEY (environment or .env); without it the
demo shows the clusters and says why no summary replaced them.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from conversation_cluster_memory import CompactionReport, ConversationClusterMemory, ConversationKey
from conversation_cluster_memory.clustering import cosine

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
    # A small window and batches so this short conversation is clustered twice, and the
    # second batch can join a cluster of the first. The thresholds are lowered: with this
    # model same-topic turns often score only 0.5-0.7, and a "do you remember" question
    # about a topic scores about 0.4 against it.
    memory = ConversationClusterMemory(
        ConversationKey("demo", user="reader"),
        keep_recent_turns=4,
        min_batch_turns=4,
        cluster_every_n_turns=4,
        similarity_threshold=0.5,
        retrieval_min_similarity=0.4,
        auto_compact="inline",
        on_compact=report,
    )
    print(f"Embedding model: {memory.vectors.model_id}")
    print(f"Recent window: {memory.keep_recent_turns} turns")
    print(f"Similarity threshold: {memory.similarity_threshold}")
    print(f"Summary model: {memory.summary_model_name}\n")
    for message in CONVERSATION:
        memory.add_message(message)

    with_vector = memory.vectors.turns()
    turns = memory.archive.turns(1, memory.archive.last_turn())
    for number, messages in turns.items():
        mark = "vector" if number in with_vector else "in window"
        print(f"turn {number:2}  [{mark:9}]  {messages[0].text}")
    window = memory.window()
    print(f"\n{len(with_vector)} turns have a vector; the window holds {len(window)} messages.")

    vectors = memory.vectors.get()
    for cluster in memory.clusters():
        print(f"\ncluster {cluster.id} (similarity of each turn to the cluster centroid)")
        for number in cluster.turns:
            similarity = cosine(vectors[number], cluster.centroid)
            print(f"  turn {number:2}  {similarity:.2f}  {turns[number][0].text}")
        if cluster.summary is not None:
            print(f"  summary [{cluster.summary.subject}]: {cluster.summary.text}")

    print("\nThe window: summaries replace their clusters' turns, the recent turns stay verbatim.")
    for message in window:
        print(f"  {message.type:6}  {message.text[:200]}")

    query = "Do you remember when we talked about running shoes?"
    print(f"\nRetrieval for: {query}")
    for hit in memory.retrieve(query):
        kind = "summary" if hit.is_summary else "original turns"
        print(f"  cluster {hit.cluster}  similarity {hit.similarity:.2f}  {kind}")
        print(f"    {hit.content[:200]}")


def report(round: CompactionReport) -> None:
    if round.summaries or round.errors:
        print(
            f"compaction round: clustered {round.clustered_turns}, "
            f"{len(round.summaries)} summaries, window {round.window_tokens_before} -> "
            f"{round.window_tokens_after} tokens"
        )
    for error in round.errors:
        print(f"  no summary: {error}")


if __name__ == "__main__":
    main()
