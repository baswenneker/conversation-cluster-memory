import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conversation_cluster_memory import ConversationClusterMemory
from tests.conftest import FakeSummaryModel, TopicEmbeddings
from tests.test_clustering import TOPICS, question


def memory_for(model: FakeSummaryModel, **options: Any) -> ConversationClusterMemory:
    settings: dict[str, Any] = {
        "embeddings": TopicEmbeddings(TOPICS),
        "embedding_text": question,
        "keep_recent_turns": 2,
        "min_batch_turns": 4,
        "cluster_every_n_turns": 5,
        "summary_model": model,
        "auto_compact": "inline",
    } | options
    return ConversationClusterMemory("c1", **settings)


def add(memory: ConversationClusterMemory, *topics: str) -> None:
    for topic in topics:
        n = memory.archive.last_turn() + 1
        memory.add_message(HumanMessage(f"{topic}: question {n}"))
        memory.add_message(AIMessage(f"answer {n}"))


def contents(memory: ConversationClusterMemory) -> list[str]:
    return [str(m.content) for m in memory.window()]


def test_a_summary_replaces_its_clusters_turns_in_the_window(summary_model) -> None:
    memory = memory_for(summary_model)
    add(memory, *"aaaaa")  # turn 5 opens: turns 1-3 outside the window, too few

    assert memory.clusters() == []

    add(memory, *"aaaaa")  # turn 10 opens: turns 1-8 clustered and summarized

    window = memory.window()
    assert isinstance(window[0], SystemMessage)
    assert window[0].text.startswith("Conversation memory")
    assert "summary 1" in window[0].text
    assert contents(memory)[1:] == ["a: question 9", "answer 9", "a: question 10", "answer 10"]


def test_the_archive_keeps_the_originals_marked_and_linked_to_the_summary(summary_model) -> None:
    memory = memory_for(summary_model)
    add(memory, *"a" * 10)

    (cluster,) = memory.clusters()
    assert cluster.summary is not None
    assert cluster.summary.text == "summary 1"
    assert memory.archive.turns(1, 1)[1][0].content == "a: question 1"
    assert memory.archive.summarized_turns() == {t: cluster.summary.id for t in range(1, 9)}


def test_the_prompt_keeps_facts_literally_and_the_output_is_text_plus_anchors(
    summary_model,
) -> None:
    memory = memory_for(summary_model)
    add(memory, *"a" * 10)

    system, request = summary_model.prompts[0]
    assert "Keep names, identifiers, numbers, dates and decisions literally" in system.text
    assert "Invent nothing" in system.text
    assert "a: question 1" in request.text
    (cluster,) = memory.clusters()
    assert cluster.summary is not None
    assert (cluster.summary.subject, cluster.summary.period, cluster.summary.names) == (
        "subject 1",
        "period 1",
        ["name 1"],
    )


def test_incremental_summarizes_the_old_summary_plus_the_new_turns(summary_model) -> None:
    memory = memory_for(summary_model)
    add(memory, *"a" * 10)  # turns 1-8 summarized as "summary 1"
    add(memory, *"a" * 5)  # turn 15 opens: turns 9-13 join the cluster

    request = summary_model.prompts[1][1].text
    assert "Previous summary:\nsummary 1" in request
    assert "a: question 9" in request and "a: question 13" in request
    assert "a: question 8" not in request
    summary = memory.clusters()[0].summary
    assert summary is not None and summary.text == "summary 2"


def test_full_summarizes_all_original_turns(summary_model) -> None:
    memory = memory_for(summary_model, resummarize_strategy="full")
    add(memory, *"a" * 15)

    request = summary_model.prompts[1][1].text
    assert "summary 1" not in request
    assert "a: question 1\n" in request and "a: question 13" in request


def test_a_failed_summary_leaves_the_cluster_unsummarized_and_is_retried(caplog) -> None:
    model = FakeSummaryModel(fail_times=1)
    reports = []
    memory = memory_for(model, on_compact=reports.append)
    add(memory, *"a" * 9)
    memory.add_message(HumanMessage("a: question 10"))  # the round fails

    assert memory.clusters()[0].summary is None
    assert not memory.window()[0].text.startswith("Conversation memory")
    assert "chat model down" in reports[-1].errors[0]
    assert "chat model down" in caplog.text

    memory.add_message(AIMessage("answer 10"))  # the next round retries

    assert memory.clusters()[0].summary is not None
    assert reports[-1].errors == []


def test_a_summary_barely_shorter_than_its_originals_is_accepted_and_flagged() -> None:
    reports = []
    long_text = " ".join(f"a: question {n} answer {n}" for n in range(1, 9))
    memory = memory_for(FakeSummaryModel(text=long_text), on_compact=reports.append)
    add(memory, *"a" * 10)

    (report,) = [r for r in reports if r.summaries]
    assert report.summaries[0].poor_compression
    assert memory.clusters()[0].summary is not None


def test_a_short_summary_is_not_flagged(summary_model) -> None:
    reports = []
    memory = memory_for(summary_model, on_compact=reports.append)
    add(memory, *"a" * 10)

    (report,) = [r for r in reports if r.summaries]
    assert not report.summaries[0].poor_compression


def tool_turn(memory: ConversationClusterMemory, n: int, result: str) -> None:
    memory.add_message(HumanMessage(f"a: fetch {n}"))
    call = {"name": "fetch", "args": {"id": n}, "id": f"t{n}"}
    memory.add_message(AIMessage("", tool_calls=[call]))
    memory.add_message(ToolMessage(result, tool_call_id=f"t{n}"))
    memory.add_message(AIMessage(f"done {n}"))


def test_the_summary_model_reads_tool_calls_and_their_arguments(summary_model) -> None:
    memory = memory_for(summary_model, keep_recent_turns=3, min_batch_turns=1)
    tool_turn(memory, 1, "ok")
    add(memory, *"aaaa")

    request = summary_model.prompts[0][1].text
    assert 'Assistant calls fetch({"id": 1})' in request
    assert "Tool result: ok" in request


def test_large_tool_results_are_shortened_to_their_header_only_in_earlier_turns() -> None:
    memory = memory_for(FakeSummaryModel(), keep_recent_turns=3, max_tool_result_tokens=50)
    large, small = "word " * 200, "word " * 10
    tool_turn(memory, 1, large)
    tool_turn(memory, 2, small)
    tool_turn(memory, 3, large)

    results = [m.text for m in memory.window() if isinstance(m, ToolMessage)]
    assert results == ["Tool result fetch: 1000 characters", small, large]
    assert memory.archive.turns(1, 1)[1][2].text == large


def test_the_tool_result_limit_defaults_to_2000_tokens(summary_model) -> None:
    memory = memory_for(summary_model, keep_recent_turns=3)
    tool_turn(memory, 1, "word " * 1900)
    tool_turn(memory, 2, "word " * 2100)
    tool_turn(memory, 3, "x")

    results = [m.text for m in memory.window() if isinstance(m, ToolMessage)]
    assert results == ["word " * 1900, "Tool result fetch: 10500 characters", "x"]


def test_background_returns_before_the_round_ends() -> None:
    model = FakeSummaryModel(gate=threading.Event())
    memory = memory_for(model, auto_compact="background")
    add(memory, *"a" * 10)  # returns while the round waits in the model

    assert model.started.wait(5)
    assert memory.clusters() == []  # the round commits its clusters with their summaries

    model.gate.set()
    assert memory.wait_idle(5)
    assert memory.clusters()[0].summary is not None


def test_inline_returns_after_the_round(summary_model) -> None:
    memory = memory_for(summary_model, auto_compact="inline")
    add(memory, *"a" * 9)
    memory.add_message(HumanMessage("a: question 10"))

    assert memory.clusters()[0].summary is not None


def test_off_runs_no_round_until_compact(summary_model) -> None:
    memory = memory_for(summary_model, auto_compact="off")
    add(memory, *"a" * 10)

    assert memory.vectors.turns() == set()
    assert memory.clusters() == []

    report = memory.compact()

    assert report.clustered_turns == list(range(1, 9))
    assert memory.clusters()[0].summary is not None


def test_acompact_runs_a_round(summary_model) -> None:
    memory = memory_for(summary_model, auto_compact="off")
    add(memory, *"a" * 10)

    report = asyncio.run(memory.acompact())

    assert len(report.summaries) == 1


def test_one_round_at_a_time_and_work_during_a_round_starts_another() -> None:
    model = FakeSummaryModel(gate=threading.Event())
    reports = []
    memory = memory_for(model, auto_compact="background", on_compact=reports.append)
    add(memory, *"a" * 10)
    assert model.started.wait(5)
    add(memory, *"b" * 5)  # during the round: turn 15 opens a check; turns 9-10 join, 11-13 are new

    model.gate.set()
    assert memory.wait_idle(5)

    assert model.max_active == 1
    assert [list(c.turns) for c in memory.clusters()] == [[*range(1, 11)], [11, 12, 13]]
    assert all(c.summary is not None for c in memory.clusters())


def test_window_during_a_round_returns_the_last_committed_state(summary_model) -> None:
    memory = memory_for(summary_model, auto_compact="background")
    add(memory, *"a" * 10)
    assert memory.wait_idle(5)
    committed = memory.window()
    summary_model.gate = threading.Event()
    summary_model.started.clear()
    add(memory, *"a" * 4)
    memory.add_message(HumanMessage("a: question 15"))  # turns 9-13 join; resummarizing

    assert summary_model.started.wait(5)
    assert memory.window()[0] == committed[0]  # still the old summary, nothing half-done

    summary_model.gate.set()
    assert memory.wait_idle(5)
    assert "summary 2" in memory.window()[0].text


class Interrupted(BaseException):
    """Stands in for a process that stops mid-round."""


class InterruptingModel(FakeSummaryModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise Interrupted


def test_a_round_interrupted_before_commit_is_redone_on_the_next_add_message(
    tmp_path: Path,
) -> None:
    def open_memory(model: FakeSummaryModel) -> ConversationClusterMemory:
        return memory_for(model, path=tmp_path)

    first = open_memory(InterruptingModel())
    add(first, *"a" * 9)
    before = first.window()
    with pytest.raises(Interrupted):
        first.add_message(HumanMessage("a: question 10"))

    model = FakeSummaryModel()
    reopened = open_memory(model)
    assert reopened.window()[:-1] == before[2:]  # the previous state, plus the new message
    assert reopened.clusters() == []  # nothing of the interrupted round was committed

    reopened.add_message(AIMessage("answer 10"))

    assert reopened.clusters()[0].summary is not None
    assert len(model.prompts) == 1


def test_wait_idle_returns_when_no_round_is_running(summary_model) -> None:
    memory = memory_for(summary_model, auto_compact="background")

    assert memory.wait_idle(1)

    summary_model.gate = threading.Event()
    add(memory, *"a" * 10)
    assert summary_model.started.wait(5)

    assert not memory.wait_idle(0.1)

    summary_model.gate.set()
    assert memory.wait_idle(5)


def test_each_report_reaches_on_compact_and_the_log_with_tokens(summary_model, caplog) -> None:
    caplog.set_level(logging.INFO, logger="conversation_cluster_memory")
    reports = []
    memory = memory_for(summary_model, on_compact=reports.append)
    add(memory, *"a" * 10)

    (report,) = [r for r in reports if r.summaries]
    assert report.clustered_turns == list(range(1, 9))
    assert report.window_tokens_before > report.window_tokens_after > 0
    assert (report.input_tokens, report.output_tokens) == (100, 10)
    assert "window_tokens_before" in caplog.text


class InterruptingSecondCall(FakeSummaryModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.prompts:
            raise Interrupted
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_a_round_commits_all_its_clusters_and_summaries_at_once(tmp_path: Path) -> None:
    memory = memory_for(InterruptingSecondCall(), path=tmp_path)
    add(memory, *"aaaabbb")
    memory.add_message(HumanMessage("b: question 8"))
    add(memory, "c")
    with pytest.raises(Interrupted):  # the first cluster is summarized, the second is not
        memory.add_message(HumanMessage("c: question 10"))

    reopened = memory_for(FakeSummaryModel(), path=tmp_path)

    assert reopened.clusters() == []
    assert not reopened.window()[0].text.startswith("Conversation memory")
