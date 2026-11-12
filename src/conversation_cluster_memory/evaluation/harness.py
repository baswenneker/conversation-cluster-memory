"""Replays the conversations, asks the questions after every compaction round in two
conditions, and scores the answers with DeepEval. Calls OpenAI: run by hand, never in the gate.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from conversation_cluster_memory import ConversationClusterMemory, default_embedding_text
from conversation_cluster_memory.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    default_embeddings,
    openai_api_key,
)
from conversation_cluster_memory.evaluation.conversations import Conversation
from conversation_cluster_memory.evaluation.report import (
    COMPRESSED,
    UNCOMPRESSED,
    EvaluationError,
    RunResult,
    token_rounds,
)
from conversation_cluster_memory.retrieval import hit_text
from conversation_cluster_memory.summaries import DEFAULT_SUMMARY_MODEL, CompactionReport

DEFAULT_JUDGE_MODEL = "gpt-6.1-terra"

ANSWER_PROMPT = (
    "Answer the user's last question from the conversation above, in one or two sentences. "
    "If the conversation does not say, answer that you do not know."
)

TURN_SUMMARY_PROMPT = "Summarize this conversation turn in one sentence. Keep names and numbers."


@dataclass
class Case:
    """One question asked in one condition after one compaction round."""

    conversation: str
    round: int
    condition: str
    question: str
    expected: str
    answer: str
    context: list[str]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, message: BaseMessage) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


@dataclass
class Settings:
    """The memory parameters a run (or one sweep cell) uses; the rest stay at their defaults."""

    params: dict[str, Any] = field(default_factory=dict)
    embedding_text: str = "default"  # or "summary": a one-sentence LLM summary of the turn


def models() -> dict[str, str]:
    """The model per role, from the environment or the defaults."""
    return {
        "embedding": DEFAULT_EMBEDDING_MODEL,
        "summary": os.environ.get("OPENAI_SUMMARY_MODEL") or DEFAULT_SUMMARY_MODEL,
        "answer": os.environ.get("OPENAI_ANSWER_MODEL") or DEFAULT_SUMMARY_MODEL,
        "judge": os.environ.get("OPENAI_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
    }


def _chat(model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=openai_api_key())


def _summary_text(model: BaseChatModel, usage: Usage) -> Callable[[Sequence[BaseMessage]], str]:
    """The embedding-text option that embeds a short LLM summary of the turn."""

    def text(messages: Sequence[BaseMessage]) -> str:
        reply = model.invoke(
            [SystemMessage(TURN_SUMMARY_PROMPT), HumanMessage(default_embedding_text(messages))]
        )
        usage.add(reply)
        return reply.text

    return text


def _texts(window: Sequence[BaseMessage]) -> list[str]:
    """The window as retrieval context for the judge: one node per message with text."""
    return [f"{m.type}: {m.text}" for m in window if m.text]


def replay(
    conversation: Conversation,
    settings: Settings,
    embeddings: Embeddings,
    summary_model: BaseChatModel,
    answer_model: BaseChatModel,
    usage: dict[str, Usage],
) -> tuple[list[Case], list[CompactionReport]]:
    """Replay with inline compaction; after every round that summarized something, ask every
    question whose fact is replayed already, with the uncompressed and the compressed window."""
    reports: list[CompactionReport] = []
    embedding_text = (
        _summary_text(summary_model, usage["embedding text"])
        if settings.embedding_text == "summary"
        else default_embedding_text
    )
    memory = ConversationClusterMemory(
        f"eval-{conversation.name}",
        embeddings=embeddings,
        summary_model=summary_model,
        embedding_text=embedding_text,
        auto_compact="inline",
        auto_retrieval=False,  # retrieval below searches the question, not the last message
        on_compact=reports.append,
        **settings.params,
    )
    # The uncompressed window: every turn verbatim (earlier large tool results shortened to
    # their header, as the memory does), so the comparison isolates compaction.
    uncompressed = ConversationClusterMemory(
        f"eval-{conversation.name}-uncompressed",
        embeddings=embeddings,
        keep_recent_turns=len(conversation.turns) + 1,
        auto_compact="off",
        auto_retrieval=False,
        max_tool_result_tokens=memory.max_tool_result_tokens,
    )
    cases: list[Case] = []
    rounds = 0
    for number, turn in enumerate(conversation.turns, start=1):
        seen = len(reports)
        for message in turn:
            memory.add_message(message)
            uncompressed.add_message(message)
        if not any(r.summaries for r in reports[seen:]):
            continue
        rounds += 1
        compressed_window = memory.window()
        first_turn = next(
            i for i, m in enumerate(compressed_window) if not isinstance(m, SystemMessage)
        )
        uncompressed_window = uncompressed.window()
        summarized = {c.id for c in memory.clusters() if c.summary is not None}
        for q in conversation.questions:
            if q.turn > number:
                continue
            # As window() does: summarized clusters are excluded before k and the budget apply.
            hits = memory._search(q.question, memory.retrieval_k, exclude=summarized)
            windows = {
                UNCOMPRESSED: uncompressed_window,
                # Hits go where window() puts them: after the system messages.
                COMPRESSED: [
                    *compressed_window[:first_turn],
                    *(SystemMessage(hit_text(h)) for h in hits),
                    *compressed_window[first_turn:],
                ],
            }
            for condition, window in windows.items():
                reply = answer_model.invoke(
                    [*window, HumanMessage(q.question), SystemMessage(ANSWER_PROMPT)]
                )
                usage["answer"].add(reply)
                cases.append(
                    Case(
                        conversation.name,
                        rounds,
                        condition,
                        q.question,
                        q.answer,
                        reply.text,
                        _texts(window),
                    )
                )
    return cases, reports


def _metrics(judge: Any) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    return [
        AnswerRelevancyMetric(model=judge),
        ContextualPrecisionMetric(model=judge),
        ContextualRecallMetric(model=judge),
        FaithfulnessMetric(model=judge),
        GEval(
            name="Factual correctness",
            criteria="Does the actual output state the same facts as the expected output? "
            "Penalize any name, number or date that differs or is missing.",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=judge,
        ),
    ]


def judge(cases: list[Case], judge_model: str) -> tuple[dict[str, dict[str, float]], float | None]:
    """DeepEval scores per condition (mean per metric) and the judging cost in dollars."""
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase

    model = OpenAIModel(model=judge_model, api_key=openai_api_key())
    test_cases = [
        LLMTestCase(
            name=str(i),
            input=c.question,
            actual_output=c.answer,
            expected_output=c.expected,
            retrieval_context=c.context,
        )
        for i, c in enumerate(cases)
    ]
    result = evaluate(
        test_cases,
        _metrics(model),
        async_config=AsyncConfig(max_concurrent=8),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
        # A metric the judge fails on comes back unscored instead of ending the paid run.
        error_config=ErrorConfig(ignore_errors=True),
    )
    collected: dict[str, dict[str, list[float]]] = {UNCOMPRESSED: {}, COMPRESSED: {}}
    cost: float | None = 0.0
    for test in result.test_results:
        condition = cases[int(test.name)].condition
        for metric in test.metrics_data or []:
            if metric.score is not None:  # not scored (a judge error): left out of the mean
                collected[condition].setdefault(metric.name, []).append(metric.score)
            if metric.evaluation_cost is None:
                cost = None
            elif cost is not None:
                cost += metric.evaluation_cost
    for condition, metrics in collected.items():
        if not metrics:
            raise EvaluationError(
                f"the judge scored no answer in the {condition} condition; "
                "check OPENAI_JUDGE_MODEL, the API key and the DeepEval output above"
            )
    scores = {
        condition: {name: sum(v) / len(v) for name, v in metrics.items()}
        for condition, metrics in collected.items()
    }
    return scores, cost


def _price(model: str, usage: Usage) -> float | None:
    """Dollars for the usage, from DeepEval's price list; None for a model it does not price."""
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA

    data = OPENAI_MODELS_DATA.get(model)
    if data is None or data.input_price is None or data.output_price is None:
        return None
    return usage.input_tokens * data.input_price + usage.output_tokens * data.output_price


def run(
    conversations: list[Conversation],
    settings: Settings,
    min_ratio: float,
    results_file: str,
    date: datetime,
) -> tuple[RunResult, list[Case]]:
    """One full evaluation run. Raises `MissingTokenMeasurementError` when a conversation
    has no measured compaction round."""
    names = models()
    embeddings = default_embeddings()
    summary_model = _chat(names["summary"])
    answer_model = _chat(names["answer"])
    usage = {"answer": Usage(), "embedding text": Usage()}
    cases: list[Case] = []
    tokens = {}
    summary_usage = Usage()
    for conversation in conversations:
        found, reports = replay(
            conversation, settings, embeddings, summary_model, answer_model, usage
        )
        cases += found
        tokens[conversation.name] = token_rounds(reports)
        summary_usage.input_tokens += sum(r.input_tokens for r in reports)
        summary_usage.output_tokens += sum(r.output_tokens for r in reports)
    # The embedding-text option calls the summary model too.
    summary_usage.input_tokens += usage["embedding text"].input_tokens
    summary_usage.output_tokens += usage["embedding text"].output_tokens
    scores, judge_cost = judge(cases, names["judge"])
    result = RunResult(
        date=date,
        models=names,
        scores=scores,
        tokens=tokens,
        cost={
            "embedding": 0.0,  # the local model runs on this machine
            "summary": _price(names["summary"], summary_usage),
            "answer": _price(names["answer"], usage["answer"]),
            "judge": judge_cost,
        },
        min_ratio=min_ratio,
        results_file=results_file,
    )
    return result, cases
