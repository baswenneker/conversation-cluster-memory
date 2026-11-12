"""Cluster summaries: the summary model's prompt and output, token counts, round reports."""

import functools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

DEFAULT_SUMMARY_MODEL = "gpt-6.1-luna"

# A summary keeping more than this share of its originals' tokens is flagged as poor
# compression. It is still accepted: a poor summary beats verbatim turns that never shrink.
POOR_COMPRESSION_RATIO = 0.8

SUMMARY_PROMPT = """\
You summarize one topic of a conversation between a user and an assistant, so that the \
assistant can continue the conversation later from the summary alone.

- Keep names, identifiers, numbers, dates and decisions literally, exactly as written.
- Invent nothing: leave out what the conversation does not say.
- When a previous summary is given, return one summary that covers it and the new turns.

Return the summary text plus anchors: the subject, the period, and the names mentioned."""

MERGE_PROMPT = """\
You merge the summaries of several earlier topics of a conversation between a user and an \
assistant into one shorter summary of that period, so that the assistant can continue the \
conversation later from it alone.

- Keep names, identifiers, numbers, dates and decisions literally, exactly as written.
- Invent nothing: leave out what the summaries do not say.
- Keep every topic recognisable; drop repetition and wording, not facts.
- Stay within the length budget given with the summaries: merges repeat over a long \
conversation, so the summary must not grow with every topic.

Return the summary text plus anchors: the subject, the period, and the names mentioned."""


class Summary(BaseModel):
    """A summary of one topic of a conversation, with anchors to find it again."""

    text: str = Field(description="The summary of the topic.")
    subject: str = Field(description="The topic, in a few words.")
    period: str = Field(description="The period the turns cover, as given in the input.")
    names: list[str] = Field(
        description="Names and identifiers of people, things and places mentioned, literally."
    )


@dataclass(frozen=True)
class ClusterSummary:
    """A stored summary: the current one of its cluster replaces the cluster's turns."""

    id: int
    text: str
    subject: str
    period: str
    names: list[str]


@dataclass
class SummaryReport:
    cluster: int
    turns: tuple[int, ...]
    original_tokens: int  # the cluster's turns as the window would hold them verbatim
    summary_tokens: int
    poor_compression: bool


@dataclass
class CompactionReport:
    """What one compaction round did.

    `window_tokens_before` is the window at the start of the round with the turns this round
    summarized still verbatim; `window_tokens_after` is the window when the round ended.
    `input_tokens` and `output_tokens` are the summary model's usage, as the API reported it.
    """

    clustered_turns: list[int] = field(default_factory=list)
    summaries: list[SummaryReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    window_tokens_before: int = 0
    window_tokens_after: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def summarize(
    model: BaseChatModel, transcript: str, previous: str | None, period: str
) -> tuple[Summary, dict[str, int]]:
    """Summarize the transcript (and the previous summary); returns the summary and its usage."""
    parts = [f"Period: {period}"]
    if previous is not None:
        parts.append(f"Previous summary:\n{previous}")
    parts.append(f"{'New turns' if previous is not None else 'Turns'}:\n{transcript}")
    return _structured(model, SUMMARY_PROMPT, "\n\n".join(parts))


def merge_summaries(
    model: BaseChatModel, summaries: Sequence[ClusterSummary], period: str, budget: int
) -> tuple[Summary, dict[str, int]]:
    """One period summary of the summaries in at most about `budget` tokens; returns it and
    its usage."""
    merged = "\n\n".join(previous_text(s) for s in summaries)
    return _structured(
        model,
        MERGE_PROMPT,
        f"Period: {period}\n\nLength budget: at most {budget} tokens.\n\nSummaries:\n{merged}",
    )


def _structured(model: BaseChatModel, prompt: str, request: str) -> tuple[Summary, dict[str, int]]:
    result = model.with_structured_output(Summary, include_raw=True).invoke(
        [SystemMessage(prompt), HumanMessage(request)]
    )
    assert isinstance(result, dict)
    if result.get("parsing_error") is not None:
        raise result["parsing_error"]
    summary = result.get("parsed")
    if not isinstance(summary, Summary):
        raise ValueError(f"the summary model returned no summary: {result.get('raw')!r}")
    raw = result.get("raw")
    usage = raw.usage_metadata if isinstance(raw, AIMessage) else None
    return summary, dict(usage or {})


def transcript(turns: Sequence[Sequence[BaseMessage]]) -> str:
    """The turns as plain text for the summary model."""
    speaker = {"human": "User", "ai": "Assistant", "tool": "Tool result"}
    lines = []
    for messages in turns:
        for message in messages:
            if message.text:
                lines.append(f"{speaker.get(message.type, message.type)}: {message.text}")
            if isinstance(message, AIMessage):
                lines += [
                    f"Assistant calls {call['name']}({json.dumps(call['args'])})"
                    for call in message.tool_calls
                ]
    return "\n".join(lines)


def previous_text(summary: ClusterSummary) -> str:
    """A stored summary as the summary model reads it: the text and its anchors."""
    return (
        f"{summary.text}\nSubject: {summary.subject}\nPeriod: {summary.period}\n"
        f"Names: {', '.join(summary.names)}"
    )


def memory_text(summaries: Sequence[ClusterSummary]) -> str:
    """The content of the "Conversation memory" system message."""
    lines = [f"[{s.period}] {s.subject}: {s.text}" for s in summaries]
    return "Conversation memory\n\n" + "\n\n".join(lines)


@functools.cache
def _encoding(model: str) -> tiktoken.Encoding:
    """The model's tiktoken tokenizer; o200k_base for an unknown model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def token_counter(model: str) -> Callable[[str], int]:
    encoding = _encoding(model)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def first_tokens(model: str, text: str, limit: int) -> str:
    """The start of the text, at most `limit` tokens of the model's tokenizer."""
    encoding = _encoding(model)
    return encoding.decode(encoding.encode(text, disallowed_special=())[:limit])
