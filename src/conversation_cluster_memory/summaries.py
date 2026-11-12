"""Cluster summaries: the summary model's prompt and output, token counts, round reports."""

import functools
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
    result = model.with_structured_output(Summary, include_raw=True).invoke(
        [SystemMessage(SUMMARY_PROMPT), HumanMessage("\n\n".join(parts))]
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
    lines = []
    for messages in turns:
        for message in messages:
            if message.text:
                speaker = {"human": "User", "ai": "Assistant", "tool": "Tool result"}
                lines.append(f"{speaker.get(message.type, message.type)}: {message.text}")
    return "\n".join(lines)


def memory_text(summaries: Sequence[ClusterSummary]) -> str:
    """The content of the "Conversation memory" system message."""
    lines = [f"[{s.period}] {s.subject}: {s.text}" for s in summaries]
    return "Conversation memory\n\n" + "\n\n".join(lines)


@functools.cache
def token_counter(model: str) -> Callable[[str], int]:
    """Counts tokens with the model's tiktoken tokenizer; o200k_base for an unknown model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))
