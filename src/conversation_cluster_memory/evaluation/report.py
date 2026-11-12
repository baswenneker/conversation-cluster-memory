"""The token metric, the quality check and the README block of an evaluation run."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from conversation_cluster_memory.summaries import CompactionReport


class EvaluationError(RuntimeError):
    """A run that cannot produce a result."""


class MissingTokenMeasurementError(EvaluationError):
    """A run has no token measurement for a compaction round, so its savings are unknown."""


@dataclass(frozen=True)
class RoundTokens:
    """The window's tokens before and after one compaction round."""

    before: int
    after: int

    @property
    def saving(self) -> float:
        return (self.before - self.after) / self.before


def token_rounds(reports: Sequence[CompactionReport]) -> list[RoundTokens]:
    """Per compaction round that summarized something: the window's tokens before and after.

    Raises `MissingTokenMeasurementError` when no round summarized anything or a round has no
    measurement: a saving that was never measured must not pass as one.
    """
    rounds = [
        RoundTokens(r.window_tokens_before, r.window_tokens_after) for r in reports if r.summaries
    ]
    if not rounds:
        raise MissingTokenMeasurementError("no compaction round summarized anything")
    if any(r.before <= 0 for r in rounds):
        raise MissingTokenMeasurementError("a compaction round has no window token count")
    return rounds


UNCOMPRESSED = "uncompressed"
COMPRESSED = "compressed"


@dataclass
class RunResult:
    """One evaluation run: what the README block shows and the results file keeps.

    `scores` holds, per condition, the mean score per metric over all questions and rounds.
    `cost` is in US dollars per role; None when no price is known for the model.
    """

    date: datetime
    models: dict[str, str]
    scores: dict[str, dict[str, float]]
    tokens: dict[str, list[RoundTokens]]
    cost: dict[str, float | None]
    min_ratio: float
    results_file: str

    def score(self, condition: str) -> float:
        values = self.scores[condition].values()
        return sum(values) / len(values)

    @property
    def ratio(self) -> float:
        uncompressed = self.score(UNCOMPRESSED)
        return self.score(COMPRESSED) / uncompressed if uncompressed else 1.0

    @property
    def passed(self) -> bool:
        # Rounded, so a ratio that is exactly the minimum is not failed by a float error.
        return round(self.ratio, 9) >= self.min_ratio


START = "<!-- eval-results:start -->"
END = "<!-- eval-results:end -->"


def update_readme(readme: str, run: RunResult) -> str:
    """The README with only the text between the evaluation markers replaced by the run."""
    start, end = readme.find(START), readme.find(END)
    if start < 0 or end < start:
        raise ValueError(f"the README has no {START} ... {END} block")
    return readme[: start + len(START)] + "\n" + render_block(run) + readme[end:]


def _money(value: float | None) -> str:
    return "n/a (no price known)" if value is None else f"${value:.4f}"


def render_block(run: RunResult) -> str:
    """The README block for the run, in Markdown, ending with a newline."""
    lines = [
        f"These numbers come from the latest local run (`uv run ccm-eval run`) on "
        f"{run.date:%Y-%m-%d %H:%M} UTC, not from CI. "
        f"Every run is kept in [`{run.results_file}`]({run.results_file}).",
        "",
        "Models: " + ", ".join(f"{role} `{name}`" for role, name in run.models.items()) + ".",
        "",
        "| Metric | Uncompressed window | Compressed window + retrieval |",
        "|---|---|---|",
    ]
    lines += [
        f"| {metric} | {score:.2f} | {run.scores[COMPRESSED].get(metric, 0.0):.2f} |"
        for metric, score in run.scores[UNCOMPRESSED].items()
    ]
    lines += [
        f"| **Mean** | **{run.score(UNCOMPRESSED):.2f}** | **{run.score(COMPRESSED):.2f}** |",
        "",
        f"Compressed scores {run.ratio:.0%} of uncompressed (minimum {run.min_ratio:.0%}): "
        f"**{'passed' if run.passed else 'failed'}**.",
        "",
        "| Conversation | Compaction rounds | Mean window tokens before | after | Mean saving |",
        "|---|---|---|---|---|",
    ]
    for name, rounds in run.tokens.items():
        n = len(rounds)
        lines.append(
            f"| {name} | {n} | {sum(r.before for r in rounds) / n:.0f} "
            f"| {sum(r.after for r in rounds) / n:.0f} "
            f"| {sum(r.saving for r in rounds) / n:.0%} |"
        )
    lines += [
        "",
        "Cost: " + ", ".join(f"{role} {_money(cost)}" for role, cost in run.cost.items()) + ".",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SweepRow:
    setting: str
    value: str
    run: RunResult | None  # None: no compaction round was measured with this value
    error: str = ""


def sweep_table(rows: Sequence[SweepRow]) -> str:
    """The sweep as a Markdown table: one row per parameter value."""
    lines = [
        "| Setting | Value | Uncompressed | Compressed | Compressed / uncompressed "
        "| Mean token saving | Judge cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.run is None:
            lines.append(f"| {row.setting} | {row.value} | {row.error} | | | | |")
            continue
        rounds = [r for conversation in row.run.tokens.values() for r in conversation]
        saving = sum(r.saving for r in rounds) / len(rounds)
        lines.append(
            f"| {row.setting} | {row.value} | {row.run.score(UNCOMPRESSED):.2f} "
            f"| {row.run.score(COMPRESSED):.2f} | {row.run.ratio:.0%} | {saving:.0%} "
            f"| {_money(row.run.cost.get('judge'))} |"
        )
    return "\n".join(lines) + "\n"
