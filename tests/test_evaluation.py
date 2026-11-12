from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.language_models import FakeListChatModel

from conversation_cluster_memory import CompactionReport
from conversation_cluster_memory.evaluation.cli import main
from conversation_cluster_memory.evaluation.conversations import generate, load
from conversation_cluster_memory.evaluation.harness import Settings, Usage, replay
from conversation_cluster_memory.evaluation.report import (
    COMPRESSED,
    UNCOMPRESSED,
    MissingTokenMeasurementError,
    RoundTokens,
    RunResult,
    token_rounds,
    update_readme,
)
from conversation_cluster_memory.summaries import SummaryReport

CONVERSATIONS_DIR = Path(__file__).parent.parent / "eval" / "conversations"


def round_report(before: int, after: int) -> CompactionReport:
    return CompactionReport(
        summaries=[SummaryReport(1, (1, 2), 300, 30, False)],
        window_tokens_before=before,
        window_tokens_after=after,
    )


def test_the_token_metric_reports_before_after_and_saving_per_round() -> None:
    rounds = token_rounds([round_report(1000, 400), round_report(800, 600)])

    assert [(r.before, r.after) for r in rounds] == [(1000, 400), (800, 600)]
    assert [r.saving for r in rounds] == [0.6, 0.25]


def test_the_token_metric_fails_without_a_compaction_round() -> None:
    with pytest.raises(MissingTokenMeasurementError):
        token_rounds([CompactionReport()])


def test_the_token_metric_fails_when_a_round_has_no_measurement() -> None:
    with pytest.raises(MissingTokenMeasurementError):
        token_rounds([round_report(1000, 400), round_report(0, 0)])


def run_result(uncompressed: float, compressed: float, min_ratio: float = 0.9) -> RunResult:
    return RunResult(
        date=datetime(2026, 11, 12, 14, 3, tzinfo=UTC),
        models={"embedding": "embed-x", "summary": "sum-y", "answer": "ans-z", "judge": "judge-w"},
        scores={
            "uncompressed": {"Faithfulness": uncompressed, "Answer relevancy": uncompressed},
            "compressed": {"Faithfulness": compressed, "Answer relevancy": compressed},
        },
        tokens={"short": [RoundTokens(1000, 400)], "long": [RoundTokens(800, 600)]},
        cost={"embedding": 0.0, "summary": 0.0123, "answer": 0.5, "judge": None},
        min_ratio=min_ratio,
        results_file="eval/results/2026-11-12T140300Z.json",
    )


def test_a_run_passes_at_the_minimum_share_of_the_uncompressed_score() -> None:
    assert run_result(0.8, 0.72).passed
    assert not run_result(0.8, 0.71).passed


def test_the_minimum_share_is_configurable() -> None:
    assert run_result(0.8, 0.6, min_ratio=0.75).passed
    assert not run_result(0.8, 0.6, min_ratio=0.8).passed


README = """# Title

Intro text.

<!-- eval-results:start -->
_old results_
<!-- eval-results:end -->

## After
Closing text.
"""


def test_the_readme_updater_replaces_only_the_marked_block() -> None:
    updated = update_readme(README, run_result(0.8, 0.76))

    before, rest = updated.split("<!-- eval-results:start -->\n")
    block, after = rest.split("<!-- eval-results:end -->")
    assert before == "# Title\n\nIntro text.\n\n"
    assert after == "\n\n## After\nClosing text.\n"
    assert "_old results_" not in block


def test_the_readme_block_shows_the_latest_run() -> None:
    block = update_readme(README, run_result(0.8, 0.76))

    assert "2026-11-12 14:03 UTC" in block
    for model in ("embed-x", "sum-y", "ans-z", "judge-w"):
        assert model in block
    assert "| Faithfulness | 0.80 | 0.76 |" in block
    assert "95%" in block  # compressed / uncompressed
    assert "60%" in block and "25%" in block  # token saving per conversation
    assert "$0.0123" in block and "$0.5000" in block
    assert "eval/results/2026-11-12T140300Z.json" in block
    assert "latest local run" in block


def test_the_readme_updater_refuses_a_readme_without_the_markers() -> None:
    with pytest.raises(ValueError, match="eval-results:start"):
        update_readme("# Title\n", run_result(0.8, 0.76))


def test_the_committed_conversations_match_their_generator() -> None:
    committed = load(CONVERSATIONS_DIR)

    assert committed == generate()
    assert [(c.name, len(c.turns)) for c in committed] == [
        ("short", 30),
        ("long", 200),
        ("switching", 80),
    ]
    for conversation in committed:
        assert len(conversation.questions) == 15
        assert all(
            q.answer and 1 <= q.turn <= len(conversation.turns) for q in conversation.questions
        )


def test_a_run_without_an_api_key_fails_with_a_clear_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    assert main(["run"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_after_every_compaction_round_the_same_questions_are_asked_in_both_conditions(
    embeddings, summary_model
) -> None:
    short = load(CONVERSATIONS_DIR)[0]
    answers = FakeListChatModel(responses=["an answer"])
    usage = {"answer": Usage(), "embedding text": Usage()}

    cases, reports = replay(short, Settings(), embeddings, summary_model, answers, usage)

    rounds = token_rounds(reports)
    assert rounds
    asked: dict[tuple[int, str], list[str]] = {}
    for case in cases:
        asked.setdefault((case.round, case.condition), []).append(case.question)
    assert {r for r, _ in asked} == set(range(1, len(rounds) + 1))
    for r in range(1, len(rounds) + 1):
        assert asked[(r, UNCOMPRESSED)] == asked[(r, COMPRESSED)]
        assert asked[(r, COMPRESSED)]
