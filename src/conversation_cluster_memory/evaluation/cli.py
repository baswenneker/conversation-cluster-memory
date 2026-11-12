"""`ccm-eval`: run the evaluation, sweep parameters, estimate a year's cost, or regenerate
the conversations.

Runs locally, by hand: `run` and `sweep` call OpenAI with OPENAI_API_KEY from the environment
or `.env`; `cost` and `generate` call no model.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from conversation_cluster_memory.embeddings import MissingOpenAIKeyError, openai_api_key
from conversation_cluster_memory.evaluation import conversations
from conversation_cluster_memory.evaluation.report import (
    EvaluationError,
    SweepRow,
    sweep_table,
    update_readme,
)

# One parameter changes at a time, the others stay at their defaults (the first row).
SWEEP: list[tuple[str, list[object]]] = [
    ("defaults", ["-"]),
    ("similarity_threshold", [0.6, 0.7]),
    ("keep_recent_turns", [5, 20]),
    ("min_batch_turns", [4, 12]),
    ("embedding_text", ["summary"]),
    ("resummarize_strategy", ["full"]),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ccm-eval", description=__doc__)
    parser.add_argument("--conversations", type=Path, default=Path("eval/conversations"))
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the evaluation and update the README block")
    run.add_argument("--readme", type=Path, default=Path("README.md"))
    run.add_argument(
        "--min-ratio",
        type=float,
        default=0.9,
        help="fail below this share of the uncompressed score (default 0.9)",
    )
    commands.add_parser("sweep", help="compare parameter values and write a table")
    commands.add_parser("generate", help="regenerate the committed conversations")
    cost = commands.add_parser(
        "cost", help="estimate a year's cost per conversation from the latest run, offline"
    )
    cost.add_argument("--turns-per-week", type=float, required=True)
    args = parser.parse_args(argv)

    if args.command == "generate":
        conversations.write(args.conversations)
        return 0
    if args.command == "cost":
        return _cost(args)
    try:
        openai_api_key()
    except MissingOpenAIKeyError as error:
        print(f"ccm-eval: {error}", file=sys.stderr)
        return 2
    # DeepEval sends usage telemetry unless told not to; this run stays local.
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    loaded = conversations.load(args.conversations)
    args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        return _run(loaded, args)
    return _sweep(loaded, args)


def _run(loaded: list[conversations.Conversation], args: argparse.Namespace) -> int:
    from conversation_cluster_memory.evaluation import harness

    now = datetime.now(UTC)
    results_file = args.results / f"{now:%Y-%m-%dT%H%M%SZ}.json"
    try:
        result, cases = harness.run(
            loaded, harness.Settings(), args.min_ratio, results_file.as_posix(), now
        )
    except EvaluationError as error:
        print(f"ccm-eval: run failed: {error}", file=sys.stderr)
        return 1
    data = {**asdict(result), "passed": result.passed, "cases": [asdict(c) for c in cases]}
    results_file.write_text(json.dumps(data, indent=1, default=str) + "\n")
    args.readme.write_text(update_readme(args.readme.read_text(), result))
    print(
        f"ccm-eval: compressed scores {result.ratio:.0%} of uncompressed "
        f"(minimum {result.min_ratio:.0%}): {'passed' if result.passed else 'failed'}. "
        f"Results: {results_file}. {args.readme} is updated; commit it yourself."
    )
    return 0 if result.passed else 1


def _cost(args: argparse.Namespace) -> int:
    """A year of compaction cost for one conversation: the latest run's measured cost per
    compaction round, times the rounds per turn it measured, times the turns in a year.

    Counts what the memory spends (summarizing, embedding); answering and judging are the
    evaluation's own cost and are left out.
    """
    results = sorted(args.results.glob("*.json"))
    if not results:
        print(f"ccm-eval: no results in {args.results}; run `ccm-eval run` first", file=sys.stderr)
        return 1
    latest = json.loads(results[-1].read_text())
    spent = [latest["cost"].get(role) for role in ("summary", "embedding")]
    if None in spent:
        print(f"ccm-eval: {results[-1]} has a cost without a known price", file=sys.stderr)
        return 1
    rounds = sum(len(r) for r in latest["tokens"].values())
    turns = sum(
        len(c.turns) for c in conversations.load(args.conversations) if c.name in latest["tokens"]
    )
    per_round = sum(spent) / rounds
    yearly_rounds = args.turns_per_week * 52 * rounds / turns
    print(
        f"ccm-eval cost: from {results[-1]}: {rounds} compaction rounds over {turns} turns, "
        f"${per_round:.4f} per round (summarizing and embedding). At "
        f"{args.turns_per_week:g} turns per week: {yearly_rounds:.0f} compaction rounds and "
        f"${per_round * yearly_rounds:.2f} per conversation per year."
    )
    return 0


def _sweep(loaded: list[conversations.Conversation], args: argparse.Namespace) -> int:
    from conversation_cluster_memory.evaluation import harness

    now = datetime.now(UTC)
    rows = []
    for setting, values in SWEEP:
        for value in values:
            settings = harness.Settings()
            if setting == "embedding_text":
                settings.embedding_text = str(value)
            elif setting != "defaults":
                settings.params[setting] = value
            print(f"ccm-eval sweep: {setting} = {value}", file=sys.stderr)
            try:
                result, _ = harness.run(loaded, settings, 0.9, "", now)
            except Exception as error:  # one failed cell must not lose the paid ones
                rows.append(SweepRow(setting, str(value), None, f"failed: {error}"))
                continue
            rows.append(SweepRow(setting, str(value), result))
    table = args.results / f"sweep-{now:%Y-%m-%dT%H%M%SZ}.md"
    table.write_text(sweep_table(rows))
    print(sweep_table(rows))
    print(f"ccm-eval sweep: table written to {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
