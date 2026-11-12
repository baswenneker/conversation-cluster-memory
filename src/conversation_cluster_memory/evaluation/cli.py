"""`ccm-eval`: run the evaluation, sweep parameters, or regenerate the conversations.

Runs locally, by hand: it calls OpenAI with OPENAI_API_KEY from the environment or `.env`.
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
    args = parser.parse_args(argv)

    if args.command == "generate":
        conversations.write(args.conversations)
        return 0
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
