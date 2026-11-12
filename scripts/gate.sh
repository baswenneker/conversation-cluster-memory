#!/usr/bin/env bash
# Factory gate: lint, format, typecheck and tests, in parallel.
# FACTORY_GATE_SCOPE=changed skips everything when no Python-relevant or motion graph file changed;
# scope "all" (or no variables) runs everything.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "${FACTORY_GATE_SCOPE:-all}" == "changed" && -s "${FACTORY_CHANGED_FILES:-}" ]]; then
  if ! grep -qE '\.py$|^docs/motion-graph/|^pyproject\.toml$|^uv\.lock$|^\.python-version$|^scripts/gate\.sh$' \
      "$FACTORY_CHANGED_FILES"; then
    echo "gate: no Python files changed, skipping"
    exit 0
  fi
fi

uv sync --locked --quiet || exit 1
# The motion graph test drives a headless browser; a no-op once it is installed.
uv run --no-sync playwright install chromium-headless-shell >/dev/null || exit 1

declare -A jobs
uv run --no-sync ruff check . & jobs[ruff-check]=$!
uv run --no-sync ruff format --check . & jobs[ruff-format]=$!
uv run --no-sync ty check & jobs[ty]=$!
uv run --no-sync pytest -q & jobs[pytest]=$!

failed=()
for name in "${!jobs[@]}"; do
  wait "${jobs[$name]}" || failed+=("$name")
done

if (( ${#failed[@]} )); then
  echo "gate: failed: ${failed[*]}"
  exit 1
fi
echo "gate: all checks passed"
