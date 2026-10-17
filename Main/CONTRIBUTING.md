# Contributing

## Setup
Use `uv sync --extra dev`. Commit `uv.lock` whenever dependencies change, and
use only uv for environment management.

## Workflow
- Work on a branch and open a pull request. Another person reviews before merge.
- Keep changes small and focused.
- Changes to `docs/interfaces.md`, the action encoder, or the observation
  encoder need review from both team members.

## Standards
- Run `uv run ruff check .` and `uv run pytest` before opening a pull request.
- Add or update tests with every behavior change.
- Don't reimplement Catan rules that Catanatron provides.

## Correctness rules
- Observations must never expose the opponent's hidden information.
- Training and evaluation must share the same rules, trading setup, and turn cap.
- Shaping discount comes from the configured PPO gamma, never a hardcoded value.
- Report win rates with confidence intervals and draw counts. Label screening
  runs as screening, not as full evaluations.

## Experiments
- Every run records its seed, resolved config, and software versions.
- Never commit `runs/`, checkpoints, logs, or `.env`.
- Never put credentials in code, configs, or logs.
