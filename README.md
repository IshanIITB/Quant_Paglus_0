# Catan AI Agent

## Overview

This project implements an AI agent for a two-player game of Catan using the provided, patched `catanatron` engine. The objective is to select strategic actions that maximize the agent's probability of winning against the baseline agents and other teams' submissions.

The agent operates under imperfect information and a strict CPU-time budget. It receives the current observation and the list of legal actions, evaluates the available choices, and returns the index of its selected action.

## Game Setup

- **Players:** 2
- **Winning condition:** First player to reach 10 victory points
- **Board:** Official spiral layout with randomized board seeds
- **Player trading:** Disabled
- **Bank and port trading:** Enabled
- **Engine:** Organizer-provided, patched `catanatron` build

The agent must comply with the game rules and the observation restrictions defined in the competition problem statement.

## Project Structure

The submission directory should contain the agent implementation and any supporting files it requires.

```text
submission/
├── agent.py
├── README.md
└── <additional modules or data files, if required>
```

Replace the example structure above with the actual files included in the submission.

## Agent Interface

The agent follows the interface required by the evaluation harness:

```python
class Agent:
    def __init__(self):
        """Initialize the agent and load required resources."""

    def decide(self, obs, playable_actions) -> int:
        """Return the index of the selected legal action."""
```

- `obs` contains the information available to the agent at the current decision.
- `playable_actions` contains the legal actions provided by the engine.
- `decide()` returns an integer index into `playable_actions`.

The engine executes the selected action. The agent does not construct actions independently.

## Environment Setup

Use Python 3.11 and the exact dependency versions supplied with the competition kit.

Install the provided dependencies:

```bash
pip install -r requirements.txt
```

Run the environment self-check:

```bash
python -m catan_kit.selfcheck
```

**Important:** Use the organizer-provided engine and its pinned dependencies. Do not install a different `catanatron` version from PyPI.

## Running an Evaluation

To evaluate the agent against the baseline F agent over 1,000 games:

```bash
python -m catan_kit.evaluate \
    --a my_team/agent.py \
    --b examples/f_agent.py \
    --games 1000
```

Replace `my_team/agent.py` with the actual path to the submitted agent.

The evaluation harness reports win rate and its 95% confidence interval, CPU usage, timeouts, faults, and loading time.

## Runtime Constraints

The agent is designed to operate within the competition's execution limits:

- One CPU core per game
- 2 GB memory per agent
- A cumulative CPU allowance starting at 300 ms
- An additional 12.5 ms per decision, up to 1.5 seconds per game
- A maximum of 3 seconds for an individual decision

If the agent exhausts its CPU allowance, the remaining moves are selected randomly by the harness. Illegal action indices, exceptions, and crashes must therefore be avoided.

## Submission Requirements

The submission must satisfy the competition rules:

- Use Python; compiled extensions and executables are not permitted.
- Include `agent.py` and all files required by the agent.
- Keep the total submission size within 100 MB.
- Do not use network access or external services during gameplay.
- Do not launch external programs or worker processes.
- Read only permitted files and write temporary data only to the designated temporary directory.
- Use only the observations and legal actions supplied by the engine.

## Evaluation and Experimentation

Agent performance should be measured using repeated games against the provided baselines. Experiments should use reproducible seeds, compare changes against previously measured versions, and track both win rate and computational cost.

Results should be recorded for each experiment, including:

- Win rate against each baseline
- Confidence intervals
- Mean CPU time per game
- Games that exceed the CPU allowance
- Faults, illegal actions, and timeouts

A change should be considered an improvement only when the evaluation results provide sufficient evidence of better performance without unacceptable runtime costs.

## Acknowledgements

This project uses the competition-provided `catanatron` engine, evaluation harness, and baseline agents. Any additional public libraries, papers, or code used in developing the agent should be acknowledged in the accompanying report, as required by the competition rules.
