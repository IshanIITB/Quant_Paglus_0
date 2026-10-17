# Our Approach: A Tuned Heuristic Catan Agent with Budgeted Lookahead

**Status:** design and plan. No experiments have been run yet, so nothing here is a measured result. Claims we have not measured are marked **unmeasured**; details not yet checked against the pinned engine are marked **to verify**.

## 1. In one paragraph

We start from **F**, the organizers' working baseline: catanatron's `ValueFunctionPlayer`, which scores each legal move by applying it to a sampled world and evaluating the resulting position with a weighted sum of 13 features (one move ahead). We keep F's evaluation and add three upgrades: **(1)** offline evolution-strategy tuning of F's weights, **(2)** a same-turn lookahead over the best few candidates, and **(3)** a CPU budget manager that reads `obs.cpu_left` and degrades gracefully. At play time the agent is pure-Python and value-based. No neural network, no reinforcement learning and no language model runs inside `decide`.

## 2. Why this approach

| Reason | Detail |
|---|---|
| Fits the clock | About 1 s of CPU per game on average, roughly 6x what F uses (about 158 ms per game, organizers' figure). Search must fit inside this. |
| Buildable in one day | F is given as working code and encodes competent Catan judgment we could not rebuild from scratch in time. |
| Explainable | Every component can be ablated and defended in the report and Q&A. |
| Honest about limits | We do not claim it beats F, O or any RL agent. The experiment plan is how we find out. |

The kit's own depth-3 AlphaBeta agent ran out of its allowance in 1,000 of 1,000 games and then played random moves. The main engineering problem is fitting search inside the clock, which is why the lookahead is deliberately shallow and budgeted.

## 3. Constraints that shape the design

- **Game:** 2 players, first to 10 VP, bank and port trades on, player trades off, 1,000-turn cap (no winner counts as a loss for both).
- **Hidden information:** only the opponent's unplayed development cards (and the deck order). `obs.sample_world(rng)` returns one complete world consistent with everything visible.
- **Clock:** CPU per game is `min(300 ms + 12.5 ms x k, 1.5 s)` after k decisions. Going over means the harness plays random legal moves for us. A single decision may not exceed 3 s wall time.
- **Machine:** one core, 2 GB, Python 3.11. Local timings may be about 2x optimistic versus the scoring machine.
- **Faults:** an exception or illegal index becomes a random move and counts as a fault. More than 2% faulty decisions disqualify the submission.
- **Interface:** `decide(obs, playable_actions)` returns an **index** into the legal list. Forced moves are played by the harness and cost no clock.

## 4. How one decision works

1. **Budget check.** Read `obs.cpu_left`. If the allowance is nearly gone, take the emergency path (step 8) instead of searching.
2. **Sample a world.** Draw one consistent world with `obs.sample_world(rng)`.
3. **One-ply scoring.** Score every legal move one step ahead with F's value function, using tuned weights from `weights.json`. If the file is missing or bad, fall back to F's default weights; loading never raises.
4. **Sure-win check.** If a deterministic move reaches 10 VP in the sampled world, play it. A random-outcome move (development-card buy, robber steal) is never treated as a guaranteed win.
5. **Same-turn lookahead.** For the top 3 candidates (top 6 for the opening settlement), find the best follow-up still inside our own turn. We do not search into the opponent's turn.
6. **Random-outcome averaging.** For a development-card buy, average the value over a few fresh sampled worlds. For a robber steal, re-execute the steal several times in the same world and average.
7. **Selection.** Pick the candidate with the best *finished* continuation. The best one-ply move is always kept as a fallback.
8. **Emergency path.** When the reserve is nearly spent: play a city, settlement or development-card buy if legal, otherwise end the turn. Never answer a trade loop with another bank trade. The budget manager spends more on the opening and skips lookahead when the budget is small, so this path should stay rare.
9. **Exception safety.** Any exception falls back to a legal index. This still counts as a fault, so the goal is zero fallbacks.

## 5. The three upgrades

### 5.1 Tuned weights (offline only)

- A `(1 + lambda)` evolution strategy over log-multipliers of F's non-zero weights. Setting the multipliers to zero reproduces F exactly. Features F weights at zero stay at zero.
- **Objective:** the mean of the worst half of an opponent pool's win rates, the same shape as the robustness part of the hackathon score.
- **Pool:** F, two randomly perturbed copies of F, and any extra opponents we can add.
- **Acceptance:** a child replaces its parent only if it wins a confirmation run on fresh, non-overlapping seeds by at least z standard errors. z is fixed and logged before tuning starts. Accepted weights are *candidates* and must be validated on a separate seed set before submission.
- **Bookkeeping:** every run is logged (weights, seeds, win rates, CPU, faults). Failed variants are kept.

### 5.2 Same-turn lookahead

Shallow on purpose. Dice and opponent replies multiply cost, so we refine only the top few candidates within our own turn and drop any candidate whose lookahead does not finish in budget.

### 5.3 CPU budget manager

Reserves allowance, spends extra on the opening, skips lookahead when `cpu_left` is low, and hands off to the emergency path as a last resort. Its constants (reserve fraction, opening spend, skip thresholds) are engineering guesses: **unmeasured**.

## 6. Alternatives we considered

| Alternative | Why not (today) |
|---|---|
| Write a heuristic from scratch | One day; F already encodes competent judgment. |
| Hand-tune weights | Slow, unprincipled, hard to ablate honestly. |
| Search into the opponent's turn | Chance nodes and replies blow the clock (kit AlphaBeta data above). |
| RL policy or learned value network at play time | Unmeasured for us; training in one day is high-risk. Published RL successes (e.g. Gendre and Kaneko 2020, reported as beating the jsettler heuristic bot, **to verify**) used custom networks and long training. |
| Distilled value network from our search | Deferred, not rejected: we lack time to generate good training data. |
| LLM inside `decide` | Latency far above the per-decision budget, non-deterministic, and no network at evaluation. |

## 7. Experiment plan

Paired comparisons on identical seed sets across versions; at least 400 games per matchup in development and 1,000 before submitting; evaluation seeds strictly separate from tuning seeds; seats alternate; failed variants are reported. O is measurable only through the 7 scored submissions, so none is spent on an unvalidated variant.

| Version | What it adds | Win rate vs F (95% CI) | Win rate vs pool | Mean CPU per game | Games over the clock | Faults |
|---|---|---|---|---|---|---|
| v0 | F itself (reference) | | | | | |
| v1 | + sure-win check, search off | | | | | |
| v2 | + same-turn lookahead | | | | | |
| v2.1 | + random-outcome averaging | | | | | |
| v3 | + tuned weights | | | | | |

Cells stay empty until filled from our own runs. Sanity expectations (not results): v1 should nearly always beat `random_agent` and score roughly half against F; a large gap either way is a bug.

## 8. Known limitations

- No cross-turn planning and no opponent modelling.
- The evaluation ceiling is F's 13 features; anything F cannot see, we cannot value.
- Tuning is noisy at a few hundred games per comparison, and the pool is narrow, so tuned weights may overfit to F's style.
- Budget constants and the emergency policy are untested in long games.

## 9. Before any upload

- Byte-compile every file in the zip; kit selfcheck passes (pinned catanatron build, never PyPI's release).
- Verify every catanatron API name we use against the pinned build (**to verify**).
- Confirm the evaluator's seed flag and output format before scripting runs (**to verify**).
- Long-game test of the emergency path with the allowance forced near zero.
- Zip holds `agent.py` at the top level, `weights.json`, and nothing else; confirm the size limit from the full problem statement (**to verify**).

## 10. Fair play

The agent uses no language model and no network at play time. We build on catanatron (Bryan Collazo, GPL-3.0-or-later) and the organizers' starter kit, and keep our submission GPL-compatible. Every number in the report will come from our own logged runs. We use only the given interface, never the evaluation seeds, the dice key or any hidden information.
