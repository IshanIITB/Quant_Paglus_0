# Heuristic Catan Agent — Team Working Document

Our agent: a tuned F-style heuristic with budgeted same-turn lookahead. This document is raw
material for the 4-page report and the presentation Q&A. Kit facts come from the team's
OVERVIEW.md (itself derived from the organizers' starter kit). Claims we have not measured are
marked **unmeasured**; API details we have not checked against the pinned engine are marked
**to verify**. Status: design and plan; no experimental results yet.

## 1. Summary

We start from F — catanatron's `ValueFunctionPlayer`, a weighted sum of 13 features with
one-move lookahead, given to us as working code — and add three upgrades: an evolution-strategy
tuning of the 13 weights (offline, never in the submission), a same-turn lookahead over the top
candidates, and a CPU budget manager that reads `obs.cpu_left` and degrades search gracefully.
The agent is a pure-Python value-based player: sample one consistent world with
`obs.sample_world`, score every legal move one step ahead, optionally refine the best few
candidates with a same-turn continuation, and return an index. No neural network and no
reinforcement learning runs at play time. We have **not** measured that this beats F, O, RL, or
any other baseline; the experiment plan below is how we will find out.

## 2. Task and constraints (kit facts)

### 2.1 Game and rules

| Item | Value |
|---|---|
| Players / win | 2 players; first to 10 VP wins |
| Board | official spiral layout, randomized board seeds |
| Player-to-player trades | off |
| Bank / port trades | on (4:1, 3:1, 2:1) |
| Engine | catanatron, pinned to one git commit (`d3f4ad05…`, kept truncated); PyPI's catanatron has a different action space and must not be installed |
| Turn cap | 1,000 turns; a game with no winner is a loss for both sides |
| Hidden information | only the opponent's unplayed development cards (and thus hidden VP cards); the resource hand of the opponent is derivable from public information |

### 2.2 Interface

The submission is one directory with `agent.py` at the top level plus files it loads (weights,
tables); total zip size at most 100 MB (team brief; not stated in OVERVIEW.md — **to verify**).
The contract:

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Up to 60 s, not clocked."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of the chosen action in playable_actions."""
```

- We return an index; we never build `Action` objects.
- Forced moves (exactly one legal action) are played by the harness, never shown to the agent
  (47% of decisions in F-vs-F play), and do not cost clock time.
- One process plays many games in a row against the same opponent; adapting to the opponent
  within a process is allowed, nothing carries over between processes.
- A raised exception or an illegal index makes the harness play a random legal move and counts
  as a fault; **more than 2% faulty decisions disqualify the submission**.
- `obs.sample_world(rng)` returns a complete `Game` consistent with everything we can see;
  `obs.cpu_left` reports the seconds of clock allowance remaining.

### 2.3 The clock and the machine

| Limit | Value | Consequence |
|---|---|---|
| CPU per game | `min(300 ms + 12.5 ms x k, 1.5 s)` after k decisions; ~1 s/game average | roughly 6x F's compute; over the clock, the harness plays random legal moves for us |
| Per decision | 3 s wall cap; a silent agent is killed and restarted | no single call may block |
| Load | 60 s for import + `__init__`, not clocked | load and warm everything in `__init__` |
| Memory / CPU | 2 GB, exactly one core | no multiprocessing inside the agent; threads buy nothing |
| Scoring machine | AWS c7a.48xlarge, Python 3.11 | local timings can be ~2x optimistic; our machine is not the reference |

Allowed imports at evaluation: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.8.0 (CPU),
networkx 3.6.1, psutil 7.2.2, and the pinned catanatron. Nothing else is installed there.

Submission limits: 7 scored submissions, at most 10 uploads in total (failures included), each
upload must first pass two error-free test games. Scoring: win rate vs F and vs O over 1,000
games each with 95% intervals; robustness (mean win rate vs the worst half of group opponents)
plus strength (`0.5 * vsF + 0.5 * vsO`) plus report and presentation.

## 3. Our approach and decision procedure

Baseline: F scores each candidate move by applying it to a sampled world and evaluating the
resulting position with a weighted sum of 13 features, one move ahead, no deeper. Kit numbers
put F at about 158 ms of CPU per game (organizers' measurement). Our three upgrades: tuned
weights, a same-turn lookahead, a CPU budget manager. Decision procedure per call:

1. Budget check first. Read `obs.cpu_left`. If the allowance is nearly gone, take the emergency
   path (step 8) instead of searching.
2. Sample one world with `obs.sample_world(rng)`. Only the opponent's unplayed development
   cards and the remaining deck are redrawn; the world agrees with everything we can see.
3. One-ply scoring. Score every legal move one step ahead with F's value function, using
   weights from `weights.json` (missing or bad file means F's default weights; the loader never
   raises).
4. Sure-win check. If a deterministic move reaches 10 VP in the sampled world, play it
   immediately. Never treat a random-outcome move (development-card buy, robber steal with a
   victim) as a guaranteed win, because the sampled outcome is not guaranteed.
5. Same-turn lookahead on the top 3 candidates (top 6 for the opening settlement): the best
   follow-up still inside our own turn. We do not search into the opponent's turn — dice and
   replies cost too much CPU (reasoning in section 4).
6. Random-outcome averaging. For a development-card buy, average the value over a few samples,
   each with a fresh sampled world (the deck is hidden). For a robber steal, re-execute the
   steal several times in the same world and average, since the steal is drawn by the engine
   inside the copy.
7. Selection. Choose the candidate with the best finished continuation, comparing only
   candidates whose lookahead finished; the best one-ply move is always kept as a fallback.
8. Emergency path (the CPU budget manager's last resort). When the reserve is nearly spent:
   play a city, a settlement, or a development-card buy if legal, otherwise end the turn; never
   answer a trade loop with another bank trade. The budget manager spends more on the opening,
   keeps a reserve, and skips the lookahead when the budget is small, so this path stays rare.
9. Exception safety. Any exception anywhere falls back to a legal action index. A fallback
   still counts as a fault on the harness side, so the goal is zero fallbacks, not tolerated
   fallbacks.

Text flow of one decision:

```
decide(obs, playable_actions)
  |
  |-- cpu_left nearly gone? ---------------------> EMERGENCY PATH (step 8 below)
  |                                            city > settlement > dev buy > end turn
  |-- sample_world(rng)            (one consistent world)
  |-- 1-ply score all actions      (F value fn, weights.json)
  |-- deterministic action wins now (10 VP)?  --> return its index
  |-- same-turn lookahead, top-3   (top-6 for opening settlement)
  |     |-- random-outcome moves: average over k samples
  |     '-- unfinished candidates are dropped
  |-- return best finished continuation (fallback: best 1-ply move)
  '-- any exception --> return a legal index  (counts as a fault; keep at zero)

EMERGENCY PATH: play a city, settlement, or dev-card buy if legal, else end the turn;
never answer a trade loop with another bank trade.
```

## 4. Design decisions and rejected alternatives

| Decision | Why | Alternative | Why rejected |
|---|---|---|---|
| Start from F's value function | F is provided, working, and cheap (~158 ms/game); we inherit a known baseline | Write our own heuristic from scratch | One day; F encodes competent Catan judgment we could not rebuild in time |
| Tune weights by evolution strategy | No gradients needed; win rate is the real objective; small parameter count | Hand-tuning | Slow, unprincipled, hard to ablate honestly |
| Same-turn lookahead only | Our average allowance is ~1 s/game; F uses ~158 ms, so we have ~6x F's compute to spend | Search into the opponent's turn (expectimax over dice and replies) | Chance nodes and opponent replies multiply the cost; the kit's own depth-3 AlphaBeta ran out of its allowance in 1,000 of 1,000 games and then played random moves. Fitting search inside the clock is the main engineering problem (kit data) |
| Heuristic at play time | Fits the clock trivially; every component is ablatable and explainable | RL policy or learned value network at play time | Unmeasured for us; training inside one day is high-risk; published RL successes (e.g. Gendre and Kaneko 2020, reported as an RL agent beating the jsettler heuristic bot — **to verify**: we have not checked the reference ourselves) used custom networks and long training efforts we cannot replicate in a day |
| No language model in the agent | Deterministic and fast; the evaluation environment offers no network, and the per-decision budget is 12.5 ms | LLM player inside `decide` | Rejected: latency far above the per-decision budget, non-deterministic, needs a network the evaluation machine does not provide, hard to credit or debug |
| F's 13 hand-set features, tuned but not replaced | Fully explainable; no training data needed; cheap enough to search on | Learned value network distilled from our search | Deferred, not rejected outright: we lack time to generate good training data; unmeasured idea, not a plan |
| Emergency path is a fixed cheap policy | Guarantee under any clock pressure | Spend down to zero and hope | An over-clock game hands our moves to the harness's random player |

Fairness note: published work reports reinforcement-learning agents beating classic Catan
heuristics (Gendre and Kaneko 2020, against jsettler — **to verify**: we have not yet checked
the exact reference or what it reports), using custom networks and long training.
We cannot replicate that in a day, and we do not claim our choice is stronger — only that it is
the one we can build, measure, and defend today. **Unmeasured by us.**

## 5. Weight tuning method and its limits

Offline only; tuned weights ship as `weights.json`, the tuner never runs at play time.

- Parameterization: a `(1 + lambda)` evolution strategy over log-multipliers of F's non-zero
  weights. A child is `w_i = w_i^F * exp(m_i)` with Gaussian `m_i`; theta = 0 reproduces F
  exactly. F's default zero weights are frozen at zero (a zero feature cannot be revived).
- Objective: the mean of the worst half of an opponent pool's win rates — the same shape as the
  robustness component of the hackathon score, so tuning optimizes the thing that is marked.
- Pool: F itself, two randomly perturbed copies of F (cheap, diverse, reproducible), and any
  extra opponents we manage to add.
- Acceptance: a child must beat the parent on a confirmation run on fresh, non-overlapping
  seeds, by a margin of at least z standard errors of the difference; only then does it replace
  the parent. z is fixed before tuning starts and logged. Accepted weights are **candidates**:
  they must then be validated on a separate seed set before any submission.
- Bookkeeping: every run is logged to a CSV (weights, seeds, win rates, CPU, faults); failed
  variants are kept, not deleted.

Limits, stated plainly: the objective is noisy (at a few hundred games the 95% interval on a
win rate is several percentage points wide); the pool is narrow, so tuned weights can overfit
to F's style and still lose to other agents; tuning cannot add features F does not have; and
the perturbed-F opponents are weak proxies for the variety of the group stage. Unmeasured.

## 6. Experiment plan

Protocol: paired comparisons on identical seed sets across versions; at least 400 games per
matchup during development and 1,000 before submitting; evaluation seeds strictly separate from
tuning seeds; seats alternate (the harness does this per game index); failed variants are kept
and reported. O's strength is only measurable through the 7 scored submissions, so we treat
leaderboard rows as precious and never burn one on an unvalidated variant.

Cells are intentionally empty; they will be filled only from our own runs.

| Version | What it adds | Win rate vs F (95% CI) | Win rate vs opponent pool | Mean CPU per game | Games over the clock | Faults |
|---|---|---|---|---|---|---|
| v0 | F itself (reference) | | | | | |
| v1 | + sure-win check, search off | | | | | |
| v2 | + same-turn lookahead | | | | | |
| v2.1 | + random-outcome averaging | | | | | |
| v3 | + tuned weights | | | | | |

Expectations to check against (not results): v1 should nearly always beat `random_agent` and
score roughly half against F with search off — a large gap either way is a bug. v2 and v2.1
should beat v0 if the lookahead is worth its CPU. Every claim above the baseline is unmeasured
until this table has numbers.

## 7. Validation checklist before any upload

- [ ] Byte-compile (`py_compile`) every file that goes into the zip.
- [ ] Kit selfcheck passes (Python 3.11, networkx 3.6.1, determinism fingerprint; the catanatron
      version it expects, 3.3.0, is the version of the **pinned build** installed from the kit's
      requirements file — never PyPI's release, whose action space differs).
- [ ] Search OFF: near-total win rate vs `random_agent` (expected near-total, kit selfcheck
      threshold is >= 90% for F) and roughly 50% vs F; a large gap from 50% is a bug.
- [ ] Fallback and exception counters surfaced in a developer mode (local runs only), target zero.
- [ ] Verify against the pinned build every catanatron API name we use:
      `action.action_type`, `game.winning_color()`, `game.playable_actions`,
      `game.state.current_color()`, `state.is_initial_build_phase` — **to verify**.
- [ ] Confirm the real seed flag of the evaluator from its `--help` output before scripting
      identical seeds per generation — **to verify**.
- [ ] Confirm the win-rate parsing (regex) against real evaluator output before trusting any CSV —
      **to verify**.
- [ ] Long-game test of the emergency path (force the allowance nearly to zero; confirm no
      timeout, no exception, no trade loop).
- [ ] Zip contents: `agent.py` at the top level, `weights.json`, and only the files the agent
      loads; nothing else; under 100 MB; case-sensitive names as on Linux.

## 8. Known limitations

- No cross-turn planning: we never reason about the opponent's next turn or our following turns.
- No opponent modelling: the pool has no adaptive or learned opponents; tuning sees F-style play only.
- The evaluation ceiling is F's 13 features; anything F cannot see, we cannot value.
- Tuning noise: with hundreds of games per comparison, small true differences are invisible; the
  z-margin and confirmation runs reduce, not remove, the risk of accepting noise.
- Budget constants (reserve fraction, opening spend, skip thresholds) are engineering guesses —
  **unmeasured** until the CPU columns of the results table exist.
- The emergency policy is heuristic (city > settlement > dev buy > end turn); it has not been
  stress-tested in long games yet.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Lookahead blows the clock in long games | Budget manager with a reserve; skip lookahead when `cpu_left` is low; emergency path as the last resort |
| Tuning overfits to F (narrow pool) | Worst-half objective, perturbed-F opponents, confirmation runs, separate validation seed set, keep failed variants |
| Exceptions or illegal indices accumulate | Global fallback to a legal index; faults counted and surfaced in developer mode; >2% disqualifies, so we monitor fault counts in every run |
| Masking or index-mapping bug returns a wrong index | Every returned index is checked against `len(playable_actions)` internally; a bad path returns index 0 of the legal list |
| 60 s load limit exceeded by heavy init | Keep `__init__` to file reads and tiny tables; no network, no model loading |
| Leaderboard overfitting | Compare versions on fixed seeds, evaluate on disjoint seed sets, remember the private set decides the top 30 |
| Local vs evaluation machine timing drift (~2x reported) | Track the "games over the clock" column in every run; budget with margin |

## 10. Fair play and credits

- Public material we use and will name in the report: catanatron by Bryan Collazo
  (GPL-3.0-or-later), which this kit builds on and from which F's value function and default
  weights come; the organizers' starter kit and its `f_agent` example (GPL-3.0-or-later); the
  papers we cite — currently Gendre and Kaneko 2020 (**to verify**: exact reference and what it
  actually reports), to be listed with full references in the report.
- The agent uses no language model and no network at play time, and calls no external service;
  we use language models only to help write code and analyze results, outside the agent.
- Every number in the report will come from a run we made with this kit; we expect organizers
  may rerun our agent and experiments.
- Licensing: our submission is built on GPL-3.0-or-later material and we keep our submission
  GPL-compatible. We state no legal conclusions beyond what the kit's rules require.
- We do not use the evaluation seeds, the dice key, or any information the kit hides; we play
  only through the given interface.

## 11. Report and presentation notes

For the 4-page report: method first (baseline F, the three upgrades, the decision flow), then
the ablation table with real numbers, then what failed and why. Keep every number traceable to
a logged run; report confidence intervals and game counts; name the seeds policy.

Likely judge questions to prepare answers for: Why same-turn lookahead only, and what would
cross-turn search cost? Why is the tuning objective the worst half of the pool? What did each
ablation (v1, v2, v2.1, v3) show, and what did you drop? What was your biggest measured
surprise? Why is the emergency path safe near the clock ceiling? Every team member must be able
to explain every design choice, including the ones that failed.

## 12. Conflicts to check, and Missing information

Conflicts to check (our text vs OVERVIEW.md; OVERVIEW.md wins where they differ):

- OVERVIEW.md says `sample_world` redraws both the opponent's unplayed development cards **and
  the development deck order**; our short summary said only the dev cards are hidden. See the
  hidden-information row in section 2.1 and the observation contract in section 2.2 — both
  follow OVERVIEW.md.
- OVERVIEW.md's "Next steps" section still describes the earlier reinforcement-learning plan and
  a pre-install machine snapshot ("Python 3.11 not installed yet"); this document supersedes it
  for the agent design, not for environment setup.
- OVERVIEW.md does not state the 100 MB zip limit or the upload script name; the team brief
  supplies 100 MB. Verify both against the full problem statement before submitting.

Missing information:

- The exact names and default values of the 13 features behind F's value function: not in
  OVERVIEW.md; read from the pinned build (**to verify**).
- The evaluator's exact command-line flags and default seed: not in OVERVIEW.md; check
  `--help` on the real script (**to verify**).
- The full pinned catanatron commit hash: deliberately kept truncated in this document.
- Tuner constants we still owe: the z margin, population size, mutation scale, and the number
  of samples k for random-outcome averaging; fix and log them before the first tuning run.
- Whether a verbatim copy of F is an acceptable submission baseline: reported as legal by the
  organizers' materials, but re-confirm from the full problem statement (**to verify**).
- Anything about O beyond "a trained bot that beats F by some margin": secret by design; our
  only measurement channel is the 7 scored submissions.

## Revision log

2026-10-17 — review fixes applied:

1. Results table (section 6): separator row had 6 columns against a 7-column header; now 7.
2. Emergency path: now a numbered step (step 8); the step 1 pointer and the flow diagram
   reference corrected from "step 7" (which was Selection).
3. Section 12 cross-reference: hidden-information row is in section 2.1, observation contract in
   section 2.2; both now cited.
4. The 100 MB zip limit in section 2.2 now carries "**to verify**" (team brief, not OVERVIEW.md).
5. Section 4 table: the LLM row and the learned-value-network row rewritten so every row reads
   Decision, Why, Alternative, Why rejected (no placeholder dashes).
6. Gendre and Kaneko 2020 marked "**to verify**" in the section 4 table, the fairness note, and
   section 10; stated nowhere as established fact.
7. Section 7 selfcheck line: catanatron 3.3.0 clarified as the pinned build's version, never
   PyPI's release.
