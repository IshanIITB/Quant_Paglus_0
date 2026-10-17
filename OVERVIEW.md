# Catan Hackathon — Overview & Notes

Summary of the `catan-hackathon-agent` repo (https://github.com/lyric-hyde/catan-hackathon-agent),
which is the starter kit for a one-day Catan agent hackathon (17 Oct 2026, 10:00–19:30 IST).

---

## 1. The competition

- Build an agent that plays **2-player Catan** (10 VP to win, official spiral board,
  randomized seeds). Player-to-player trading is **off**; bank/port trading works normally.
- Opponents: the organizer bots **F** (Catanatron's `ValueFunctionPlayer`, one-move lookahead
  over 13 weighted features) and **O** (their secret trained bot), plus other teams in a group stage.
- Engine: `catanatron` pinned to an exact git commit (`d3f4ad05…`). **Never** `pip install catanatron`
  from PyPI — the action space differs (`DISCARD_RESOURCE` vs `DISCARD`, 2-tuple robber moves).
- Engine wins all rule disputes. A game hitting 1,000 turns is a **loss for both sides**.

## 2. The agent contract

A folder containing `agent.py` that defines:

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Called once; up to 60 s, not clocked."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of the chosen action in playable_actions."""
```

- You return an integer index; you never build `Action` objects.
- Forced moves (exactly one legal action) are played by the harness and never shown to the agent
  (47% of decisions in F-vs-F play).
- One process plays many games in a row; the agent **may adapt to its opponent within a process**,
  but nothing carries over between processes.
- Load everything in `__init__` — the first `decide` is charged to the first game.

## 3. What the agent sees (`catan_kit/obs.py`)

| Sees | Does not see |
|---|---|
| Board, own hand (exact), own dev cards, own public & actual VPs | Opponent's unplayed dev cards (by type) |
| Bank counts, turn number, all played dev cards | Opponent's hidden VPs (VP cards in hand) |
| Opponent's exact **resource** hand (derivable from public info anyway) | Development deck order |
| Opponent's dev-card **count** | |

- The dev deck is the only real hidden information in 2-player Catan.
- `obs.sample_world(rng)` → a complete `Game` consistent with everything visible
  (opponent's unplayed cards + remaining deck redrawn). Sample several worlds to search
  under uncertainty (PIMC-style).
- `obs.cpu_left` = seconds of allowance remaining (use it to budget search).

## 4. Hard limits (enforced by the harness, read from outside the process)

- **CPU chess clock**: `min(300 ms + 12.5 ms × k, 1,500 ms)` after k decisions → ~1 s/game
  average (~6× F's compute). Over the clock → random moves for the rest of that game.
- **2 GB RAM**, **one core** (no multiprocessing inside the agent; threads don't help).
- **3 s** max per decision (silent agent is killed + restarted, random moves finish the game).
- **60 s** to import + `__init__` (not clocked).
- Errors / illegal indices → random move; **>2% of decisions = disqualification**.
- Eval machine: AWS c7a.48xlarge, Python 3.11, 2 GB + 1 core per agent. Leaderboard numbers
  are measured there — local laptop timings can be ~2× optimistic.

## 5. Allowed imports at evaluation

Only: `numpy==2.4.6`, `scipy==1.17.1`, `scikit-learn==1.9.1`, `torch==2.8.0` (CPU),
`networkx==3.6.1`, `psutil==7.2.2`, and the pinned catanatron.
**No stable-baselines3, no gymnasium, no huggingface_hub, no sb3-contrib on the eval machine.**

Implication for the RL plan: train offline with SB3 locally if desired, but the submission
must be pure PyTorch — export policy weights to a `.pt`/state_dict and reimplement the forward
pass + action masking inside `decide()`. A learned **value function + shallow expectimax search**
fits the ~1 s/game budget better than a deep policy net (cf. TD-Gammon pattern in the reading list).

## 6. catan_kit modules

| Module | Role |
|---|---|
| `__init__.py` | applies the determinism patch on import — import before creating any Game |
| `_patch.py` | engine determinism fixes: sorted maritime-trade offers, `PYTHONHASHSEED=0` re-exec, feature-ordering random-state guard |
| `obs.py` | the masked observation + `masked_copy` / `sample_world` for imperfect-info search |
| `match.py` | the harness: one OS process per agent, chess clock, masked copies, seat alternation (even index = A first), dice decoupled from board seed via HMAC |
| `host.py` | the agent-side protocol (JSON replies, index-only answers, per-game reseeding) |
| `evaluate.py` | **the official scoring script** (`python -m catan_kit.evaluate --a … --b … --games N`) |
| `sandbox.py` | optional cgroup v2 isolation, off locally (`CATAN_SANDBOX=1` on their machine) |
| `selfcheck.py` | install check: Python 3.11, catanatron 3.3.0, networkx 3.6.1, determinism fingerprint (seed 4256 → `('Color.RED', 111)`), f_agent ≥90% vs random |

## 7. Scoring & schedule

- 7 scored submissions (max 10 uploads total, failures count); each must pass 2 error-free test games.
- Scored on 1,000 games vs F and vs O, seats alternated, 95% CI shown.
- Funnel: uploads close 17:00 IST → top 30 by private-set mean → groups of 10, new boards,
  round-robin 1,000 games → top 2 per group present.
- Marks: robustness (mean win rate vs worst half of group opponents) + strength
  (`0.5·vsF + 0.5·vsO`) ≈ report (≤4 pages) + presentation/questions.

## 8. Workspace status (this machine)

- `QP_0/catan_kit/` — the 8 kit modules extracted from the repo.
- `/tmp/opencode/catan-hackathon-agent/` — full repo clone (examples: `f_agent.py`, `random_agent.py`).
- System Python is 3.14.4; **Python 3.11 is not installed yet**, no venv, no catanatron.

## 9. Next steps

1. Install Python 3.11 (e.g. `uv python install 3.11`), create venv in `QP_0`.
2. Install `requirements.txt` (+ CPU torch on Linux) — pinned versions only.
3. Run `catan_kit.selfcheck` — fix everything before writing agent code.
4. Baseline: run `f_agent.py` vs `random_agent.py` through `catan_kit.evaluate`.
5. Build the agent: index-mapping layer (legal-action list ⇄ fixed action space),
   observation encoder from `Obs`, torch inference inside the clock budget.
6. Evaluate with `evaluate.py` against `f_agent.py` before every submission.
