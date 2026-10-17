# Catan RL: Two-Player Self-Play Reinforcement Learning Agent

A modular reinforcement learning project that trains an agent to play two-player Catan using Catanatron, MaskablePPO, Stable-Baselines3, and Hugging Face. The agent learns through simulated games, improves by playing historical versions of itself, and is evaluated against independent baseline opponents.

This document consolidates the design documents and the package structure, with the review fixes applied: the simulator audit now comes first, tests were added for configuration, seeding, and the baseline bots, and one dependency manager (`uv`) is chosen from day one.

---

## 1. Project overview

**Game rules (from the briefing deck).** Two players compete on a hex-tile island. Five resources (wood, brick, sheep, wheat, ore) are produced when dice numbers are rolled. Players build roads, settlements (+1 VP), and cities (+2 VP total) and buy development cards (Knight, Road Building, Year of Plenty, hidden Victory Point). A rolled 7 activates the robber. Trading is with the bank at 4:1, or 3:1 and 2:1 at ports. Largest Army (3 Knights) and Longest Road (5 connected segments) are worth +2 VP each. **First to 10 VP wins.**

**Objectives**
1. Train a MaskablePPO policy from simulated game experience.
2. Support self-play against historical checkpoints and baseline bots.
3. Evaluate with win rates, Victory Point margins, confidence intervals, and Elo.
4. Publish validated checkpoints to the Hugging Face Hub.
5. Optionally generate natural-language move explanations with a small instruct LLM.

**Key principle.** Catanatron is the source of truth for the game rules. This project does not reimplement mechanics the simulator already provides.

---

## 2. Technology stack

| Area | Choice |
|---|---|
| Game simulator | Catanatron, configured for 2 players |
| RL algorithm | MaskablePPO from `sb3-contrib` |
| RL framework | Stable-Baselines3 |
| Environment interface | Gymnasium-compatible wrappers (check whether the pinned Catanatron targets Gym or Gymnasium) |
| Deep learning | PyTorch |
| Model distribution | Hugging Face Hub (`huggingface_sb3` if compatible, otherwise `huggingface_hub` directly) |
| Dependency management | `uv` with a committed `uv.lock` |
| Experiment tracking | TensorBoard plus structured CSV/JSON logs |
| Optional explanations | Small Qwen or Llama instruct model |

**What MaskablePPO is.** It is PPO with invalid-action masking, provided as code in `sb3-contrib`. It is an algorithm, not a model hosted on Hugging Face. Each turn the environment supplies a legal-action mask, illegal actions get zero probability, and the agent spends its learning on strategy instead of on avoiding illegal moves. The trained `.zip` checkpoint it produces is the model you upload to the Hub.

**The gameplay model is trained from scratch.** No pretrained LLM selects moves. I'm not aware of a verified pretrained Catan checkpoint on the Hub, so don't plan around finding one.

**Compatibility must be verified early.** Pin and test Catanatron, Gymnasium, Stable-Baselines3, sb3-contrib, PyTorch, and the Hub tooling together before finalizing the pipeline. Whether `huggingface_sb3` loads MaskablePPO checkpoints correctly is unconfirmed. If it doesn't, build a custom packaging and loading workflow.

---

## 3. Gym vs. AlphaBeta: how they relate

They are not alternatives.
- **Gymnasium** is an interface for how a learning algorithm talks to a game. OpenAI Gym is no longer maintained, and Gymnasium is its successor.
- **AlphaBeta** is a search-based playing strategy that needs no training.

AlphaBeta's roles here, in order of priority:
1. **Benchmark** (used from the start).
2. **Training opponent** in the pool, if its cost is acceptable.
3. **Teacher** for imitation pretraining, only if the baseline stalls.
4. **Shaping hint**, only if it doesn't leak hidden information.

---

## 4. Package architecture

### 4.1 Core: `catan_rl/`
- **Environment wrapper:** Catanatron to Gymnasium, game resets, transitions, termination vs. truncation.
- **Observation encoder:** numeric features from information the acting player may legitimately know.
- **Action mask:** legal-action detection for MaskablePPO.
- **Action encoder:** a stable mapping between Catanatron actions and the policy's action representation. This is the most important early design decision.
- **Rewards:** terminal rewards plus potential-based shaping using the PPO gamma.
- **Config, seeding, and logging utilities.**

### 4.2 Policy
An MLP with policy and value heads to start. Graph-based board encoders come only after a reliable baseline. The inference module loads a checkpoint and selects legal actions.

### 4.3 Training and self-play
The trainer collects rollouts and runs PPO updates. The self-play module assigns seats and dispatches decisions between the current policy and historical opponents. The opponent pool samples historical checkpoints and baseline bots. A checkpoint manager tracks latest, best, and historical models.

**Self-play integration is a critical task.** Catanatron's Gym environment may assume one learning agent against a configured opponent. Inspect the pinned version before assuming it supports loading multiple policies. If it doesn't, build an adapter that invokes a historical checkpoint as a bot whenever it is that opponent's turn, translating observations, masks, and actions consistently. Before full training, a prototype must complete one whole game between the current policy and a historical checkpoint.

### 4.4 Opponents
Random bot, weighted-random bot, an AlphaBeta adapter, and a policy opponent that loads saved RL checkpoints through the same validated inference interface.

### 4.5 Evaluation
Matchups with alternating first-player seats, win/draw/loss accounting, Victory Point margins, Wilson confidence intervals, Elo, tournaments, and reports.

### 4.6 Hugging Face integration
Upload, download of a specific revision, model card, and compatibility validation. Weights that deserialize are not enough to call a model deployable.

### 4.7 Optional LLM explanations
Converts a chosen move and its context into a structured summary, and a small instruct model writes the explanation. It never chooses actions or affects rewards, and it stays outside the decision path.

---

## 5. Final directory structure

```
catan-rl/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── CONTRIBUTING.md
│
├── catan_rl/
│   ├── __init__.py
│   ├── config.py              # load/merge YAML, validate, experiment IDs, frozen configs
│   ├── seeding.py             # independent random streams
│   ├── logging_utils.py       # structured training/eval logs
│   ├── environment.py         # Catanatron wrapper, Gymnasium interface, termination
│   ├── action_mask.py         # legal-action masks
│   ├── observations.py        # state encoding (single home for features)
│   ├── action_encoder.py      # Catanatron action <-> policy action mapping
│   ├── rewards.py             # terminal + potential-based shaping (gamma from config)
│   ├── constants.py
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── network.py
│   │   └── inference.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── self_play.py
│   │   ├── opponent_pool.py
│   │   ├── callbacks.py
│   │   └── checkpoint_manager.py
│   ├── opponents/
│   │   ├── __init__.py
│   │   ├── random_bot.py
│   │   ├── weighted_random_bot.py
│   │   ├── alphabeta_bot.py
│   │   └── policy_opponent.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── elo.py
│   │   ├── tournament.py
│   │   └── evaluation_report.py
│   ├── hub/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── download.py
│   │   ├── model_card.py
│   │   └── validation.py
│   └── explanations/          # optional
│       ├── __init__.py
│       ├── state_summary.py
│       ├── llm_explainer.py
│       └── prompts.py
│
├── configs/
│   ├── default.yaml
│   ├── training.yaml
│   ├── rewards.yaml
│   ├── evaluation.yaml
│   ├── opponents.yaml
│   └── hub.yaml               # no credentials
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── tournament.py
│   ├── benchmark.py
│   ├── upload_checkpoint.py
│   ├── download_checkpoint.py
│   └── explain_move.py
│
├── tests/
│   ├── test_config.py         # NEW
│   ├── test_seeding.py        # NEW
│   ├── test_environment.py
│   ├── test_action_mask.py
│   ├── test_observations.py
│   ├── test_action_encoder.py
│   ├── test_rewards.py
│   ├── test_determinism.py
│   ├── test_bots.py           # NEW
│   ├── test_training.py
│   ├── test_self_play.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_hub.py
│
├── docs/
│   ├── catanatron_audit.md
│   └── interfaces.md
│
└── runs/                      # generated, not version controlled
    ├── checkpoints/
    ├── logs/
    ├── evaluations/
    └── configs/
```

**Tests added in this revision**
- `test_config.py`: YAML loading and merging, validation errors, experiment ID generation, frozen config saved, and the shaping discount matching the configured PPO gamma.
- `test_seeding.py`: the same master seed gives the same streams, different streams stay independent, and changing one stream doesn't disturb another.
- `test_bots.py`: Random, Weighted Random, and AlphaBeta adapters only ever select legal actions and complete full games without errors.

You don't need to create every file up front. Create modules as you reach them, and don't commit empty placeholders.

---

## 6. How training data is generated

There is no labeled dataset. The agent generates its own experience:

1. **Initialize** a two-player game in Catanatron.
2. **Encode** the acting player's observation from legitimately available information.
3. **Compute** the legal-action mask.
4. **Select and execute** an action with MaskablePPO.
5. **Record** observation, action, mask, reward, next observation, and termination.
6. **Finish** when a player reaches 10 VP or the turn cap is hit.
7. **Update** the policy and value network, then repeat with the newer policy.

Rollout data is an intermediate artifact. Keep the checkpoints, configs, aggregate metrics, and selected game records needed for debugging and reproducibility.

---

## 7. Hidden information and observation safety

The encoder must not expose what the acting player couldn't legitimately know.
- Audit the pinned Catanatron's feature code for how it represents hands, development cards, and public vs. private Victory Points.
- Separate public, own-private, and opponent-hidden information.
- Add an automated test: changing the opponent's hidden resources or unplayed dev cards must not change the acting player's observation.
- Apply identical restrictions in training, evaluation, and inference.

---

## 8. Action representation and trading rules

- Support construction, development-card play, robber placement, discarding, and trading.
- Keep a stable mapping between actions and mask entries even though the legal set changes per state.
- Establish exactly which trades the pinned Catanatron supports (4:1 bank, 3:1 and 2:1 ports, and whether player-to-player trading exists). Don't claim player-to-player trading unless the simulator and wrapper support it.
- Keep the rule configuration identical across training and evaluation.

---

## 9. Reward design

**Terminal:** win +1, loss -1, turn-cap draw 0.

**Shaping (small, potential-based):**

```
shaping = lambda * ( gamma * Phi(s') - Phi(s) )     # lambda about 0.1 to 0.3

Phi(s) = (my_VP - opp_VP) / 10
       + 0.02 * (my_production_pips - opp_production_pips) / 10
       + 0.01 * resource_diversity
```

Rules:
- `gamma` is read from the active training config, never hardcoded, so the policy-preserving property of potential-based shaping holds.
- Keep shaping small relative to the +/-1 terminal reward.
- Don't reward raw resource hoarding.
- Log terminal and shaping rewards separately.
- Compare shaped vs. unshaped runs by evaluated win rate, not by training return.
- Handle terminal vs. truncated states deliberately. A turn-cap draw is recorded as a draw.

---

## 10. Self-play against historical checkpoints

- The pool holds historical checkpoints and baseline bots.
- A configurable mix of recent checkpoints, older ones, and bots gives diversity. Older checkpoints help stop the policy forgetting how to beat earlier strategies.
- The trainer tracks which player is the learner at every decision, so observations, rewards, and outcomes use the right perspective.
- Both first-player seats are used during training.
- Opponents never see information they shouldn't.
- Training records which opponents and seats were used.
- Validate a single complete game before scaling to long runs.

---

## 11. Turn cap and episode handling

One canonical turn cap (for example 300) lives in the config and is used for training, evaluation, and tournaments. It is recorded in every experiment's metadata. Changing it changes the draw rate, so results under different caps aren't directly comparable. Evaluation reports include the number of cap draws and the fraction of games reaching the cap.

---

## 12. Compute and parallel training

- Record CPU, GPU, RAM, and runtime before committing to large runs.
- Simulation is expected to be CPU-bound, and a GPU won't fix that.
- Benchmark steps per second, games per hour, PPO update time, and memory.
- Use vectorized environments. Choose the count empirically (start at 4) instead of maximizing it.
- Each environment gets its own seed. Opponent selection and seat assignment are recorded.
- Every experiment declares its step budget, wall-clock budget, checkpoint frequency, and evaluation budget.
- Start with a short pilot run to set realistic budgets.

---

## 13. Initial PPO hyperparameters

These are starting hypotheses, not tuned values.

| Parameter | Initial value |
|---|---:|
| Learning rate | 0.0003 |
| Batch size | 256 |
| Rollout length per environment | 256 steps |
| Discount factor (gamma) | 0.99 |
| Entropy coefficient | 0.01 |
| PPO epochs per update | 10 |
| Clip range | 0.2 |
| Parallel environments | 4, subject to benchmarking |
| Policy | MLP |
| Shaping | Small, configurable, potential-based |

Four environments at 256 steps give 1,024 transitions per rollout, subject to episode boundaries. Monitor entropy, approximate KL, value diagnostics, throughput, win rate, and draw rate. Tune only after the environment, masks, rewards, and self-play adapter are confirmed correct.

---

## 14. Evaluation protocol

- At least **500 games** per required matchup against Random, Weighted Random, and AlphaBeta, with seats alternated as evenly as possible.
- Record wins, losses, draws, average VP margin, game length, and turn-cap frequency.
- Use **Wilson intervals** for win rate, and state how draws are treated while keeping the draw count separate.
- Use separate evaluation seeds from training seeds, and identical rules, board settings, and turn cap for comparable policies.
- Document AlphaBeta's search settings and compute budget.
- **Screening vs. final:** use cheap, frequent diagnostic evaluations during training, and reserve full 500-game matchups for promoted checkpoints. Never present screening results as equivalent to a full evaluation.
- Compute Elo from documented checkpoint matchups, specify draw handling, and read it alongside baseline results and confidence intervals.
- Include a checkpoint-vs-checkpoint test early to confirm later versions beat earlier ones.

Optional leaderboard number:

```
Final Score = 0.7 * WinRate_vs_strongest_baseline
            + 0.2 * WinRate_vs_previous_checkpoints
            + 0.1 * (avg_VP_margin / 10)
```

---

## 15. Success criteria and stopping rules

| Metric | Proposed target |
|---|---|
| Win rate vs. Random | at least 95% |
| Win rate vs. Weighted Random | at least 90% |
| Win rate vs. AlphaBeta | at least 50% (stretch goal) |
| Invalid actions via the masked interface | zero |
| Seat balance | alternated as evenly as possible |
| Confidence intervals | reported for every required matchup |

Milestones: first a reliable agent that finishes games with no invalid actions and reproducible evaluation, then a statistically supported improvement over Random and Weighted Random.

**Stop or pause training** when the step or wall-clock budget is exhausted, when repeated evaluations show no meaningful improvement, or when a correctness or stability problem appears.

**Promotion rule:** a candidate must improve a predefined metric while staying acceptable against the other baselines. Select the final checkpoint on validation results, then evaluate it once on a retained final test set. Report the best result along with its uncertainty. One favorable tournament is not enough evidence.

---

## 16. Reproducibility and logging

- Each experiment has a unique ID and master seed.
- Separate random streams cover board generation, environment resets, opponent selection, and policy sampling.
- Save the effective config, software versions, opponent pool, training progress, and checkpoint metadata together.
- Log PPO losses, entropy, value diagnostics, episode outcomes, shaping rewards, and game lengths. Record evaluation by opponent, seat, checkpoint, and seed.
- Bitwise reproducibility may not hold across hardware and software versions. The goal is reproducible enough to tell real improvement from noise.

---

## 17. Hugging Face model lifecycle

Training produces a local checkpoint, which is validated, evaluated, and then published with metadata. Anyone can download a specific revision, rebuild the policy, and run an inference smoke test.

Each release documents its training config, opponent pool, evaluation results, limitations, software versions, observation and action schema versions, and reward design. Tie releases to immutable commits or version identifiers. Never put credentials in source code, committed configs, or logs.

---

## 18. Competition and organizer requirements (unconfirmed)

Find out and record:
- opponent type (fixed bot, other teams' agents, or a human)
- per-move time limit
- hardware available at evaluation time
- allowed model size and external dependencies
- whether network access is allowed during matches
- judging metric and format (single game, multiple games, tournament, fixed boards, hidden scenarios)

These can change the architecture. A strict move deadline favors fast local inference and rules out an LLM in the decision path. If the network is blocked, the agent must not depend on downloading a model during a match. Keep interfaces configurable until this is settled.

---

## 19. Testing strategy

Correctness comes before playing strength. The test files are listed in section 5. What they cover:
- **Environment:** resets, transitions, legal actions, terminal conditions.
- **Masks:** mask equals the simulator's legal set, and invalid actions can't be selected.
- **Observations:** dimensions, player perspective, no hidden-information leakage.
- **Action mapping:** policy outputs map to the intended Catanatron actions.
- **Rewards:** win/loss/draw, shaping, consistent discounting.
- **Config and seeding:** merge behavior, frozen configs, stream independence.
- **Bots:** only legal actions, complete games.
- **Self-play:** one full game between current policy and a historical checkpoint.
- **Inference:** loading, masked action selection.
- **Determinism, training round trips, evaluation accounting, and Hub metadata.**

Run remote Hub tests separately so network failures don't block the local correctness suite.

---

## 20. Implementation order (revised: audit first)

1. **Simulator audit (first).** Install and pin Catanatron. Write `docs/catanatron_audit.md` covering 2-player support, what observations expose (including hidden information), the action space, trading support, and Gym vs. Gymnasium. This decides how observations and actions get encoded.
2. **Foundation.** Config loading, seeding, environment wrapper, observation encoding, action masking, action mapping. Write `docs/interfaces.md`.
3. **Performance baseline.** Benchmark throughput and parallel environment counts before choosing a training budget.
4. **Baseline learning.** Rewards, MLP policy, trainer, checkpoint manager, structured logging.
5. **Integration tests.** Environment, masking, inference, determinism, and one full game against a historical checkpoint.
6. **Self-play and evaluation.** Opponent pool, baseline bots, tournaments, confidence intervals, Elo.
7. **Deployment.** Hub upload/download, validation, model card.
8. **Advanced.** Better architectures and optional LLM explanations, only if results and compute justify them.

---

## 21. Catanatron audit checklist (for `docs/catanatron_audit.md`)

Fill in each item from the actual pinned version, and note the source file and version for every finding.

1. Does it support exactly 2 players, and how is that configured?
2. What does its Gym environment return: observation shape and contents, reward, termination vs. truncation?
3. Is it Gym or Gymnasium, and does it match the Stable-Baselines3 version you pin?
4. How are legal actions produced, and is the action space fixed-size or state-dependent? How does an action index map to an action?
5. Which trades exist: bank 4:1, ports 3:1 and 2:1, player-to-player?
6. What do the features expose about the opponent's hand and dev cards? Anything private leaking?
7. How does the environment configure the opponent, and can it accept an arbitrary policy as a bot?
8. How is the turn cap or max game length handled, and is truncation signaled?
9. What AlphaBeta bot ships with it, and what are its depth and heuristic settings?
10. What random-seed controls are available (board, dice, shuffles)?
11. What is the simulation speed (steps per second, single process)?

---

## 22. Team ownership (suggested)

- **You:** environment, action interfaces, configuration, seeding, training integration.
- **Your teammate:** simulator audit, benchmarking, evaluation infrastructure, baseline opponents.
- **Both:** review `docs/interfaces.md` and jointly validate `test_self_play.py`, since those are the critical integration boundaries.

---

## 23. Main risks

1. Catanatron's environment may not support loading multiple policies, so plan an adapter.
2. Hidden-information leakage would invalidate results.
3. `huggingface_sb3` and MaskablePPO compatibility is unverified.
4. Version mismatches among Catanatron, Gymnasium, and sb3-contrib.
5. AlphaBeta evaluation cost at higher search depths.
6. Inconsistent turn caps or trading rules between training and evaluation.
7. Organizer requirements (time limits, network, judging) not yet confirmed.

## Conclusion

The agent learns from simulator-generated experience, MaskablePPO provides legal-move learning, historical checkpoints provide self-play diversity, and independent evaluation measures real playing strength. The immediate next step is the Catanatron audit: pin the version, answer the section 21 questions, and let those findings shape the environment wrapper and action encoder.
