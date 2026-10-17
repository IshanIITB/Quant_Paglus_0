# Catan RL Agent (Hyde @ IITB)

An AI agent that plays **two-player Catan**. This README explains the game (from the briefing deck), why reinforcement learning fits, which Hugging Face models and tools you can use, and a recommended scoring system.

> **Note:** the PDF is a rules primer only. It doesn't state the exact task or judging criteria. This README assumes the goal is "build the strongest 2-player Catan agent." If the organizers specify a scoring rubric, use that for the final evaluation and keep the training reward below as your internal signal.

---

## 1. The game in one page

| Concept | Rule (from the deck) |
|---|---|
| Players | 2 compete on a resource-rich island |
| Board | Hexagonal tiles; each tile makes one resource and has a dice number |
| Resources | Wood (forest), Brick (hills), Sheep (pasture), Wheat (fields), Ore (mountains) |
| Production | Roll dice; every settlement/city adjacent to a tile with that number collects. A city produces 2 |
| Building | Spend resources on roads, settlements, cities, or development cards |
| Dev cards | Knight (move robber, steal), Road Building (2 free roads), Year of Plenty (take any 2 resources), Victory Point (hidden, 1 VP) |
| Robber | Rolling a 7 activates it: blocks a tile and may steal a resource |
| Trading | Bank trades at 4:1, or 3:1 / 2:1 at ports |
| **Win** | **First to 10 Victory Points** |

### Victory points

| Source | VP |
|---|---|
| Settlement | +1 |
| City | +2 total (upgrade from settlement) |
| Victory Point card | +1 each |
| Largest Army (3+ Knights played) | +2 |
| Longest Road (5+ connected segments) | +2 |

Roads are worth 0 VP by themselves; they're what lets you reach new settlement spots and fight for Longest Road.

---

## 2. Why this is an RL problem

- **Long horizon:** hundreds of decisions, and the win/loss signal arrives only at the end.
- **Stochastic:** dice and shuffled dev cards.
- **Hidden information:** the opponent's hand and dev cards are unknown.
- **Large, changing action space:** legal moves differ every turn (where to build, what to trade, where to put the robber).

Practical consequence: use **action masking** so the agent can only choose legal moves.

---

## 3. Recommended approach

There are two realistic paths. For a hackathon-length project, **Path A is the safer bet**.

### Path A: Classic RL agent (recommended)
A small neural policy trained by self-play on a fast simulator.

- **Simulator:** [Catanatron](https://github.com/bcollazo/catanatron), an open-source Catan engine with a Gym-style environment and built-in baseline bots. Configure it for 2 players.
- **Algorithm:** **MaskablePPO** from `sb3-contrib` (Stable-Baselines3). It supports invalid-action masking out of the box.
- **Observation:** board features (tile resource, number, ownership), your hand, opponent's public info, VP totals, dev cards held, robber location.
- **Hugging Face role:** SB3 has first-class Hub integration through the `huggingface_sb3` package. Use it to push and pull checkpoints (`load_from_hub` / `package_to_hub`). Browse the Hub for `catan` to check whether anyone has published a pretrained checkpoint, but don't count on finding one.
- **Optional upgrade:** AlphaZero-style search (MCTS guided by a policy/value net). It is stronger but takes much more engineering.

### Path B: LLM agent trained with RL
An instruction-tuned language model reads the game state as text and outputs a move. Train it with **GRPO** using Hugging Face **TRL** (`GRPOTrainer`), optionally with **Unsloth** for memory-efficient training.

Hugging Face models that are capable and practical for this:

| Model | Size | Why consider it |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Fast iteration; fits on a single consumer GPU or Colab |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Good balance of reasoning and speed |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Strongest of this family for strategy; needs LoRA/QLoRA |
| `Qwen/Qwen3-4B` (or `Qwen3-8B`) | 4B / 8B | Newer family with strong reasoning; good GRPO base |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Solid small model (gated; accept the license on the Hub) |
| `google/gemma-2-2b-it` | 2B | Lightweight alternative |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Very cheap to train; good for prototyping |

Tips: use LoRA/QLoRA, give the model a compact text description of the state plus an explicit list of legal moves, and have it pick an index from that list (so illegal moves are impossible). Verify exact model IDs and licenses on the Hub before you start, since names and availability change.

### Path C (offline RL, optional)
`transformers` includes a **Decision Transformer** (`DecisionTransformerModel`). If you can generate lots of games from strong bots, you can train on those trajectories offline, then fine-tune with self-play.

### Honest comparison

| | Path A (MaskablePPO) | Path B (LLM + GRPO) |
|---|---|---|
| Speed of simulation/training | Fast | Slow (every move is an LLM call) |
| Likely strength | Higher, given enough self-play | Lower unless heavily trained |
| Explainable moves | No | Yes (can output reasoning) |
| Setup effort | Moderate | Higher |

A good hybrid: train Path A for strength, and optionally use an LLM only to narrate or explain moves in the demo.

---

## 4. Recommended scoring system

Keep three things separate: the **game score**, the **training reward**, and the **evaluation metrics**.

### 4.1 Game score (official)
Use the deck's rules: **first to 10 VP wins.** Track VP from settlements, cities, VP cards, Largest Army and Longest Road.

### 4.2 Training reward (what the agent learns from)

Use a sparse terminal reward plus *potential-based* shaping so shaping can't change what the optimal strategy is.

```
R_terminal = +1 (win)   |   -1 (loss)   |   0 (turn-cap draw)

R_shaping  = λ · ( γ·Φ(s') − Φ(s) )          # λ ≈ 0.1–0.3

Φ(s) = (my_VP − opp_VP) / 10
       + 0.02 · (my_production_pips − opp_production_pips) / 10
       + 0.01 · resource_diversity            # distinct resource types produced (0–5) / 5
```

| Term | Purpose |
|---|---|
| Win/loss ±1 | The true objective |
| VP difference | Dense signal that tracks progress toward 10 VP |
| Production pips (dice-probability dots on adjacent tiles) | Rewards strong economy early, before VP appears |
| Resource diversity | Discourages starving on a missing resource |

Guidelines:
- Keep shaping small relative to the ±1 win/loss, or the agent will farm shaping rewards instead of winning.
- Set a **turn cap** (for example 300 turns) so stalled games end; treat caps as a draw (0).
- Do **not** reward raw resource hoarding. It encourages stalling and is easy to exploit.
- Optional penalty: −0.001 per step to discourage dithering.

### 4.3 Evaluation metrics (how you judge the final agent)

| Metric | How |
|---|---|
| **Win rate** (primary) | vs. baselines: Random → Weighted Random → Catanatron's stronger bots (e.g., AlphaBeta) |
| Average final VP and VP margin | Shows how close losses are and how decisive wins are |
| Average game length (turns) | Efficiency |
| **Elo rating** | Round-robin among your own checkpoints and baselines |
| Statistical confidence | At least 500 games per matchup; report 95% confidence intervals |
| Seat fairness | Alternate who goes first; first-player advantage is real |

A simple leaderboard score if you want one number:

```
Final Score = 0.7 · WinRate_vs_strongest_baseline
            + 0.2 · WinRate_vs_previous_checkpoints
            + 0.1 · (avg_VP_margin / 10)
```

---

## 5. Suggested roadmap

1. Install Catanatron and run bot-vs-bot games to confirm the 2-player setup.
2. Wrap the environment with action masking; log observations and legal-move counts.
3. Train MaskablePPO against Random, then against stronger bots, then via self-play.
4. Evaluate every N updates with the metrics above; keep the best checkpoints.
5. Push checkpoints to the Hugging Face Hub with `huggingface_sb3`.
6. (Optional) Add an LLM layer for move explanations, or try GRPO with a small Qwen model.

---

## 6. Open questions to confirm with the organizers

- Is the opponent a fixed bot, another team's agent, or a human?
- Are there time limits per move?
- Are player-to-player trades allowed, or bank-only (the deck only covers bank/port trades)?
- Is the final judging by win rate, a leaderboard, or a live match?
