"""Trimmed Catan agent.

Layers
  0. F's value function (catanatron base_fn) scoring the position after each move.
  1. Optional tuned weights from weights.json (empty file = exactly F's weights).
  2. Budgeted same-turn lookahead on the top-K moves (own action sequences only).
     Moves with random outcomes (buying a development card, robber steals) are
     averaged over a few outcomes instead of trusting one sample.
  3. Clock guard: budget from obs.cpu_left, a reserve, and a cheap emergency path.
  4. Safety wrapper: any exception falls back to a random legal index.

No environment variables are read. For a local no-search ablation, set
SEARCH = False below (tools/tune.py --no-search does this in a temp copy).
"""
import json
import os
import random
import tempfile
import time

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = True

# --- search / clock settings (CPU seconds). Profile and adjust. ---
TOP_K = 3                  # candidates that get a lookahead in a normal decision
TOP_K_OPENING = 6          # candidates for an opening settlement
RANDOM_EXTRA = 2           # extra outcomes averaged for dev-card buys / robber steals
OPENING_BUDGET = 0.08      # per opening settlement; two of them take about 150 ms of
                           # the 300 ms start, opening roads then get one-ply only
MAX_STEP = 0.05            # ceiling for one normal decision (the real limit is below)
RESERVE = 0.05             # floor of allowance never planned away
RESERVE_DECISIONS = 15     # reserve is at least this many average one-ply decisions
EXPECTED_DECISIONS = 56    # average decisions per game, from the problem statement
MIN_REMAINING = 20         # never assume fewer than this many decisions remain
WIN_VALUE = 1e30

CHEAP_PRIORITY = ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD",
                  "BUILD_ROAD", "PLAY_KNIGHT_CARD", "MARITIME_TRADE", "END_TURN"]


def _load_weights():
    weights = dict(DEFAULT_WEIGHTS)
    try:
        with open(os.path.join(HERE, "weights.json")) as f:
            override = json.load(f)
        for key, value in override.items():
            if key in weights:
                weights[key] = float(value)
    except Exception:
        pass  # missing or bad file: fall back to F's default weights
    return weights


def _winner(game):
    try:
        return game.winning_color()
    except Exception:
        return None


def _name(action):
    try:
        return action.action_type.name
    except Exception:
        return ""


def _is_random(action):
    """True if the action's outcome depends on a random draw we only sampled once."""
    name = _name(action)
    if name == "BUY_DEVELOPMENT_CARD":
        return True
    if name == "MOVE_ROBBER":
        try:
            return action.value[1] is not None  # a victim means a random card steal
        except Exception:
            return False
    return False


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.value_fn = get_value_fn("base_fn", _load_weights())
        self.n = 0              # decisions asked so far in the current game
        self.last_turn = 0
        self.ema = 0.004        # running average CPU of one decision's one-ply pass
        self.stats = {"decisions": 0, "searched": 0, "timeout_dropped": 0,
                      "emergency": 0, "fallback_random": 0}
        self._warm_up()         # loading is not charged to any game

    def _warm_up(self):
        try:
            from catanatron import Color, Game, RandomPlayer

            game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
            self.value_fn(game, Color.RED)
            child = game.copy()
            child.execute(game.playable_actions[0])
            self.value_fn(child, Color.RED)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def decide(self, obs, playable_actions) -> int:
        try:
            index = self._decide(obs, playable_actions)
        except Exception:
            self.stats["fallback_random"] += 1
            index = self.rng.randrange(len(playable_actions))
        self._dump_stats()
        return index

    def _decide(self, obs, actions):
        t0 = time.process_time()
        turn = getattr(obs, "turn", 0)
        if turn < self.last_turn:   # turn counter went back: a new game started
            self.n = 0
        self.last_turn = turn
        self.n += 1
        self.stats["decisions"] += 1

        cpu_left = getattr(obs, "cpu_left", 1.0)
        if cpu_left < 2 * self.ema:  # nearly out of allowance: skip evaluation
            self.stats["emergency"] += 1
            return self._cheap(actions)

        me = obs.me
        world = obs.sample_world(self.rng)
        opening = self._is_opening(world, actions)
        deadline = t0 + self._budget(cpu_left, opening)

        # One-ply pass (this is F). Take a sure win if there is one.
        scored = []  # (value, index, successor)
        for i, action in enumerate(actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            if not _is_random(action) and _winner(succ) == me:
                return i
            scored.append((self.value_fn(succ, me), i, succ))
        self.ema = 0.8 * self.ema + 0.2 * (time.process_time() - t0)
        if not scored:
            return self.rng.randrange(len(actions))
        scored.sort(key=lambda t: -t[0])
        best_i = scored[0][1]
        if not SEARCH or len(scored) == 1:
            return best_i

        # Same-turn lookahead on the top-K candidates, within the budget.
        k = TOP_K_OPENING if opening else TOP_K
        results = []  # (value, index)
        for _, i, succ in scored[:k]:
            if time.process_time() >= deadline:
                break
            value = self._candidate_value(obs, world, actions[i], succ, me, deadline)
            if value is None:       # ran out of time mid-candidate: drop it
                self.stats["timeout_dropped"] += 1
                break
            results.append((value, i))
        if len(results) >= 2:       # only compare candidates that finished
            self.stats["searched"] += 1
            return max(results, key=lambda r: r[0])[1]
        return best_i

    # ------------------------------------------------------------------
    def _candidate_value(self, obs, world, action, succ, me, deadline):
        """Mean lookahead value over a few outcomes for random moves, else one outcome."""
        random_outcome = _is_random(action)
        outcomes = [succ]
        if random_outcome:
            for _ in range(RANDOM_EXTRA):
                if time.process_time() >= deadline:
                    return None
                extra = self._resample(obs, world, action)
                if extra is not None:
                    outcomes.append(extra)
        total = 0.0
        for outcome in outcomes:
            value = self._lookahead(outcome, me, deadline, check_win=not random_outcome)
            if value is None:
                return None
            total += value
        return total / len(outcomes)

    def _resample(self, obs, world, action):
        try:
            # A dev card draw is fixed by the sampled deck order, so draw a new world.
            # A robber steal is random inside the engine, so re-executing is enough.
            base = obs.sample_world(self.rng) if _name(action) == "BUY_DEVELOPMENT_CARD" else world
            outcome = base.copy()
            outcome.execute(action)
            return outcome
        except Exception:
            return None

    def _lookahead(self, succ, me, deadline, check_win):
        """Best one-ply value of a follow-up inside my own turn, or None on timeout."""
        if check_win and _winner(succ) == me:
            return WIN_VALUE
        if succ.state.current_color() != me:   # my turn ended: just score it
            return self.value_fn(succ, me)
        best = None
        for action in succ.playable_actions:
            if time.process_time() >= deadline:
                return None
            child = succ.copy()
            try:
                child.execute(action)
            except Exception:
                continue
            if not _is_random(action) and _winner(child) == me:
                return WIN_VALUE
            value = self.value_fn(child, me)
            if best is None or value > best:
                best = value
        return best if best is not None else self.value_fn(succ, me)

    # ------------------------------------------------------------------
    def _is_opening(self, world, actions):
        try:
            return bool(getattr(world.state, "is_initial_build_phase", False)) and (
                _name(actions[0]) == "BUILD_SETTLEMENT"
            )
        except Exception:
            return False

    def _budget(self, cpu_left, opening):
        """CPU seconds this decision may use, including the one-ply pass."""
        if opening:
            return min(OPENING_BUDGET, 0.3 * cpu_left)
        reserve = max(RESERVE, RESERVE_DECISIONS * self.ema)
        remaining = max(MIN_REMAINING, EXPECTED_DECISIONS - self.n)
        return min(MAX_STEP, max(0.0, cpu_left - reserve) / remaining)

    def _cheap(self, actions):
        """No evaluation at all: pick by a fixed action-type priority."""
        for name in CHEAP_PRIORITY:
            for i, action in enumerate(actions):
                if _name(action) == name:
                    return i
        return 0

    def _dump_stats(self):
        if self.stats["decisions"] % 50:
            return
        try:
            path = os.path.join(tempfile.gettempdir(), f"catan_agent_stats_{os.getpid()}.json")
            with open(path, "w") as f:
                json.dump(self.stats, f)
        except Exception:
            pass
