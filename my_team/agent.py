"""Tuned F-style heuristic Catan agent with budgeted same-turn lookahead.

Implements the decision procedure in APPROACH_REPORT.md section 4. Variant
toggles for the ablation plan are read from the CATAN_AGENT_VARIANT env var:
  v1   = one-ply scoring + sure-win check (lookahead off, averaging off)
  v2   = v1 + same-turn lookahead
  v2.1 = v2 + random-outcome averaging
  v3   = v2.1 with tuned weights (default; weights come from weights.json)
Weights load from weights.json beside this file; a missing or bad file means
F's DEFAULT_WEIGHTS. Nothing in this file ever raises.
"""
import json
import os
import random
import sys
import time
import traceback

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VARIANT = os.environ.get("CATAN_AGENT_VARIANT", "v3")
LOOKAHEAD_ON = VARIANT in ("v2", "v2.1", "v3")
AVERAGING_ON = VARIANT in ("v2.1", "v3")

LOOKAHEAD_CANDIDATES = 3
OPENING_CANDIDATES = 6
MAX_CONTINUATION_STEPS = 6
SAMPLES = 3
EMERGENCY_S = 0.05
SKIP_LOOKAHEAD_S = 0.35
OPENING_SKIP_LOOKAHEAD_S = 0.20
LOOKAHEAD_BUDGET_S = 0.012
OPENING_LOOKAHEAD_BUDGET_S = 0.045


def _load_weights():
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return dict(DEFAULT_WEIGHTS)
        parsed = {}
        for key in DEFAULT_WEIGHTS:
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return dict(DEFAULT_WEIGHTS)
            parsed[key] = float(value)
        return parsed
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def _emergency_index(actions):
    for name in ("BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD", "END_TURN"):
        for index, action in enumerate(actions):
            if action.action_type.name == name:
                return index
    return 0


def _is_random(action):
    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return True
    if action.action_type == ActionType.MOVE_ROBBER:
        try:
            return action.value is not None and action.value[1] is not None
        except Exception:
            return False
    return False


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.value_fn = get_value_fn("base_fn", _load_weights())
        self.debug = os.environ.get("CATAN_AGENT_DEBUG") == "1"
        self.faults = 0

    def decide(self, obs, playable_actions) -> int:
        try:
            actions = list(playable_actions)
            index = self._decide(obs, actions)
            if not isinstance(index, int) or not 0 <= index < len(actions):
                raise ValueError(f"bad index {index!r}")
            return index
        except Exception:
            self.faults += 1
            if self.debug:
                sys.stderr.write("agent fallback:\n" + traceback.format_exc())
            return 0

    def _decide(self, obs, actions):
        t0 = time.monotonic()
        cpu_left = obs.cpu_left if obs.cpu_left is not None else 10.0
        if cpu_left < EMERGENCY_S:
            return self._emergency_index(actions)
        world = obs.sample_world(self.rng)
        opening = bool(getattr(world.state, "is_initial_build_phase", False))
        scored = self._one_ply(world, obs, actions)
        if not scored:
            return self._emergency_index(actions)
        if scored[0][0] == float("inf"):
            return scored[0][1]
        if LOOKAHEAD_ON:
            budget = OPENING_LOOKAHEAD_BUDGET_S if opening else LOOKAHEAD_BUDGET_S
            skip_at = OPENING_SKIP_LOOKAHEAD_S if opening else SKIP_LOOKAHEAD_S
            width = OPENING_CANDIDATES if opening else LOOKAHEAD_CANDIDATES
            if cpu_left >= skip_at and time.monotonic() - t0 < budget:
                finished = {}
                for value, index, successor, action in scored[:width]:
                    if successor is None:
                        continue
                    if time.monotonic() > t0 + budget:
                        break
                    cont = self._continuation(successor, obs.me, t0 + budget)
                    if cont is not None:
                        finished[index] = cont
                if finished:
                    best_index = max(finished, key=finished.get)
                    if finished[best_index] > scored[0][0]:
                        return best_index
        return scored[0][1]

    def _one_ply(self, world, obs, actions):
        results = []
        for index, action in enumerate(actions):
            successor = None
            if _is_random(action) and AVERAGING_ON:
                value = self._random_value(world, obs, action)
            else:
                try:
                    successor = world.copy()
                    successor.execute(action)
                except Exception:
                    continue
                value = self.value_fn(successor, obs.me)
                if successor.winning_color() == obs.me:
                    return [(float("inf"), index, successor, action)]
            results.append((value, index, successor, action))
        results.sort(key=lambda item: item[0], reverse=True)
        return results

    def _random_value(self, world, obs, action):
        total = 0.0
        count = 0
        for _ in range(SAMPLES):
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                sample = obs.sample_world(self.rng)
            else:
                sample = world
            successor = sample.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            total += self.value_fn(successor, obs.me)
            count += 1
        if count == 0:
            return None
        return total / count

    def _continuation(self, world, me, deadline):
        try:
            game = world
            forced = 0
            scored_steps = 0
            while forced + scored_steps < MAX_CONTINUATION_STEPS:
                if time.monotonic() > deadline:
                    return None
                if game.winning_color() is not None:
                    break
                if game.state.current_color() != me:
                    break
                acts = list(game.playable_actions)
                if not acts:
                    break
                if len(acts) == 1:
                    game.execute(acts[0])
                    forced += 1
                    continue
                if scored_steps >= 1:
                    break
                best_action, best_value = None, None
                for action in acts:
                    if time.monotonic() > deadline:
                        return None
                    successor = game.copy()
                    try:
                        successor.execute(action)
                    except Exception:
                        continue
                    value = self.value_fn(successor, me)
                    if best_value is None or value > best_value:
                        best_value, best_action = value, action
                if best_action is None:
                    return None
                game.execute(best_action)
                scored_steps += 1
            return self.value_fn(game, me)
        except Exception:
            return None
