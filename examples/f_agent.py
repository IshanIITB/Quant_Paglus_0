"""F, written against the agent interface. This is the bot you have to beat.

F scores a position with a weighted sum of 13 features and plays the move leading to the
best-scoring position. It looks one move ahead and no further. The weights are catanatron's,
unchanged.

Two things here are the point of the exercise.

1. F needs a full game to score a position, but your agent receives an observation. This agent
   calls obs.sample_world() ONCE, so it is guessing the opponent's development cards. Sampling
   several worlds and averaging usually decides better, and costs more CPU. That trade is yours.

2. DEFAULT_WEIGHTS names 13 features. Ask which of them the engine's 614-number feature vector
   actually contains.
"""
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_index, best_value = 0, None
        for index, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            value = VALUE_FN(successor, obs.me)
            if best_value is None or value > best_value:
                best_index, best_value = index, value
        return best_index
