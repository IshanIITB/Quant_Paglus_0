"""The simplest legal agent. It scores about 0.000 against F. Start here."""
import random


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        return self.rng.randrange(len(playable_actions))
