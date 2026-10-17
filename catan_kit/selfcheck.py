"""Check that the kit is installed correctly. Run this first.

    python -m catan_kit.selfcheck

If a check fails, fix it before you write any agent code: a broken environment produces
plausible numbers, not error messages.
"""
import importlib.metadata as md
import os
import random
import sys

#: The interpreter we score on.
REFERENCE_PYTHON = (3, 11)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)
    return condition


def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()          # may restart this process

    print("catan_kit self-check\n")

    print("environment")
    # catanatron 3.3.0, numpy, scipy and scikit-learn all require Python 3.11 or newer.
    check("python >= 3.11", sys.version_info >= (3, 11), f"found {sys.version.split()[0]}")
    # A newer interpreter is not a failure. We score every submission on our own machine, so it
    # changes your local numbers and not your rank. Measured on an older build: the same 60-game
    # run gave 3901 agent decisions on 3.11 and 3847 on 3.9, with every dependency pinned and
    # PYTHONHASHSEED=0. Using 3.11 is the only way to match us exactly.
    if sys.version_info[:2] != REFERENCE_PYTHON:
        print(f"  [note] we score on Python {REFERENCE_PYTHON[0]}.{REFERENCE_PYTHON[1]}. "
              f"You are on {sys.version_info[0]}.{sys.version_info[1]}, so your local numbers "
              f"will differ slightly from the leaderboard.")
    try:
        version = md.version("catanatron")
    except Exception:
        version = "not installed"
    check("catanatron == 3.3.0", version == "3.3.0", f"found {version}")
    try:
        nx = md.version("networkx")
    except Exception:
        nx = "not installed"
    check("networkx == 3.6.1", nx == "3.6.1", f"found {nx}")

    import catan_kit
    from catan_kit import _patch
    check("determinism patch applied", _patch.is_applied())
    check("PYTHONHASHSEED == 0", os.environ.get("PYTHONHASHSEED") == "0",
          f"found {os.environ.get('PYTHONHASHSEED')!r}")

    print("\nengine")
    from catanatron.models.enums import ActionType
    expected = {
        "ROLL", "END_TURN", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
        "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "MOVE_ROBBER", "DISCARD_RESOURCE",
        "MARITIME_TRADE", "OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE",
        "CONFIRM_TRADE", "CANCEL_TRADE",
    }
    found = {a.name for a in ActionType}
    check("action set matches the pinned engine", found == expected,
          "" if found == expected else f"unexpected: {sorted(found ^ expected)}")

    print("\ndeterminism  (the same seed must give the same game)")
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catanatron.players.value import ValueFunctionPlayer

    def fingerprint(seed):
        random.seed(seed)
        g = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)],
                 seed=seed, vps_to_win=10)
        return str(g.play()), g.state.num_turns

    prints = [fingerprint(4256) for _ in range(4)]
    check("one game repeats exactly", len(set(prints)) == 1, f"{prints[0]} x{len(prints)}")
    check("seed 4256 matches the reference", prints[0] == ("Color.RED", 111),
          f"got {prints[0]}, reference ('Color.RED', 111)")

    print("\nexample agents")
    from catan_kit.evaluate import run
    import pathlib
    here = pathlib.Path(__file__).resolve().parent.parent / "examples"
    rnd, f = str(here / "random_agent.py"), str(here / "f_agent.py")

    r = run(f, rnd, games=20, quiet=True)
    check("f_agent beats random_agent", r["win_rate"] >= 0.9,
          f"win rate {r['win_rate']:.2f} over 20 games")
    check("f_agent raises no errors", r["errors"] == 0, f"{r['errors']} errors")
    check("f_agent stays inside the CPU clock", r["games_over_budget"] == 0,
          f"{r['mean_cpu_s']*1000:.0f} ms mean per game, {r['games_over_budget']} games over")

    r = run(rnd, f, games=20, quiet=True)
    check("random_agent loses to f_agent", r["win_rate"] <= 0.1,
          f"win rate {r['win_rate']:.2f} over 20 games")

    a = run(f, rnd, games=15, quiet=True)
    b = run(f, rnd, games=15, quiet=True)
    check("a repeated run is identical", a["decisions"] == b["decisions"],
          f"{a['decisions']} then {b['decisions']} decisions")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed. You are ready to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
