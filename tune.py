"""Local-only weight tuner (do NOT put this in the submission zip).

(1 + lambda) evolution strategy on log-multipliers of F's weights.
  * Objective = mean win rate over the WORST HALF of an opponent pool (same shape
    as the robustness score), not the win rate against F alone.
  * Pool = F, two randomly perturbed copies of F, plus any --opp files.
  * A child is accepted only after a confirmation run on fresh seeds, and only if
    it beats the parent by z standard errors (approximate, see se_diff).
  * Weights that are 0 in F's defaults stay frozen at 0. State this in the report.
  * By default the lookahead is OFF while tuning (much cheaper). Weights tuned
    without search are then used with search; state that mismatch in the report and
    validate the final weights with search on (or use --with-search).

Each candidate runs from a temporary copy of my_team/, so agent.py needs no
environment switches.

Example (after checking `python -m catan_kit.evaluate --help` for the seed flag):
    python tools/tune.py --seed-flag seed --workers 6

--seed-flag takes the flag name WITHOUT dashes ("seed" becomes --seed).

Cost: the tool prints games per generation at start. Do timing-sensitive checks of
the search agent (CPU per game, games over the clock) in a separate quiet run,
because parallel load inflates CPU readings.

Before trusting it:
  * Check WIN_RE against what evaluate.py actually prints.
  * If evaluate.py already uses several processes, keep --workers small.
  * Check how evaluate.py uses its seed flag (start index or single seed). Seeds
    here are spaced by --seed-gap so generations never share boards if it is a
    start index.
  * Sanity check: run once with absurd weights and confirm the win rate collapses.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from catanatron.players.value import DEFAULT_WEIGHTS

AGENT_DIR = "my_team"
F_AGENT = "examples/f_agent.py"
WIN_RE = re.compile(r"win\s*rate[^0-9]*([0-9]*\.?[0-9]+)", re.I)
CONFIRM_BASE = 5_000_000   # confirmation seeds live far from training seeds

F_TEMPLATE = '''import random

from catanatron.players.value import get_value_fn

VALUE_FN = get_value_fn("base_fn", {weights!r})


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = VALUE_FN(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
'''


def write_perturbed_f(path, seed, sigma=0.5):
    rng = random.Random(seed)
    weights = {k: v * math.exp(sigma * rng.gauss(0, 1)) if v != 0 else v
               for k, v in DEFAULT_WEIGHTS.items()}
    with open(path, "w") as f:
        f.write(F_TEMPLATE.format(weights=weights))


def make_candidate_dir(weights, search):
    root = tempfile.mkdtemp(prefix="catan_cand_")
    dst = os.path.join(root, "agent_dir")
    shutil.copytree(AGENT_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "weights.json"), "w") as f:
        json.dump(weights, f)
    if not search:
        agent_py = os.path.join(dst, "agent.py")
        with open(agent_py) as f:
            src = f.read()
        assert "SEARCH = True" in src, "agent.py no longer has the SEARCH = True line"
        with open(agent_py, "w") as f:
            f.write(src.replace("SEARCH = True", "SEARCH = False", 1))
    return root, os.path.join(dst, "agent.py")


def run_eval(agent_py, opponent, games, seed, seed_flag):
    cmd = [sys.executable, "-m", "catan_kit.evaluate", "--a", agent_py,
           "--b", opponent, "--games", str(games)]
    if seed_flag:
        cmd += ["--" + seed_flag.lstrip("-"), str(seed)]
    last_error = None
    for _ in range(3):   # retry: a crash or odd output should not kill the tuner
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            text = proc.stdout + proc.stderr
            m = WIN_RE.search(text)
            if not m:
                raise RuntimeError("could not parse win rate; output tail:\n" + text[-800:])
            rate = float(m.group(1))
            return rate / 100.0 if rate > 1.0 else rate
        except Exception as e:
            last_error = e
            time.sleep(2)
    raise last_error


def pool_rates(weights, pool, games, seed, args):
    """Per-opponent win rates, or None if the evaluation kept failing."""
    root, agent_py = make_candidate_dir(weights, args.with_search)
    try:
        return [run_eval(agent_py, opp, games, seed, args.seed_flag) for opp in pool]
    except Exception as e:
        print("evaluation failed:", e, flush=True)
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def objective(rates):
    """Mean of the worst half of the per-opponent win rates (-1 for a failed run)."""
    if rates is None:
        return -1.0
    worst = sorted(rates)[:max(1, math.ceil(len(rates) / 2))]
    return sum(worst) / len(worst)


def se_diff(games, n_pool):
    """Rough standard error of a difference of two objectives (conservative p = 0.5)."""
    m = max(1, math.ceil(n_pool / 2))
    return math.sqrt(2 * 0.25 / (games * m))


def factors_from_weights(path, tunable):
    with open(path) as f:
        weights = json.load(f)
    factors = {}
    for k in tunable:
        d, v = DEFAULT_WEIGHTS[k], weights.get(k, DEFAULT_WEIGHTS[k])
        factors[k] = math.log(v / d) if v and (v > 0) == (d > 0) else 0.0
    return factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="games per opponent per candidate")
    ap.add_argument("--confirm-games", type=int, default=300)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--lam", type=int, default=4, help="children per generation")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--z", type=float, default=1.64)
    ap.add_argument("--workers", type=int, default=6, help="parallel evaluations (cores minus 1-2)")
    ap.add_argument("--rng-seed", type=int, default=0, help="seed of the tuner's own random numbers")
    ap.add_argument("--seed-flag", default=None,
                    help="evaluate.py flag that sets board seeds, without dashes (e.g. seed)")
    ap.add_argument("--seed-gap", type=int, default=10_000, help="spacing between generations' seeds")
    ap.add_argument("--with-search", action="store_true", help="tune with the lookahead on (slow)")
    ap.add_argument("--init", default=None, help="start from an existing weights.json")
    ap.add_argument("--opp", action="append", default=[], help="extra opponent agent files")
    ap.add_argument("--out", default="my_team/weights.json")
    args = ap.parse_args()
    if not args.seed_flag:
        print("WARNING: no --seed-flag given, so every run reuses the same boards.")

    os.makedirs("tools/opponents", exist_ok=True)
    os.makedirs("tools/runs", exist_ok=True)
    pool = [F_AGENT]
    for s in (1, 2):
        p = f"tools/opponents/f_perturbed_{s}.py"
        write_perturbed_f(p, seed=s)
        pool.append(p)
    pool += args.opp
    per_gen = args.lam * len(pool) * args.games + 2 * len(pool) * args.confirm_games
    print(f"opponent pool: {pool}\ngames per generation: {per_gen} "
          f"(x {args.gens} generations)", flush=True)

    tunable = [k for k, v in DEFAULT_WEIGHTS.items() if v != 0]
    rng = random.Random(args.rng_seed)

    def weights_of(factors):
        return {k: DEFAULT_WEIGHTS[k] * math.exp(factors.get(k, 0.0)) for k in DEFAULT_WEIGHTS}

    parent = factors_from_weights(args.init, tunable) if args.init else {k: 0.0 for k in tunable}
    sigma = args.sigma
    executor = ThreadPoolExecutor(args.workers)

    log_path = "tools/runs/tune_log.csv"
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["time", "gen", "sigma", "games", "child_obj", "confirm_diff",
                          "margin", "accepted", "child_rates", "parent_rates"])
        for gen in range(1, args.gens + 1):
            train_seed = 1000 + args.rng_seed * 100_000_000 + gen * args.seed_gap
            confirm_seed = CONFIRM_BASE + args.rng_seed * 100_000_000 + gen * args.seed_gap
            children = [{k: parent[k] + sigma * rng.gauss(0, 1) for k in tunable}
                        for _ in range(args.lam)]
            rates = list(executor.map(
                lambda c: pool_rates(weights_of(c), pool, args.games, train_seed, args), children))
            objs = [objective(r) for r in rates]
            j = max(range(len(children)), key=objs.__getitem__)
            if objs[j] < 0:
                print(f"gen {gen}: every child evaluation failed, skipping", flush=True)
                continue

            # Confirm the best child against the parent on fresh seeds.
            child_r, parent_r = executor.map(
                lambda c: pool_rates(weights_of(c), pool, args.confirm_games, confirm_seed, args),
                [children[j], parent])
            if child_r is None or parent_r is None:
                print(f"gen {gen}: confirmation failed, skipping", flush=True)
                continue
            diff = objective(child_r) - objective(parent_r)
            margin = args.z * se_diff(args.confirm_games, len(pool))
            accepted = diff > margin
            log.writerow([int(time.time()), gen, round(sigma, 3), args.games, round(objs[j], 4),
                          round(diff, 4), round(margin, 4), accepted,
                          [round(x, 3) for x in child_r], [round(x, 3) for x in parent_r]])
            logf.flush()
            print(f"gen {gen}: best child {objs[j]:.3f} | confirm diff {diff:+.3f} vs margin "
                  f"{margin:.3f} | accepted={accepted} sigma={sigma:.2f}", flush=True)
            if accepted:
                parent = children[j]
                with open(args.out, "w") as f:
                    json.dump(weights_of(parent), f, indent=1)
            else:
                sigma *= 0.85
    print("done. Validate my_team/weights.json on 1000 fresh games with search ON, "
          "including vs F, before submitting.")


if __name__ == "__main__":
    main()
