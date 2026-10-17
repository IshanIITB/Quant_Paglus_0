"""Local-only weight tuner (do NOT put this in the submission zip).

(1 + lambda) evolution strategy on log-multipliers of F's weights.
  * Objective = mean win rate over the WORST HALF of an opponent pool (the same
    shape as the robustness score), not the win rate against F alone.
  * Pool = F, two randomly perturbed copies of F, plus any --opp files.
  * A child is accepted only after a confirmation run on fresh seeds, and only if
    it beats the parent by z standard errors (approximate, see se_diff).
  * Weights that are 0 in F's defaults stay frozen at 0. State this in the report.

Each candidate runs from a temporary copy of my_team/, so the agent itself needs
no environment switches.

    python tools/tune.py --games 200 --confirm-games 400 --gens 20 --lam 8 \
        --workers 3 --seed-arg --seed

Before trusting it:
  * Run `python -m catan_kit.evaluate --help`. Pass the real seed flag via
    --seed-arg, otherwise every run reuses the same boards and the
    "fresh seeds" confirmation is not fresh (the script warns you).
  * Check WIN_RE against what evaluate.py prints.
  * If evaluate.py already uses several processes, keep --workers small.
  * Sanity check: tune once with absurd weights and confirm the win rate collapses.
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


def run_eval(agent_py, opponent, games, seed, seed_arg):
    cmd = [sys.executable, "-m", "catan_kit.evaluate", "--a", agent_py,
           "--b", opponent, "--games", str(games)]
    if seed_arg:
        cmd += [seed_arg, str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200).stdout
    m = WIN_RE.search(out)
    if not m:
        raise RuntimeError("could not parse win rate; output tail:\n" + out[-800:])
    rate = float(m.group(1))
    return rate / 100.0 if rate > 1.0 else rate


def pool_rates(weights, pool, games, seed, args):
    root, agent_py = make_candidate_dir(weights, not args.no_search)
    try:
        return [run_eval(agent_py, opp, games, seed, args.seed_arg) for opp in pool]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def objective(rates):
    """Mean of the worst half of the per-opponent win rates."""
    worst = sorted(rates)[:max(1, math.ceil(len(rates) / 2))]
    return sum(worst) / len(worst)


def se_diff(games, n_pool):
    """Rough standard error of a difference of two objectives (conservative p = 0.5)."""
    m = max(1, math.ceil(n_pool / 2))
    return math.sqrt(2 * 0.25 / (games * m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--confirm-games", type=int, default=400)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--z", type=float, default=1.64)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-arg", default=None, help="evaluate.py flag that sets board seeds")
    ap.add_argument("--no-search", action="store_true", help="tune weights without the lookahead")
    ap.add_argument("--opp", action="append", default=[], help="extra opponent agent files")
    ap.add_argument("--out", default="my_team/weights.json")
    args = ap.parse_args()
    if not args.seed_arg:
        print("WARNING: no --seed-arg given, so every run reuses the same boards.")

    os.makedirs("tools/opponents", exist_ok=True)
    os.makedirs("tools/runs", exist_ok=True)
    pool = [F_AGENT]
    for s in (1, 2):
        p = f"tools/opponents/f_perturbed_{s}.py"
        write_perturbed_f(p, seed=s)
        pool.append(p)
    pool += args.opp
    print("opponent pool:", pool, flush=True)

    tunable = [k for k, v in DEFAULT_WEIGHTS.items() if v != 0]
    rng = random.Random(args.seed)

    def weights_of(factors):
        return {k: DEFAULT_WEIGHTS[k] * math.exp(factors.get(k, 0.0)) for k in DEFAULT_WEIGHTS}

    parent = {k: 0.0 for k in tunable}
    sigma = args.sigma
    executor = ThreadPoolExecutor(args.workers)

    log_path = "tools/runs/tune_log.csv"
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["time", "gen", "sigma", "games", "parent_obj", "child_obj",
                          "confirm_diff", "margin", "accepted", "child_rates", "parent_rates"])
        for gen in range(1, args.gens + 1):
            gen_seed = args.seed * 1000 + gen
            children = [{k: parent[k] + sigma * rng.gauss(0, 1) for k in tunable}
                        for _ in range(args.lam)]
            jobs = [parent] + children   # parent shares this generation's seeds
            rates = list(executor.map(
                lambda c: pool_rates(weights_of(c), pool, args.games, gen_seed, args), jobs))
            objs = [objective(r) for r in rates]
            j = max(range(1, len(jobs)), key=objs.__getitem__)

            # Confirm the best child against the parent on fresh seeds.
            confirm_seed = 100000 + args.seed * 1000 + gen
            child_r, parent_r = executor.map(
                lambda c: pool_rates(weights_of(c), pool, args.confirm_games, confirm_seed, args),
                [jobs[j], parent])
            diff = objective(child_r) - objective(parent_r)
            margin = args.z * se_diff(args.confirm_games, len(pool))
            accepted = diff > margin
            log.writerow([int(time.time()), gen, round(sigma, 3), args.games,
                          round(objs[0], 4), round(objs[j], 4), round(diff, 4),
                          round(margin, 4), accepted,
                          [round(x, 3) for x in child_r], [round(x, 3) for x in parent_r]])
            logf.flush()
            print(f"gen {gen}: parent {objs[0]:.3f} best child {objs[j]:.3f} | confirm diff "
                  f"{diff:+.3f} vs margin {margin:.3f} | accepted={accepted} sigma={sigma:.2f}",
                  flush=True)
            if accepted:
                parent = jobs[j]
                with open(args.out, "w") as f:
                    json.dump(weights_of(parent), f, indent=1)
            else:
                sigma *= 0.85
    print("done. Validate my_team/weights.json on 1000 fresh games, including vs F, before submitting.")


if __name__ == "__main__":
    main()
