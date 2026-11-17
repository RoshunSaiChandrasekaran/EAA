import numpy as np
from ioh import get_problem, ProblemClass
from GA import GA

TOTAL_BUDGET = 100_000          # max evaluations for the WHOLE tuning (both problems)
PROBLEMS = [(18, 50), (23, 49)]  # (fid, dimension)
REF = {18: 8.17, 23: 7.0}       # reference values used to normalise (best known / optimum)

space = {
    "pop_size":          [10, 20, 30, 50, 100, 200],
    "mutation_strength": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "crossover_rate":    [0.0, 0.3, 0.6, 0.9, 1.0],
    "tournament_size":   [2, 3, 5, 8],
}

used = 0  # global evaluation counter


def run_once(fid, dim, hp, evals, seed):
    """One GA run on one problem (no logger). Returns best f normalised by the optimum."""
    global used
    p = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    GA(p, max_evals=evals, seed=seed, **hp)
    used += p.state.evaluations
    return p.state.current_best.y / REF[fid]


def score(hp, evals, n_runs, rng):
    """Mean normalised performance over both problems -> a config must be good on BOTH."""
    per_problem = []
    for fid, dim in PROBLEMS:
        vals = [run_once(fid, dim, hp, evals, int(rng.integers(1_000_000))) for _ in range(n_runs)]
        per_problem.append(np.mean(vals))
    return float(np.mean(per_problem)), per_problem


def tune_hyperparameters(seed=1):
    rng = np.random.default_rng(seed)
    # Successive halving: (number of configs kept, runs per problem, evals per run)
    configs = [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(12)]
    stages = [(4, 1, 1500), (2, 2, 2500), (1, 1, 5000)]

    for keep, n_runs, evals in stages:
        results = []
        for hp in configs:
            s, per = score(hp, evals, n_runs, rng)
            results.append((s, hp))
            print(f"[{used:6d} evals used] score={s:.4f} per-problem={np.round(per, 4)} {hp}")
        results.sort(key=lambda t: -t[0])
        configs = [hp for _, hp in results[:keep]]
        print(f"--> kept {keep}, best so far: {configs[0]}\n")
    assert used <= TOTAL_BUDGET, f"budget exceeded: {used}"
    print(f"Total evaluations used: {used}")
    return configs[0]


if __name__ == "__main__":
    best = tune_hyperparameters()
    print("BEST:", best)
