import numpy as np
from ioh import get_problem, ProblemClass
from GA import GA

TOTAL_BUDGET = 100_000
PROBLEMS = [(18, 50), (23, 49)]          # (fid, dimension)
REF = {18: 8.17, 23: 7.0}                # normalisation references

FIXED = dict(crossover_type="uniform", survivor="plus", force_flip=True)

space = {
    "pop_size":          [20, 30, 50, 75, 100, 150],
    "mutation_type":     ["standard", "heavy"],
    "mutation_strength": [0.5, 1.0, 1.5, 2.0, 3.0],   # used by "standard"
    "beta":              [1.5, 2.0, 2.5, 3.0],        # used by "heavy"
    "crossover_rate":    [0.0, 0.3, 0.6, 0.9, 1.0],
    "tournament_size":   [2, 3, 5, 8],
}

used = 0  # global evaluation counter


def run_once(fid, dim, hp, evals, seed):
    """One GA run on one problem, no logger. Score clipped to [0, 1]."""
    global used
    p = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    GA(p, max_evals=evals, seed=seed, **FIXED, **hp)
    used += p.state.evaluations
    return min(max(p.state.current_best.y / REF[fid], 0.0), 1.0)


def score(hp, evals, n_runs, rng):
    """Mean clipped score over both problems: a config must be good on BOTH."""
    per_problem = []
    for fid, dim in PROBLEMS:
        vals = [run_once(fid, dim, hp, evals, int(rng.integers(1_000_000)))
                for _ in range(n_runs)]
        per_problem.append(np.mean(vals))
    return float(np.mean(per_problem)), per_problem


def tune_hyperparameters(seed=1):
    rng = np.random.default_rng(seed)
    configs = [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(11)]
    configs.append(dict(pop_size=100, mutation_type="standard", mutation_strength=1.0,
                        beta=2.0, crossover_rate=0.6, tournament_size=8))   # prior-knowledge candidate
    # (configs kept, runs per problem, evals per run)
    stages = [(2, 1, 2000), (1, 2, 5000)]

    for keep, n_runs, evals in stages:
        results = []
        for hp in configs:
            s, per = score(hp, evals, n_runs, rng)
            results.append((s, hp))
            print(f"[{used:6d} used] score={s:.4f} per-problem={np.round(per, 3)} {hp}")
        results.sort(key=lambda t: -t[0])
        configs = [hp for _, hp in results[:keep]]
        print(f"--> kept {keep}, best so far: {configs[0]}\n")

    assert used <= TOTAL_BUDGET, f"budget exceeded: {used}"
    print(f"Total evaluations used: {used}")
    return {**FIXED, **configs[0]}


if __name__ == "__main__":
    best = tune_hyperparameters()
    print("BEST:", best)