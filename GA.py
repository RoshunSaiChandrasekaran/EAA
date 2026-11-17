from typing import Tuple
import numpy as np
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000   # evaluations per run in the final evaluation
SEED = 42       # fixed seed -> reproducible results


def GA(problem: ioh.problem.PBO,
       pop_size: int = 100,
       mutation_strength: float = 1.0,   # expected number of bit flips per offspring
       crossover_rate: float = 0.9,      # probability that two parents are recombined
       tournament_size: int = 3,
       crossover_type: str = "uniform",
       survivor: str = "generational",    # "generational" or "plus"
       force_flip: bool = False,          # guarantee at least 1 bit flip per child
       mutation_type: str = "standard",   # "standard" or "heavy"
       beta: float = 1.5,                 # power-law exponent for heavy mutation
       max_evals: int = budget,
       seed: int = None) -> None:
    """Baseline bitstring GA (maximisation). Generational, 1 elite, uniform crossover,
    tournament selection, bit-flip mutation with p = mutation_strength / n."""
    rng = np.random.default_rng(seed)
    n = problem.meta_data.n_variables
    p_mut = min(1.0, mutation_strength / n)
    evals = 0

    def evaluate(x):
        nonlocal evals
        evals += 1
        return problem(x.tolist())

    # --- initial population (random) ---
    pop = rng.integers(0, 2, size=(pop_size, n))
    fit = np.full(pop_size, -np.inf)
    for i in range(pop_size):
        if evals >= max_evals:
            return
        fit[i] = evaluate(pop[i])

    def tournament():
        idx = rng.integers(0, pop_size, size=tournament_size)
        return pop[idx[np.argmax(fit[idx])]]

    ks = np.arange(1, n // 2 + 1)
    k_probs = ks ** (-beta)
    k_probs /= k_probs.sum()

    # --- main loop ---
    while evals < max_evals and not problem.state.optimum_found:
        if survivor == "plus":
            n_off = pop_size                       # produce pop_size offspring
            off, off_fit = [], []
        else:
            best = np.argmax(fit)
            off, off_fit = [pop[best].copy()], [fit[best]]   # elitism
            n_off = pop_size
        while len(off) < n_off and evals < max_evals:
            p1, p2 = tournament(), tournament()
            if rng.random() < crossover_rate:
                if crossover_type == "two_point":
                    a, b = np.sort(rng.choice(n + 1, size=2, replace=False))
                    idx = np.arange(n)
                    mask = (idx >= a) & (idx < b)
                else:
                    mask = rng.random(n) < 0.5
                child = np.where(mask, p2, p1)
            else:
                child = p1.copy()
            if mutation_type == "heavy":
                k = rng.choice(ks, p=k_probs)               # number of bits to flip
                flips = np.zeros(n, dtype=bool)
                flips[rng.choice(n, size=k, replace=False)] = True
            else:
                flips = rng.random(n) < p_mut
                if force_flip and not flips.any():
                    flips[rng.integers(n)] = True
            child = np.where(flips, 1 - child, child)
            off.append(child)
            off_fit.append(evaluate(child))
        off, off_fit = np.array(off), np.array(off_fit)
        if survivor == "plus":
            all_pop = np.vstack([pop, off])
            all_fit = np.concatenate([fit, off_fit])
            keep = np.argsort(-all_fit)[:pop_size]  # best pop_size of parents+offspring
            pop, fit = all_pop[keep], all_fit[keep]
        else:
            pop, fit = off, off_fit
    


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
    l = logger.Analyzer(
        root="data",
        folder_name="run",
        algorithm_name="genetic_algorithm",
        algorithm_info="Practical assignment of the EA course",
    )
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    #HP = dict(pop_size=100, mutation_strength=1.0, crossover_rate=0.6, tournament_size=8, crossover_type="uniform", survivor="plus")
    #HP = dict(pop_size=10, mutation_strength=1.0, crossover_rate=0.6, tournament_size=3, crossover_type="uniform", survivor="plus")
    #HP = dict(pop_size=100, mutation_strength=1.0, crossover_rate=0.6, tournament_size=8,crossover_type="uniform", survivor="plus", force_flip=True)
    #HP = dict(pop_size=100, mutation_strength=1.0, crossover_rate=0.6, tournament_size=8,crossover_type="uniform", survivor="plus", force_flip=True,mutation_type="heavy", beta=2.5)
    #HP = dict(pop_size=20, mutation_strength=1.5, crossover_rate=0.0, tournament_size=8,crossover_type="uniform", survivor="plus", force_flip=True,mutation_type="standard")
    HP = dict(pop_size=30, crossover_rate=0.9, tournament_size=5,
          crossover_type="uniform", survivor="plus", force_flip=True,
          mutation_type="heavy", beta=2.0)
    print("Hyperparameters:", HP)

    F18, _logger = create_problem(dimension=50, fid=18)
    results_f18 = []
    for run in range(20):
        GA(F18, seed=SEED + run, **HP)
        results_f18.append(F18.state.current_best.y)
        F18.reset()
    _logger.close()
    print("F18 mean/std/max:", np.mean(results_f18), np.std(results_f18), np.max(results_f18))

    F23, _logger = create_problem(dimension=49, fid=23)
    results_f23 = []
    for run in range(20):
        GA(F23, seed=SEED + run, **HP)
        results_f23.append(F23.state.current_best.y)
        F23.reset()
    _logger.close()
    print("F23 mean/std/max:", np.mean(results_f23), np.std(results_f23), np.max(results_f23))
