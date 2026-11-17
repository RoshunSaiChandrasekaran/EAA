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

    # --- main loop ---
    while evals < max_evals and not problem.state.optimum_found:
        best = np.argmax(fit)
        new_pop = [pop[best].copy()]          # elitism: keep best individual
        new_fit = [fit[best]]
        while len(new_pop) < pop_size and evals < max_evals:
            p1, p2 = tournament(), tournament()
            if rng.random() < crossover_rate:   # uniform crossover
                mask = rng.random(n) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            flips = rng.random(n) < p_mut       # bit-flip mutation
            child = np.where(flips, 1 - child, child)
            new_pop.append(child)
            new_fit.append(evaluate(child))
        pop = np.array(new_pop)
        fit = np.array(new_fit)
    # no return value needed


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
    # NOTE: this uses the default hyperparameters. After tuning, put the tuned values here.
    HP = dict(pop_size=100, mutation_strength=1.0, crossover_rate=0.9, tournament_size=3)

    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20):
        GA(F18, seed=SEED + run, **HP)
        F18.reset()
    _logger.close()

    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20):
        GA(F23, seed=SEED + run, **HP)
        F23.reset()
    _logger.close()
