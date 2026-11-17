# EA Practical Assignment Part I: 

Problems: **F18 (LABS, n=50)** and **F23 (N-Queens, 7x7 = 49 bits)**, both maximisation.
Final evaluation: 5,000 function evaluations per run, 20 independent runs per problem.
Tuning budget: 100,000 evaluations in total, one shared hyperparameter setting for both problems.
**Final deadline: Dec 3, 23:59** (optional feedback submission: Nov 19).

## 1. Files

| File | Purpose |
|---|---|
| `GA.py` | The GA, `create_problem`, and the final 20-run experiment in `__main__` |
| `tuning.py` | Hyperparameter tuning (successive halving). Calls `GA` |
| `data/run/` | IOH logs written by `python GA.py`. Zip this folder for IOHanalyzer. **Delete `data/` before every final run**, otherwise old experiments end up in the logs |

Setup: `pip install ioh numpy`. Run `python tuning.py` first, then paste the best values into `HP` in `GA.py`'s `__main__` block and run `python GA.py`.

Placeholders still to do at the end: rename the function `GA` to `studentnumber1_studentnumber2_GA`, rename the files as the assignment requires, and keep the template's `tune_hyperparameters` call working (it currently returns a dict instead of 3 values).

## 2. Interface contract (keep this, so tuning works with any version)

```python
GA(problem, pop_size, mutation_strength, crossover_rate, tournament_size,
   crossover_type, survivor, force_flip, mutation_type, beta,
   max_evals=5000, seed=None)
```

- `mutation_strength` is the **expected number of bit flips per child** (per-bit probability is `mutation_strength / n`), so it means the same thing on every problem size.
- The GA keeps its **own evaluation counter** and stops at `max_evals`. Tuning passes smaller values, the final run uses 5000.
- Reproducibility: `GA.py` seeds run `r` with `SEED + r` (`SEED = 42`), and `tuning.py` uses `default_rng(1)`. Do not remove the seeds. If the numbers in the report do not match what the graders get, the grade is 0.

## 3. What the GA does (current version)

Generational GA on a bitstring: random initial population, tournament selection, optional crossover (uniform or two-point), mutation (standard bit-flip or heavy-tailed), survivor selection either generational with 1 elite or (mu+lambda) (`"plus"`), and optionally at least one forced bit flip per child (`force_flip`). It stops early if `problem.state.optimum_found`.

## 4. Experiments done so far (design phase, same 20 seeds for each)

Fixed setting for the comparisons: pop 100, mutation_strength 1.0, crossover_rate 0.6, tournament 8.

| # | Change | F18 mean ± std | F23 mean ± std | Verdict |
|---|---|---|---|---|
| 0 | Baseline: uniform crossover, generational, 1 elite | 3.93 ± 0.36 | 6.15 ± 0.73 | reference |
| 1 | Two-point crossover | 3.86 ± 0.42 | 6.15 ± 0.57 | no difference |
| 2 | (mu+lambda) survivor selection | 3.77 ± 0.32 | 6.45 ± 0.59 | inconclusive (F23 up, F18 down) |
| 3a | `plus`, pop 20, tournament 3 | 3.89 ± 0.42 | 6.10 ± 0.44 | |
| 3b | `plus`, pop 10, tournament 3 | 4.02 ± 0.38 | 5.50 ± 0.50 | small population: F18 slightly up, F23 clearly down |
| 4 | **Force at least 1 bit flip** (on top of `plus`) | **4.33 ± 0.39** | **6.65 ± 0.48** | **clear gain (about +0.56 on F18)** |
| 5a | Heavy-tailed mutation, beta 1.5 | 4.36 ± 0.48 | 6.50 ± 0.50 | no significant difference |
| 5b | Heavy-tailed mutation, beta 2.5 | 4.29 ± 0.47 | 6.85 ± 0.36 | no significant difference |

Rule of thumb: with 20 runs the standard error of a mean is about 0.1 (F18) and 0.1 to 0.15 (F23), so differences below about 0.2 are noise.

Why `force_flip` helped: with per-bit probability 1/n, about 36% of children get zero flips, and without crossover they are exact copies of a parent. Every copy wastes one evaluation of a 5,000 budget.

## 5. Tuning

`tuning.py` samples 11 random configurations plus one prior-knowledge candidate, then runs successive halving: stage 1 is 12 configs x 2 problems x 1 run x 2,000 evals, stage 2 is the top 2 x 2 problems x 2 runs x 5,000 evals. The score is the mean of the clipped (0 to 1) normalised best value per problem: F18 / 8.17 (best known for n=50, because IOH reports `inf` as the optimum) and F23 / 7. A global counter plus an `assert` guard the 100,000 limit. Structure fixed by the design experiments: `survivor="plus"`, `force_flip=True`, `crossover_type="uniform"`.

Tuned results, final 20 runs at 5,000 evaluations:

| Setting | F18 | F23 |
|---|---|---|
| Hand-picked (pop 100, tournament 8, crossover 0.6): **not a tuning result, do not submit as tuned** | 4.33 ± 0.39 | 6.65 ± 0.48 |
| Tuning run 1 (older script, 16 configs, stage 2 at 2,500 evals): pop 20, mutation only, tournament 8 | 4.11 ± 0.39 | 6.65 ± 0.48 |
| **Tuning run 2 (current script): pop 30, crossover 0.9, tournament 5, heavy mutation beta 2.0** | **4.14 ± 0.41** | **6.75 ± 0.43** |

Current submission candidate: `HP = dict(pop_size=30, crossover_rate=0.9, tournament_size=5, crossover_type="uniform", survivor="plus", force_flip=True, mutation_type="heavy", beta=2.0)`.

Known limitations (worth discussing in the report):
- The tuning objective is noisy. F23 per run is almost always 0.57, 0.86 or 1.0, so one run barely separates configurations. The hand-picked setting was eliminated in stage 1 on a single run.
- Stage 1 uses fewer than 5,000 evaluations, which favours small populations.
- Only 12 configurations were evaluated. About 18,000 evaluations were unused.
- All three F18 results above are statistically tied, so the plain GA appears to plateau around 4 to 4.4 on F18 at this budget. Only `force_flip` made a measurable difference.

## 6. Tasks for the teammate

### Task A: Option B, permutation encoding for F23 (main task)
Goal: see whether a problem-aware encoding beats the bitstring on F23 (current: 6.75 ± 0.43, optimum 7).

1. Add a flag, e.g. `encoding="bitstring"` or `"permutation"`. F18 must **keep using the bitstring GA**, and the hyperparameters must stay shared across problems.
2. Individual = a permutation of `0..6` (queen in row `i` is in column `perm[i]`). Decode to the 49-bit board before calling `problem(x)`: `x = np.zeros(49); x[i*7 + perm[i]] = 1`. Compute 7 as `int(np.sqrt(n))`. Every individual then already has one queen per row and column, so only diagonal conflicts remain.
3. Operators: swap mutation (use `mutation_strength` as the expected number of swaps) and a permutation crossover (order crossover or PMX). Tournament selection, `plus` survivor selection and `force_flip` (at least one swap) can be reused.
4. Check how IOH scores F23 on a decoded board (`problem(x)` on a known valid solution, and on an invalid one) before trusting any result.
5. Compare against the current F23 numbers using the **same 20 seeds** and the same HP. Then re-run `tuning.py` for the new version, because the best hyperparameters will probably differ. The report should compare two *tuned* algorithms.
6. The evaluation counter works the same way: one `problem(x)` call is one evaluation. Never compute fitness yourself to avoid counting.

### Task B : improving F18
F18 is the weak spot (mean about 4.1 against a best known 8.17, and typical good optimizers reach about 7). Ideas, in rough order of effort:
- **Memetic local search on the elite only**: spend a small part of the budget on first-improvement 1-bit-flip local search applied to the best individual. Every neighbour costs one evaluation, so keep it to a few percent of the budget.
- **Restarts or population re-initialisation** if the best value has not improved for K generations.
- **Symmetry-aware moves**: LABS fitness is invariant under reversing the sequence, flipping all bits, and flipping every other bit. Crossing "equivalent but differently oriented" parents may be destructive.
- Skew-symmetric sequences only apply to odd n, so they do not apply to n = 50.

**Do not** compute LABS energy or neighbour fitness offline to rank moves without calling `problem()`. It bypasses the evaluation budget, so it is probably not allowed. Ask the TA before doing anything like it.

Success criterion: F18 mean clearly above the current 4.14 (more than about 0.3), measured with the same 20 seeds and then **re-tuned** with `tuning.py`. Stop after the time box either way, and report a negative result honestly.

## 7. Open questions for the TA
1. Is the 100,000-evaluation tuning budget the total over both problems, or per problem? We assumed the total.
2. Is it fine for `tune_hyperparameters` to return a dict (we tune more than 3 hyperparameters)?
3. Are short, manual design comparisons at fixed hyperparameters allowed, as long as the submitted values come out of the tuning procedure?
