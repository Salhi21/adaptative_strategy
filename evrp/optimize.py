import math
import random
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

import numpy as  np

from evrp.data import Problem
from evrp.solution import generate_initial_solution, quick_repair
from evrp.costs import full_cost
from . import heuristics
from .elite import update_elite_archive, cluster_elite_archive
from .operators import _vnd_with_sa
from .q_learning import get_best_action, update as q_update, decay_epsilon

ACTIONS = ["H1", "H2", "H3", "H4"]


def _safe_cost(sol, problem: Problem) -> float:
    """Compute solution cost robustly (inf on NaN/None)."""
    c = full_cost(sol, problem)
    if c is None or isinstance(c, float) and math.isnan(c):
        return float("inf")
    return c


def initialize_algorithm(problem: Problem, pop_size: int, rng):
    """
    Initialize population, elite archive, and clustering structures.
    Used for metric-driven hyper-heuristic.
    """
    # --- 1) Initial population ---
    P = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        P.append(sol)

    # --- 2) Elite and centroids ---
    elite = []       # [(cost, sol, embedding)]
    centroids = []

    # --- 3) Global best ---
    best_cost = float("inf")
    best_sol = None

    return P, elite, centroids, best_cost, best_sol
# === Metrics ===
def population_diversity(population: list[np.ndarray]) -> float:
    """Average pairwise Euclidean diversity."""
    N = len(population)
    if N < 2:
        return 0.0
    total_dist = 0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_dist += np.linalg.norm(np.array(population[i]) - np.array(population[j]))
            count += 1
    return total_dist / count


def population_convergence(fitnesses: list[float]) -> float:
    """Convergence metric based on fitness spread (std/mean)."""
    f_mean = np.mean(fitnesses)
    f_std = np.std(fitnesses)
    return f_std / (abs(f_mean) + 1e-9)


def fitness_improvement_rate(best_hist: list[float]) -> float:
    """Relative improvement of best fitness over time."""
    if len(best_hist) < 2:
        return 1.0
    return abs(best_hist[-1] - best_hist[-2]) / (abs(best_hist[-2]) + 1e-9)


# === Main optimization ===
def main_optimization_metrics(problem: Problem, cfg: SimpleNamespace, rng: random.Random):
    """
    Adaptive hyper-heuristic for bi-level optimization (aligned with framework diagram).
    """

    # === Initialization ===
    (P, elite, centroids, best_c, best_s) = initialize_algorithm(problem, cfg.pop_size, rng)

    # 1. Evaluate initial population (solve LL for each)
    costs_P = [full_cost(s, problem) for s in P]
    fitness = [1.0 / (1.0 + c) if math.isfinite(c) else 0.0 for c in costs_P]
    best_idx = min(range(len(costs_P)), key=lambda i: costs_P[i])
    best_c, best_s = costs_P[best_idx], P[best_idx]
    best_history = [best_c]

    # Scientific thresholds
    conv_threshold = getattr(cfg, "conv_threshold", 0.08)
    div_threshold  = getattr(cfg, "div_threshold", 0.10)
    alpha_thresh   = getattr(cfg, "alpha", 0.01)
    term_thresh    = getattr(cfg, "term_threshold", 1e-4)

    # === Main optimization loop ===
    for gen in range(cfg.max_gens):
        # 2. Upper-level selection (tournament)
        M = []
        tsize = max(1, min(getattr(cfg, "tournament_size", 2), len(P)))
        for _ in range(len(P)):
            idxs = rng.sample(range(len(P)), tsize)
            winner = max(idxs, key=lambda i: fitness[i])
            M.append(P[winner])

        # 3. Apply upper-level perturbation to generate offspring Q_t
        if getattr(cfg, "use_local_search", True):
            M = [_vnd_with_sa(parent, problem, rng) for parent in M]

        # 4. Compute convergence metrics for Q_t (after perturbation)
        costs_M = [full_cost(s, problem) for s in M]
        fitness_M = [1.0 / (1.0 + c) if math.isfinite(c) else 0.0 for c in costs_M]
        div = population_diversity([np.array(flatten_solution(s)) for s in M])
        conv = population_convergence(fitness_M)
        delta_fit = fitness_improvement_rate(best_history)

        converged = conv < conv_threshold
        weak_div  = div < div_threshold

        # 5. Decide heuristic according to convergence/diversity tree
        if not converged:
            action = "H1"
        else:
            if weak_div:
                if delta_fit < alpha_thresh:
                    action = "H2"
                else:
                    action = "H4"
            else:
                action = "H3"

        # 6. Apply selected heuristic
        P_new = []
        for parent in M:
            if action == "H1":
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)
                # Cluster + archive update only for H1
                update_elite_archive(elite, child, full_cost(child, problem))
                centroids = cluster_elite_archive(elite)

            elif action == "H2":
                child = heuristics.heuristic_h2_selective_ll(parent, elite, problem, rng)
            elif action == "H3":
                child = heuristics.heuristic_h3_relaxed_ll(parent, problem, rng)
            elif action == "H4" and elite and centroids:
                child = heuristics.heuristic_h4_similarity_based(parent, centroids, elite, problem, rng)
            else:
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)

            child = quick_repair(child, problem)
            P_new.append(child)

        # 7. Evaluate offspring
        costs_new = [full_cost(c, problem) for c in P_new]

        # 8. Survivor selection (μ + λ)
        combined = P + P_new
        combined_costs = costs_P + costs_new
        order = sorted(range(len(combined)), key=lambda i: combined_costs[i])
        P = [combined[i] for i in order[:cfg.pop_size]]

        # Update metrics and best solution
        costs_P = [full_cost(s, problem) for s in P]
        fitness = [1.0 / (1.0 + c) if math.isfinite(c) else 0.0 for c in costs_P]
        best_idx = min(range(len(costs_P)), key=lambda i: costs_P[i])
        best_c, best_s = costs_P[best_idx], P[best_idx]
        best_history.append(best_c)

        # 9. Conditional post-heuristic perturbation
        if weak_div and delta_fit < alpha_thresh:
            print("[INFO] Diversity collapsed → applying post-heuristic perturbation")
            P = [_vnd_with_sa(sol, problem, rng) for sol in P]

        # 10. Logging and termination
        bc = f"{best_c:.2f}" if math.isfinite(best_c) else "inf"
        print(f"[gen {gen:03d}] best={bc} div={div:.3f} conv={conv:.3f} Δf={delta_fit:.4f} act={action}")

        if delta_fit < term_thresh:
            print(f">>> Early convergence detected at generation {gen}.")
            break

    return best_s, best_c
def flatten_solution(sol):
    """Flatten multi-route solution for metric computation."""
    vec = []
    if isinstance(sol, dict):
        for routes in sol.values():
            for route in routes:
                vec.extend(route)
    return vec[:50]  # truncate for normalization