import math
import random
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

from evrp.data import Problem
from evrp.solution import generate_initial_solution, quick_repair
from evrp.costs import full_cost
from . import heuristics
from .elite import update_elite_archive, cluster_elite_archive
from .q_learning import get_best_action, update as q_update, decay_epsilon

ACTIONS = ["H1", "H2", "H3", "H4"]


def _safe_cost(sol, problem: Problem) -> float:
    """Compute solution cost robustly (inf on NaN/None)."""
    c = full_cost(sol, problem)
    if c is None or isinstance(c, float) and math.isnan(c):
        return float("inf")
    return c


def initialize_algorithm(problem: Problem, pop_size: int, eps_start: float, rng):
    """
    Initialize population, elite archive, centroids, and Q-learning structures.
    Used for cost-minimization Q-learning hyper-heuristic.
    """
    # --- 1) Initial population generation ---
    P = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        P.append(sol)

    # --- 2) Elite & centroids ---
    elite = []       # stores tuples (cost, sol, emb)
    centroids = []

    # --- 3) Global best placeholders ---
    best_cost = float("inf")
    best_sol = None

    # --- 4) Q-learning setup ---
    Q = {}
    state = (0, "start")
    eps = eps_start

    # Initialize Q-values optimistically high (since we minimize cost)
    for a in ACTIONS:
        Q[(state, a)] = 1e6  # large initial cost; encourages exploration

    return P, elite, centroids, best_cost, best_sol, Q, state, eps

def main_optimization(problem: Problem, cfg: SimpleNamespace, rng: random.Random):
    """
    Q-learning–driven hyper-heuristic EA for (E)VRP.
    Expects cfg to provide:
      - pop_size, max_gens, tournament_size
      - alpha, gamma
      - eps_start, eps_min, decay
    """
    (
        P,
        elite,
        centroids,
        best_c,
        best_s,
        Q,
        state,
        eps,
    ) = initialize_algorithm(problem, cfg.pop_size, cfg.eps_start, rng)

    for gen in range(cfg.max_gens):
        # ---- 1) Fitness (inverse cost, robust to inf) ----
        costs_P = [full_cost(s, problem) for s in P]
        fitness = [1.0 / (1.0 + c) if math.isfinite(c) else 0.0 for c in costs_P]

        # ---- 2) Mating pool (tournament) ----
        M: List[Any] = []
        tsize = max(1, min(getattr(cfg, "tournament_size", 2), len(P)))
        for _ in range(len(P)):
            idxs = rng.sample(range(len(P)), tsize)
            winner_idx = max(idxs, key=lambda i: fitness[i])
            M.append(P[winner_idx])

        # ---- 3) Choose action (ε-greedy) ----
        action = rng.choice(ACTIONS) if rng.random() < eps else get_best_action(Q, state, ACTIONS)

        # ---- 4) Generate offspring using selected heuristic ----
        P_new: List[Any] = []
        use_h4 = action == "H4" and bool(elite) and len(centroids) > 0

        for parent in M:
            if action == "H1":
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)
            elif action == "H2":
                child = heuristics.heuristic_h2_selective_ll(parent, elite, problem, rng)
            elif action == "H3":
                child = heuristics.heuristic_h3_relaxed_ll(parent, problem, rng)
            elif use_h4:
                child = heuristics.heuristic_h4_similarity_based(parent, centroids, elite, problem, rng)
            else:
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)

            child = quick_repair(child, problem)
            P_new.append(child)

        # ---- 5) Evaluate offspring costs ----
        costs_new = []
        for c in P_new:
            if action in ("H3", "H4"):
                c_cost = full_cost(c, problem, ul_only=True)
            else:
                c_cost = full_cost(c, problem)
            costs_new.append(c_cost)

        # ---- 6) Best offspring ----
        if all(math.isinf(c) for c in costs_new):
            best_off_cost = float("inf")
            best_off = P_new[0]
        else:
            best_idx = min(range(len(P_new)), key=lambda i: costs_new[i])
            best_off = P_new[best_idx]
            best_off_cost = costs_new[best_idx]

        improved = best_off_cost < best_c
        if improved:
            best_c, best_s = best_off_cost, best_off
            update_elite_archive(elite, best_off, best_off_cost)
            centroids = cluster_elite_archive(elite)

        # ---- 7) Q-learning update (cost-based) ----
        # Normalize cost for stable learning
        if not math.isfinite(best_off_cost) or best_off_cost <= 0:
            norm_cost = 1.0  # neutral cost
        elif not math.isfinite(best_c) or best_c <= 0:
            norm_cost = best_off_cost
        else:
            norm_cost = min(best_off_cost / (1.0 + best_c), 1e6)  # clamp huge ratios

        next_state = (1 if improved else 0, action)
        q_update(Q, state, action, norm_cost, next_state, cfg.alpha, cfg.gamma, ACTIONS)

        # ---- 8) Survivor selection (μ+λ) ----
        combined = P + P_new
        combined_costs = costs_P + costs_new
        order = sorted(range(len(combined)), key=lambda i: combined_costs[i])
        P = [combined[i] for i in order[: cfg.pop_size]]

        # ---- 9) Epsilon decay ----
        eps = max(cfg.eps_min, decay_epsilon(eps, cfg.eps_min, cfg.decay))
        state = next_state

    # ---- 10) Logging ----
    bc = f"{best_c:.2f}" if math.isfinite(best_c) else "inf"
    print(f"[gen {gen}] best={bc} eps={eps:.3f} act={action}")

    # Print Q-values for all states (aggregated)
    snapshot = {}
    for (s, a), val in Q.items():
        if s not in snapshot:
            snapshot[s] = {}
        snapshot[s][a] = round(val, 3)

    for s, qs in snapshot.items():
        print(f"  state={s} -> {qs}")

    return best_s, best_c

