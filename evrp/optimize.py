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
    P = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        P.append(sol)

    elite = []          # <-- was {} ; make it a list
    centroids = []
    best_cost = float('inf'); best_sol = None
    Q = {}; state = (0, 'start'); eps = eps_start
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
        # ---- 1) Fitness (with per-generation cost cache) ----
        costs_P = [_safe_cost(s, problem) for s in P]
        fitness = [1.0 / (1.0 + max(0.0, c)) for c in costs_P]  # robust if cost==inf

        # ---- 2) Mating pool (tournament, clamped) ----
        M: List[Any] = []
        tsize = max(1, min(getattr(cfg, "tournament_size", 2), len(P)))
        for _ in range(len(P)):
            idxs = rng.sample(range(len(P)), tsize)
            winner_idx = max(idxs, key=lambda i: fitness[i])
            M.append(P[winner_idx])

        # ---- 3) Choose action (ε-greedy) ----
        if rng.random() < eps:
            action = rng.choice(ACTIONS)
        else:
            action = get_best_action(Q, state, ACTIONS)

        # ---- 4) Generate offspring via selected heuristic (+repair) ----
        P_new: List[Any] = []
        # Use cached centroids when calling H4; if elite empty, fallback to H1.
        use_h4 = action == "H4" and bool(elite) and len(centroids) > 0

        for parent in M:
            if action == "H1":
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)
            elif action == "H2":
                child = heuristics.heuristic_h2_selective_ll(parent, elite, problem, rng)
            elif action == "H3":
                child = heuristics.heuristic_h3_relaxed_ll(parent, problem, rng)
            elif use_h4:
                # IMPORTANT: pass cached centroids; do NOT recluster inside H4
                child = heuristics.heuristic_h4_similarity_based(parent, centroids, elite, problem, rng)
            else:
                # Guard: if H4 chosen but no elite/centroids yet, fallback to H1
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, problem, rng)

            child = quick_repair(child, problem)
            P_new.append(child)

        # ---- 5) Best offspring of this generation ----
        costs_new = [_safe_cost(s, problem) for s in P_new]
        # handle all-inf edge case safely
        if all(math.isinf(c) for c in costs_new):
            best_off_cost = float("inf")
            best_off = P_new[0]
        else:
            best_idx = min(range(len(P_new)), key=lambda i: costs_new[i])
            best_off = P_new[best_idx]
            best_off_cost = costs_new[best_idx]

        # ---- 6) Reward & elite update ----
        prev_best = best_c
        # Positive reward only when improving the global best
        reward = max(0.0, prev_best - best_off_cost)

        improved = best_off_cost < best_c
        if improved:
            best_c, best_s = best_off_cost, best_off
            update_elite_archive(elite, best_off, best_off_cost)
            # Recompute centroids ONCE per generation after elite update
            centroids = cluster_elite_archive(elite)

        # ---- 7) Q-learning update ----
        next_state = (1 if improved else 0, action)
        q_update(Q, state, action, reward, next_state, cfg.alpha, cfg.gamma, ACTIONS)

        # ---- 8) Survivor selection (μ+λ) ----
        combined = P + P_new
        combined_costs = costs_P + costs_new  # aligns with combined order
        # Sort by cost safely
        order = sorted(range(len(combined)), key=lambda i: combined_costs[i])
        P = [combined[i] for i in order[: cfg.pop_size]]

        # ---- 9) Epsilon decay (once per generation, clamped) ----
        eps = decay_epsilon(eps, cfg.eps_min, cfg.decay)
        eps = max(cfg.eps_min, eps)  # ensure clamp
        state = next_state

        if gen % 50 == 0:
            # defensive formatting if best_c is inf
            bc = f"{best_c:.2f}" if not math.isinf(best_c) else "inf"
            print(f"[gen {gen}] best={bc} eps={eps:.3f} act={action}")

    return best_s, best_c
