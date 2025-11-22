# evrp/costs.py
from __future__ import annotations
from typing import List
from .data import Problem

def calculate_travel_cost(solution: List[List[int]], problem: Problem) -> float:
    D = problem.distance_matrix
    total = 0.0
    for route in solution:
        for i in range(len(route) - 1):
            total += D[route[i]][route[i+1]]
    return total

def full_cost(solution, problem, ul_only: bool = False):
    """
    Compute total cost of a solution.
    If ul_only=True, only the upper-level (distance-based) cost is computed
    without solving the lower-level (energy/charging) problem.
    """
    # --- UL distance computation ---
    D = getattr(problem, "distance_matrix", None)
    if D is None:
        raise ValueError("Problem instance missing distance matrix.")

    base_distance = 0.0
    for route in solution:
        for i in range(len(route) - 1):
            base_distance += D[route[i]][route[i + 1]]

    # --- Early return if only UL cost is needed ---
    if ul_only:
        return base_distance

    # --- Full hierarchical cost (UL + LL) ---
    from evrp.heuristics import solve_ll_exact
    ok, ll_cost = solve_ll_exact(solution, problem)
    if not ok:
        return float("inf")

    return base_distance + ll_cost

# Back-compat

