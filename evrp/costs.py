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

def full_cost(solution: List[List[int]], problem: Problem) -> float:
    # ✅ Lazy import prevents costs <-> heuristics circular dependency
    from .heuristics import solve_ll_exact  # must be snake_case

    base = calculate_travel_cost(solution, problem)
    ok, ll_cost = solve_ll_exact(solution, problem)  # -> (bool, float)
    if not ok:
        return float("inf")
    return base + ll_cost

# Back-compat

