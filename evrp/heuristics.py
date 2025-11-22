# heuristics.py
from __future__ import annotations

from math import isfinite
import random
from typing import List, Tuple, Optional

from evrp.costs import full_cost
from evrp.operators import apply_ul_operator
from evrp.elite import update_elite_archive, cluster_elite_archive
from .cluster import embed_solution, nearest_centroid_idx, sqdist
from .solution import quick_repair

try:
    from evrp.lower_level import solve_ll_exact  # adjust the path to your project
except Exception:
    pass
from math import isfinite

from math import isfinite


# ---------------------------
# Centroid / embedding helpers
# ---------------------------

def _ensure_centroids(elite, centroids, rng: Optional[Random] = None):  # <-- default
    if centroids:
        return centroids, False
    # build fresh centroids using the SAME rng
    return cluster_elite_archive(elite, rng=rng), True


def find_nearest_centroid(solution: List[List[int]], centroids: List[List[float]], problem) -> int:
    """
    Embed solution and return index of nearest centroid in embedding space.
    Returns -1 if centroids is empty.
    """
    if not centroids:
        return -1
    dim = getattr(problem, "embed_dim", 256)
    v = embed_solution(solution, dim)
    return nearest_centroid_idx(v, centroids)


# ---------------------------
# Unified Lower-level (energy) solver
# ---------------------------

def solve_ll(sol_or_route, problem, rng: Optional[random.Random] = None, return_trace: bool = False):
    """
    Lower-Level EVRP solver (charge-to-full) that injects charging stations directly
    into the route. Returns:
        (ok: bool, ll_solution_with_stations, ll_cost: float, trace)
    """
    # Precompute frequently accessed attributes
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = getattr(problem, "energy_consumption", 1.0)
    stations = tuple(problem.stations or ())
    ev_range_km = BMAX / alpha

    # Precompute station attributes with defaults
    detour_km = getattr(problem, "station_detour_km", {}) or {}
    price_map = getattr(problem, "station_energy_price", {}) or {}
    wait_cost = getattr(problem, "station_wait_cost", {}) or {}
    price_def = getattr(problem, "energy_cost", 0.0)
    wait_def = getattr(problem, "waiting_cost", 0.0)

    K = getattr(problem, "k_nearest_stations", 5)
    init_soc = (getattr(problem, "init_soc_ratio", 1.0) or 1.0) * BMAX

    # Cache for station candidates with distance precomputation
    _cand_cache = {}

    def candidate_stations(i: int, j: int) -> List[int]:
        """Get feasible stations between i and j with caching."""
        if i not in _cand_cache:
            # Pre-sort stations by distance + detour
            _cand_cache[i] = sorted(stations,
                                    key=lambda b: D[i][b] + detour_km.get(b, 0.0))[:K]

        # Filter stations that can reach destination j
        max_dist_to_j = ev_range_km
        return [b for b in _cand_cache[i] if D[b][j] <= max_dist_to_j]

    def _one_route(route: List[int]):
        """Solve charging for a single route."""
        if len(route) < 2:
            result = (True, route, 0.0, [] if return_trace else None)
            return result

        soc = init_soc
        ll_cost = 0.0
        legs_trace = [] if return_trace else None
        route_with_stations = [route[0]]  # start at depot

        for t in range(len(route) - 1):
            i, j = route[t], route[t + 1]
            need_direct = alpha * D[i][j]

            # Try direct drive first (common case optimization)
            if soc >= need_direct:
                soc -= need_direct
                route_with_stations.append(j)
                if return_trace:
                    legs_trace.append({"i": i, "j": j, "stop": None, "cost": 0.0})
                else:
                    pass


            # Need charging - find best station
            best_cost = float("inf")
            best_b = None

            for b in candidate_stations(i, j):
                # Check if we can reach station
                detour_b = detour_km.get(b, 0.0)
                need_ib = alpha * (D[i][b] + detour_b)
                if need_ib > soc:
                    continue

                # Check if station can reach destination
                if BMAX < alpha * D[b][j] - 1e-9:
                    continue

                # Calculate cost
                soc_arr_b = soc - need_ib
                energy_to_full = BMAX - soc_arr_b  # Always positive due to need_ib check
                wbk = wait_cost.get(b, wait_def)
                rbk = price_map.get(b, price_def)
                cand_cost = D[i][b] + detour_b + wbk + rbk * energy_to_full

                if cand_cost < best_cost:
                    best_cost, best_b = cand_cost, b

            if best_b is None:
                # No feasible station found
                return False, route, float("inf"), [] if return_trace else None

            # Apply charging stop
            detour_best = detour_km.get(best_b, 0.0)
            soc -= alpha * (D[i][best_b] + detour_best)  # Travel to station
            soc = BMAX  # Charge to full
            soc -= alpha * D[best_b][j]  # Travel to destination
            ll_cost += best_cost

            route_with_stations.extend([best_b, j])

            if return_trace:
                legs_trace.append({"i": i, "j": j, "stop": best_b, "cost": best_cost})

            # Validate SOC bounds
            if soc < -1e-9 or soc > BMAX + 1e-9:
                return False, route, float("inf"), [] if return_trace else None

        return True, route_with_stations, ll_cost, legs_trace

    # --- Main execution ---
    is_multi_route = sol_or_route and isinstance(sol_or_route[0], list)

    if is_multi_route:
        total_cost = 0.0
        ll_solution = []
        all_traces = [] if return_trace else None

        for route in sol_or_route:
            ok, route_ll, cost, trace = _one_route(route)
            if not ok:
                return False, sol_or_route, float("inf"), [] if return_trace else []

            ll_solution.append(route_ll)
            total_cost += cost
            if return_trace:
                all_traces.append(trace)

        return True, ll_solution, total_cost, all_traces
    else:
        ok, route_ll, cost, trace = _one_route(sol_or_route)
        return ok, route_ll, cost, (trace if return_trace else [])
def solve_ll_exact(sol_or_route, problem, rng: Optional[random.Random] = None):
    """Stable API: return exactly (ok: bool, ll_cost: float)."""

    ok, _same_input, ll_cost, _trace = solve_ll(sol_or_route, problem, rng, return_trace=False)
    return bool(ok), float(ll_cost)

def _solve_ll_exact3(solution, problem, rng=None):
    """
    Normalize various return shapes from solve_ll / solve_ll_exact
    to a strict 3-tuple:
      (ok: bool, ll_sol, ll_cost: float)
    """
    try:
        res = solve_ll_exact(solution, problem, rng)
    except TypeError:
        res = solve_ll_exact(solution, problem)

    # Case 1: already the right shape (ok, sol, cost)
    if isinstance(res, tuple) and len(res) == 3:
        ok, sol, cost = res
        return bool(ok), sol, float(cost)

    # Case 2: (ok, cost) → from your current solve_ll_exact
    if isinstance(res, tuple) and len(res) == 2:
        ok, cost = res
        return bool(ok), solution if ok else None, float(cost)

    # Case 3: raw cost
    if isinstance(res, (int, float)):
        cost = float(res)
        return isfinite(cost), solution if isfinite(cost) else None, cost

    # Fallback → failure
    return False, None, float("inf")


def solve_ll_with_trace(sol_or_route, problem, rng=None):
    """
    Strict interface for trace variant:
    Returns (ok: bool, sol, cost: float, trace: list)
    """
    ok, sol, cost, trace = solve_ll(sol_or_route, problem, rng, return_trace=True)

    return bool(ok), sol, float(cost), trace
# ---------------------------
# Cheap UL feasibility screen
# ---------------------------
#hello
def is_promising_ul(sol_or_route, problem) -> bool:
    """
    Necessary (not sufficient) UL pre-check:
      For every arc i->j, require either:
        - D[i][j] <= EV max leg (full battery), OR
        - exists station b with D[i][b] (+detour) <= max leg AND D[b][j] <= max leg.
    """
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = getattr(problem, "energy_consumption", 0.0) or 1e-9
    max_leg_km = BMAX / alpha

    stations = list(problem.stations or [])
    detour = getattr(problem, "station_detour_km", {}) or {}

    def arc_ok(i: int, j: int) -> bool:
        if D[i][j] <= max_leg_km:
            return True
        for b in stations:
            if (D[i][b] + detour.get(b, 0.0) <= max_leg_km) and (D[b][j] <= max_leg_km):
                return True
        return False

    if sol_or_route and isinstance(sol_or_route[0], list):
        for route in sol_or_route:
            for t in range(len(route) - 1):
                if not arc_ok(route[t], route[t + 1]):
                    return False
        return True
    else:
        route = sol_or_route
        for t in range(len(route) - 1):
            if not arc_ok(route[t], route[t + 1]):
                return False
        return True


# ---------------------------
# UL Heuristics (actions)
# ---------------------------

def heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng):
    """
    H1 – Full Hierarchical:
    Solve LL exactly for each UL solution, align with nearest centroid,
    and apply a UL operator. Full exploitation mode.
    """
    c_parent = full_cost(parent, problem)  # Evaluate full cost once
    centroids, _ = _ensure_centroids(elite, centroids, rng)

    start = parent
    if centroids:
        ci = find_nearest_centroid(parent, centroids, problem)
        if ci >= 0:
            target = centroids[ci]
            best_e, best_d = None, float("inf")
            for cost, sol, emb in elite:
                d = sqdist(emb, target)
                if d < best_d:
                    best_d, best_e = d, sol
            if best_e is not None:
                start = best_e

    child = apply_ul_operator(start, problem, rng)
    return quick_repair(child, problem)


def heuristic_h2_selective_ll(parent, elite, problem, rng):
    """
    H2 – Selective LL Evaluation:
    Evaluate LL only for promising ULs, then apply UL operator for exploration.
    """
    if is_promising_ul(parent, problem):
        ok, ll_sol, ll_cost = _solve_ll_exact3(parent, problem, rng)
        if ok:
            update_elite_archive(
                elite, ll_sol, full_cost(ll_sol, problem),
                dim=getattr(problem, "embed_dim", 256),
                max_size=getattr(problem, "elite_max", 120)
            )

    child = apply_ul_operator(parent, problem, rng)
    return quick_repair(child, problem)


def heuristic_h3_relaxed_ll(parent, problem, rng):
    """
    H3 – Relaxed LL:
    Skip LL solving; perform quick UL exploration only.
    """
    child = apply_ul_operator(parent, problem, rng)
    return quick_repair(child, problem)


def heuristic_h4_similarity_based(parent, centroids, elite, problem, rng):
    """
    H4 – Similarity-based:
    Start from the elite solution closest to the parent’s nearest centroid.
    """
    centroids, _ = _ensure_centroids(elite, centroids, rng)
    if not centroids:
        child = apply_ul_operator(parent, problem, rng)
        return quick_repair(child, problem)

    ci = find_nearest_centroid(parent, centroids, problem)
    if ci < 0:
        child = apply_ul_operator(parent, problem, rng)
        return quick_repair(child, problem)

    target = centroids[ci]
    best_sol, best_d = None, float("inf")
    for cost, sol, emb in elite:
        d = sqdist(emb, target)
        if d < best_d:
            best_d, best_sol = d, sol

    start = best_sol if best_sol is not None else parent
    child = apply_ul_operator(start, problem, rng)
    return quick_repair(child, problem)
#ok, sol, ll_cost, trace = heuristics.solve_ll(solution, problem, return_trace=True)
def get_used_stations(ll_solution, problem):
    """Extract all stations used in the solution"""
    if ll_solution and isinstance(ll_solution[0], list):
        # Multiple routes
        stations = []
        for route in ll_solution:
            stations.extend([node for node in route if node in problem.stations])
        return stations
    else:
        # Single route
        return [node for node in ll_solution if node in problem.stations]