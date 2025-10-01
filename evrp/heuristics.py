# heuristics.py
from __future__ import annotations

from math import isfinite
import random
from typing import List, Tuple, Optional

from evrp.costs import full_cost
from evrp.operators import apply_ul_operator
from evrp.elite import update_elite_archive, cluster_elite_archive
from .cluster import embed_solution, nearest_centroid_idx, sqdist
try:
    from evrp.lower_level import solve_ll_exact  # adjust the path to your project
except Exception:
    pass
from math import isfinite

from math import isfinite

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
    Unified LL feasibility & cost under 'charge-to-full'.

    Returns a consistent 4-tuple:
      (ok: bool, same_input, ll_cost: float, trace)
    where `trace` is:
      - [] when return_trace=False
      - For a route: list[{"i","j","stop","cost"}] per leg
      - For a solution: list[route_trace]
    """
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = getattr(problem, "energy_consumption", 0.0) or 1e-9  # avoid division by zero
    stations = tuple(problem.stations or ())
    ev_range_km = BMAX / alpha

    # Optional maps
    detour_km = getattr(problem, "station_detour_km", {}) or {}       # extra distance per station b
    price_map = getattr(problem, "station_energy_price", {}) or {}    # $/kWh per station b
    wait_cost = getattr(problem, "station_wait_cost", {}) or {}       # $/visit per station b
    price_def = getattr(problem, "energy_cost", 0.0) or 0.0
    wait_def = getattr(problem, "waiting_cost", 0.0) or 0.0

    # Prune: consider top-K nearest stations from node i
    K = getattr(problem, "k_nearest_stations", 5)
    _cand_cache = {}  # i -> [b1, b2, ...]

    def candidate_stations(i: int, j: int) -> List[int]:
        lst = _cand_cache.get(i)
        if lst is None:
            lst = sorted(stations, key=lambda b: D[i][b] + detour_km.get(b, 0.0))[:K]
            _cand_cache[i] = lst
        # from full at b, must reach j
        return [b for b in lst if D[b][j] <= ev_range_km]

    def _one_route(route: List[int]):
        if not route or len(route) < 2:
            return True, 0.0, [] if return_trace else None

        soc = (getattr(problem, "init_soc_ratio", 1.0) or 1.0) * BMAX
        ll_cost = 0.0
        legs_trace = [] if return_trace else None

        for t in range(len(route) - 1):
            i, j = route[t], route[t + 1]
            need_direct = alpha * D[i][j]

            # Direct drive if enough SoC
            if soc >= need_direct:
                soc -= need_direct
                if return_trace:
                    legs_trace.append({"i": i, "j": j, "stop": None, "cost": 0.0})
                continue

            # Otherwise pick cheapest feasible station (charge-to-full)
            best_cost = float("inf"); best_b = None
            for b in candidate_stations(i, j):
                need_ib = alpha * (D[i][b] + detour_km.get(b, 0.0))
                if need_ib > soc:
                    continue
                soc_arr_b = soc - need_ib
                energy_to_full = max(0.0, BMAX - soc_arr_b)
                # from full at b, must reach j
                if BMAX - alpha * D[b][j] < -1e-9:
                    continue

                detour_term = D[i][b] + detour_km.get(b, 0.0)
                wbk = wait_cost.get(b, wait_def)
                rbk = price_map.get(b, price_def)
                cand = detour_term + wbk + rbk * energy_to_full
                if cand < best_cost:
                    best_cost, best_b = cand, b

            if best_b is None:
                # infeasible leg
                return False, float("inf"), [] if return_trace else None

            # Apply stop at best_b: i->b (consume), charge to full, b->j (consume)
            soc -= alpha * (D[i][best_b] + detour_km.get(best_b, 0.0))
            soc = BMAX
            soc -= alpha * D[best_b][j]
            ll_cost += best_cost

            if return_trace:
                legs_trace.append({"i": i, "j": j, "stop": best_b, "cost": best_cost})

            if soc < -1e-9 or soc > BMAX + 1e-9:
                return False, float("inf"), [] if return_trace else None

        return True, ll_cost, legs_trace

    # Whole solution vs single route
    if sol_or_route and isinstance(sol_or_route[0], list):
        total = 0.0
        all_traces = [] if return_trace else None
        for r in sol_or_route:
            ok, c, tr = _one_route(r)
            if not ok:
                return False, sol_or_route, float("inf"), [] if return_trace else []
            total += c
            if return_trace:
                all_traces.append(tr)
        return True, sol_or_route, total, all_traces
    else:
        ok, c, tr = _one_route(sol_or_route)
        return ok, sol_or_route, c, (tr if return_trace else [])


# --- Backward-compat wrappers (keep your existing imports working) ---

def solve_ll_exact(sol_or_route, problem, rng: Optional[random.Random] = None):
    """Stable API: return exactly (ok: bool, ll_cost: float)."""
    ok, _same_input, ll_cost, _trace = solve_ll(sol_or_route, problem, rng, return_trace=False)
    return bool(ok), float(ll_cost)

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
    H1: Full hierarchical.
      1) Add parent to elite with its cost & embedding.
      2) Ensure centroids exist (compute here if missing).
      3) Choose the elite solution whose embedding is closest to the nearest centroid of parent.
      4) Mutate and return child.
    """
    c_parent = full_cost(parent, problem)  # UL distance + LL cost (inf if infeasible)
    update_elite_archive(
        elite, parent, c_parent,
        dim=getattr(problem, "embed_dim", 256),
        max_size=getattr(problem, "elite_max", 120)
    )

    centroids, _ = _ensure_centroids(elite, centroids, rng)   # <-- PASS RNG

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

    return apply_ul_operator(start, problem, rng)


def heuristic_h2_selective_ll(parent, elite, problem, rng: random.Random = random):
    """
    H2: Selective LL evaluation.
      - If UL-necessary checks pass, run LL; if feasible, store in elite.
      - Always return a UL-perturbed child to keep exploration going.
    """
    if is_promising_ul(parent, problem):
        ok, ll_sol, ll_cost = _solve_ll_exact3(parent, problem, rng)
        if ok:
            update_elite_archive(
                elite, ll_sol, full_cost(ll_sol, problem),
                dim=getattr(problem, "embed_dim", 256),
                max_size=getattr(problem, "elite_max", 120)
            )
    return apply_ul_operator(parent, problem, rng)


def heuristic_h3_relaxed_ll(parent, problem, rng: random.Random = random):
    """
    H3: Relaxed / no LL checks (fast structural exploration).
    """
    return apply_ul_operator(parent, problem, rng)


def heuristic_h4_similarity_based(parent, centroids, elite, problem, rng: random.Random = random):
    """
    H4: Similarity-based start.
      - Ensure centroids (compute here if missing).
      - Start from the elite solution closest to the parent's nearest centroid, then mutate.
      - Fallback to parent if no elite/centroids.
    """
    centroids, _ = _ensure_centroids(elite, centroids, rng)  # <-- pass rng

    if not centroids:
        return apply_ul_operator(parent, problem, rng)

    ci = find_nearest_centroid(parent, centroids, problem)
    if ci < 0:
        return apply_ul_operator(parent, problem, rng)

    target = centroids[ci]
    best_sol, best_d = None, float("inf")
    for cost, sol, emb in elite:
        d = sqdist(emb, target)
        if d < best_d:
            best_d, best_sol = d, sol

    start = best_sol if best_sol is not None else parent
    return apply_ul_operator(start, problem, rng)
#ok, sol, ll_cost, trace = heuristics.solve_ll(solution, problem, return_trace=True)
