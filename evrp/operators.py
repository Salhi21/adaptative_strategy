import math, random
from .solution import clone_solution
from .costs import full_cost

def _valid_customer_pos(route):
    # positions 1..len-2 (exclude depots)
    return range(1, max(1, len(route)-1))

def _two_opt_once(sol, problem, rng):
    for r_idx, route in enumerate(sol):
        n = len(route)
        if n < 4: continue
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                cand = clone_solution(sol)
                cand[r_idx] = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                yield cand

def _relocate_once(sol, problem, rng):
    R = len(sol)
    for a in range(R):
        ra = sol[a]
        if len(ra) < 3: continue
        for i in _valid_customer_pos(ra):
            node = ra[i]
            for b in range(R):
                rb = sol[b]
                for j in range(1, len(rb)):  # insert before last depot
                    if a == b and (j == i or j == i+1): continue
                    cand = clone_solution(sol)
                    cand[a].pop(i)
                    cand[b].insert(j, node)
                    yield cand

def _swap_once(sol, problem, rng):
    R = len(sol)
    for a in range(R):
        ra = sol[a]
        if len(ra) < 3: continue
        for b in range(a, R):
            rb = sol[b]
            if len(rb) < 3: continue
            for i in _valid_customer_pos(ra):
                for j in _valid_customer_pos(rb):
                    cand = clone_solution(sol)
                    cand[a][i], cand[b][j] = cand[b][j], cand[a][i]
                    yield cand

def _accept(old_cost, new_cost, T, rng):
    if new_cost <= old_cost: return True
    return T > 1e-12 and (rng.random() < math.exp(-(new_cost-old_cost)/T))

def _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2):
    current = clone_solution(parent)
    cur_cost = full_cost(current, problem)
    neighborhoods = (_two_opt_once, _relocate_once, _swap_once)

    T = T0
    for _ in range(max_passes):
        improved = False
        for gen in neighborhoods:
            best_local, best_cost = current, cur_cost
            for cand in gen(current, problem, rng):
                c = full_cost(cand, problem)
                if c < best_cost or _accept(cur_cost, c, T, rng):
                    best_local, best_cost = cand, c
            if best_cost < cur_cost:
                current, cur_cost = best_local, best_cost
                improved = True
            T *= 0.8
        if not improved: break
    return current

def apply_ul_operator(parent, problem, rng=random, n_candidates: int = 8):
    return _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2)

def apply_ul_operator_guided(parent, ll_hint_cost, problem, rng=random, n_candidates: int = 8):
    return _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2)
