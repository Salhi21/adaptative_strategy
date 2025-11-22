from typing import List
from .data import Problem

def clone_solution(sol): return [r[:] for r in sol]
def hash_solution(sol) -> int: return hash(tuple(tuple(r) for r in sol))

def generate_initial_solution(problem: Problem) -> List[List[int]]:
    import random
    customers = (problem.customers or [])[:]
    random.shuffle(customers)
    routes = [[] for _ in range(problem.vehicles)]
    for i, c in enumerate(customers):
        routes[i % problem.vehicles].append(c)
    return [[problem.depot] + r + [problem.depot] for r in routes]

def quick_repair(solution, problem: Problem):
    """Only enforce depot at start/end. No station insertion."""
    repaired = []
    for r in solution:
        route = r[:]
        if not route or route[0] != problem.depot:
            route = [problem.depot] + route
        if route[-1] != problem.depot:
            route = route + [problem.depot]
        repaired.append(route)
    return repaired
