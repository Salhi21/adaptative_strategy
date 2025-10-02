from evrp.costs import calculate_travel_cost, full_cost
import argparse, random
from types import SimpleNamespace
from evrp.data import load_evrp, apply_defaults
from evrp.optimize import main_optimization
from evrp.heuristics import solve_ll_with_trace, solve_ll

def describe_solution(problem, sol):
    print("Routes:")
    for r in sol:
        print("  ", " -> ".join(map(str, r)))
    print("Travel distance:", calculate_travel_cost(sol, problem))
    print("Total cost     :", full_cost(sol, problem))

def print_routes_with_recharges(problem, sol):
    ok, _, _, traces = solve_ll(sol, problem, return_trace=True)
    print(f"LL feasible: {ok}")
    for r_idx, (route, tr) in enumerate(zip(sol, traces), start=1):
        tokens = [str(route[0])]  # start at depot
        for leg in tr:
            i, j, b = leg["i"], leg["j"], leg["stop"]
            if b is not None:
                # show station stop inline: (station, next-customer)
                tokens.append(f"({b},{j})")
            else:
                tokens.append(str(j))
        print(f"R{r_idx}: " + " -> ".join(tokens))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="Path to .evrp instance file")
    ap.add_argument("--max-gens", type=int, default=200)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--tournament-size", type=int, default=2)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-min", type=float, default=0.1)
    ap.add_argument("--decay", type=float, default=0.995)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--waiting-cost", type=float, default=None, help="$/hour to monetize time (optional)")
    ap.add_argument("--energy-cost", type=float, default=None, help="fallback $/kWh (optional)")
    ap.add_argument("--charge-rate", type=float, default=None, help="fallback kW if a station lacks a rate (optional)")
    ap.add_argument("--speed", type=float, default=None, help="vehicle speed km/h (optional)")
    args = ap.parse_args()

    try:
        instance_path = resolve_instance_path(args.instance)
    except NameError:
        instance_path = args.instance

    print(f"Loading instance: {instance_path}")
    problem = load_evrp(instance_path)
    problem = apply_defaults(problem)

    # Global constants (stations from data; no decoration/randomization)
    problem.energy_capacity = 100.0  # Bmax
    problem.energy_consumption = 1/6  # alpha (kWh/km)
    problem.init_soc_ratio = 1.0
    # Waiting cost per recharge event (wbk) — use per-station map or fallback:
    problem.waiting_cost = 5.0  # used as per-visit default w_bk
    problem.energy_cost = 4.22  # $/kWh default r_bk

    # Allow CLI overrides
    if args.waiting_cost is not None:
        problem.waiting_cost = args.waiting_cost
    if args.energy_cost is not None:
        problem.energy_cost = args.energy_cost
    if args.charge_rate is not None:
        problem.charge_rate = args.charge_rate
    if args.speed is not None:
        problem.speed = args.speed

    cfg = SimpleNamespace(
        max_gens=args.max_gens,
        pop_size=args.pop,
        tournament_size=args.tournament_size,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        decay=args.decay,
        alpha=args.alpha,
        gamma=args.gamma,
    )

    rng = random.Random(args.seed)
    best_sol, best_overall_cost = main_optimization(problem, cfg, rng)

    print("=== DONE ===")
    print("Best cost:", best_overall_cost)
    if best_sol:
        print("Best solution routes (node sequences):")
        for i, r in enumerate(best_sol, start=1):
            print(f"  R{i} ({len(r)} nodes): " + " -> ".join(map(str, r)))
    print("Best solution with recharge stops inline:")
    print_routes_with_recharges(problem, best_sol)


if __name__ == "__main__":
    main()
