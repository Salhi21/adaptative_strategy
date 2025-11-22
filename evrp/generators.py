# evrp/generators.py
import random

def decorate_with_pevrp_params(problem, seed=42, charge_rate_kW=None):
    """Populate missing per-station fields + global pEVRP defaults."""
    rng = random.Random(seed)

    # Global defaults
    problem.speed = 50.0
    problem.energy_capacity = 100.0           # kWh
    problem.energy_consumption = 1/6.0        # kWh/km (gamma=6)
    problem.init_soc_ratio = 1.0
    problem.fixed_charge_time_h = getattr(problem, "fixed_charge_time_h", 0.5)  # 30 min
    problem.capacity = getattr(problem, "capacity", 1000)
    problem.waiting_cost = getattr(problem, "waiting_cost", 0.0)  # $/h; set >0 to monetize time
    problem.energy_cost = getattr(problem, "energy_cost", 0.0)    # fallback $/kWh
    problem.charge_rate = getattr(problem, "charge_rate", None)   # fallback kW

    # Ensure the dict attributes exist
    for attr in ("station_charge_rate","station_energy_price","station_wait_time",
                 "station_wait_cost","station_detour_km"):
        if not hasattr(problem, attr) or getattr(problem, attr) is None:
            setattr(problem, attr, {})

    base_rate = charge_rate_kW if charge_rate_kW is not None else (problem.energy_capacity / 0.5)  # full in 30 min
    for b in (problem.stations or []):
        problem.station_charge_rate.setdefault(b, max(1.0, rng.gauss(base_rate, 0.05*base_rate)))
        problem.station_energy_price.setdefault(b, max(0.0, rng.gauss(0.134, 0.02)))  # $/kWh
        problem.station_wait_time.setdefault(b,   max(0.0, rng.gauss(1.0, 0.1)))      # hours per visit
        problem.station_detour_km.setdefault(b,   max(0.0, rng.gauss(10.0, 1.0)))     # km extra
        # optional per-visit lump-sum cost (if you want it)
        problem.station_wait_cost.setdefault(b, 0.0)

    return problem
