# data.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from math import sqrt


# =========================
# Problem definition
# =========================
@dataclass
class Problem:
    # Basic instance info
    name: str = "unnamed"
    vehicles: int = 1
    capacity: int = 1000
    depot: int = 1
    customers: List[int] = field(default_factory=list)
    stations: List[int] = field(default_factory=list)

    # Time windows — not used (kept for future compatibility)
    has_time_windows: bool = False
    ready_time: Dict[int, float] = field(default_factory=dict)
    due_time: Dict[int, float] = field(default_factory=dict)
    service_duration: Dict[int, float] = field(default_factory=dict)

    # Geometry / distances (1-based indexing; coords[0] is dummy)
    coords: List[Tuple[float, float]] = field(default_factory=list)
    distance_matrix: List[List[float]] = field(default_factory=list)

    # Energy model
    energy_capacity: float = 100.0     # B_max (kWh) — from file
    energy_consumption: float = 1.0    # kWh/km (fixed per project spec)
    init_soc_ratio: float = 1.0        # α0 (start SoC as fraction of B_max)
    speed: Optional[float] = None      # km/h (unused unless you add time)

    # Costs & charging
    waiting_cost: float = 5.0          # $/hour (fixed per project spec)
    energy_cost: float = 4.22          # $/kWh  (fixed per project spec)
    charge_rate: Optional[float] = None  # kW (global fallback; station may override)
    fixed_charge_time_h: float = 0.5   # 30 minutes overhead per recharge (optional)

    # Demands (capacity). Defaults to zero unless a DEMAND section is added later.
    demands: Dict[int, int] = field(default_factory=dict)

    # Per-station optional overrides (all optional; use if your instances provide them)
    station_charge_rate: Dict[int, float] = field(default_factory=dict)   # kW
    station_energy_price: Dict[int, float] = field(default_factory=dict)  # $/kWh
    station_wait_time: Dict[int, float] = field(default_factory=dict)     # hours per visit
    station_wait_cost: Dict[int, float] = field(default_factory=dict)     # $ per visit
    station_detour_km: Dict[int, float] = field(default_factory=dict)     # extra km on arrival

    # Convenience
    @property
    def n(self) -> int:
        """Total number of nodes (depot + customers + stations)."""
        return len(self.coords) - 1


# =========================
# Helpers
# =========================
def _euc2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Build a 1-based dense distance matrix (index 0 row/col left as zeros).
    """
    n = len(coords) - 1
    D = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ci = coords[i]
        for j in range(1, n + 1):
            if i != j:
                D[i][j] = _euc2d(ci, coords[j])
    return D


# =========================
# Loader for EVRP instances
# =========================
def load_evrp(path: str) -> Problem:
    """
    Load an EVRP instance.

    Expected header keys:
      NAME, VEHICLES, DIMENSION, STATIONS, CAPACITY, ENERGY_CAPACITY,
      (ENERGY_CONSUMPTION is ignored and forced to 1.0 per project spec),
      followed by NODE_COORD_SECTION and coordinates lines: "idx x y".

    Assumptions:
      - Depot index = 1.
      - Stations are the last `STATIONS` nodes.
      - Customers are the nodes between depot and stations.
      - Coordinates are 1-based indexed in the file; we maintain 1-based arrays.
    """
    name: str = ""
    vehicles: Optional[int] = None
    capacity: Optional[int] = None
    dimension: Optional[int] = None
    stations_cnt: Optional[int] = None
    energy_capacity: Optional[float] = None

    coords: List[Tuple[float, float]] = [(-1.0, -1.0)]  # pad index 0 (unused)
    reading_coords = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if reading_coords:
                # Stop when the section ends (next header or blank)
                if line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 3:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        if idx >= len(coords):
                            coords.extend([(0.0, 0.0)] * (idx - len(coords) + 1))
                        coords[idx] = (x, y)
                    continue
                else:
                    reading_coords = False
                    # fall through to parse potential next headers

            # Header parsing
            if line.startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("VEHICLES:"):
                vehicles = int(line.split(":", 1)[1].strip())
            elif line.startswith("DIMENSION:"):
                dimension = int(line.split(":", 1)[1].strip())
            elif line.startswith("STATIONS:"):
                stations_cnt = int(line.split(":", 1)[1].strip())
            elif line.startswith("CAPACITY:"):
                capacity = int(line.split(":", 1)[1].strip())
            elif line.startswith("ENERGY_CAPACITY:"):
                energy_capacity = float(line.split(":", 1)[1].strip())
            elif line.startswith("ENERGY_CONSUMPTION:"):
                # Present in some files, but we override to 1.0 (project spec).
                _ = float(line.split(":", 1)[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

    # Validations
    if dimension is None or stations_cnt is None:
        raise ValueError("Missing DIMENSION or STATIONS in instance header.")
    if vehicles is None or capacity is None or energy_capacity is None:
        raise ValueError("Missing VEHICLES, CAPACITY, or ENERGY_CAPACITY in instance header.")
    if len(coords) - 1 != dimension:
        raise ValueError(f"Expected {dimension} coordinates, found {len(coords) - 1}.")

    depot = 1
    num_customers = dimension - stations_cnt - 1
    if num_customers < 0:
        raise ValueError("Computed negative number of customers. Check STATIONS and DIMENSION.")

    customers = list(range(2, 2 + num_customers))  # nodes immediately after depot
    stations = list(range(dimension - stations_cnt + 1, dimension + 1))  # last nodes are stations

    problem = Problem(
        name=name,
        vehicles=vehicles,
        capacity=capacity,
        depot=depot,
        customers=customers,
        stations=stations,
        coords=coords,
        energy_capacity=energy_capacity,
        # Fixed values (per project spec):
        energy_consumption=1.0,
        waiting_cost=5.0,
        energy_cost=4.22,
    )

    # Precompute distances
    problem.distance_matrix = build_distance_matrix(problem.coords)

    # Default zero demands unless you add a DEMAND section later
    problem.demands = {i: 0 for i in range(1, problem.n + 1)}

    return problem


# =========================
# Optional: runtime overrides
# =========================
def apply_defaults(
    problem: Problem,
    *,
    charge_rate: Optional[float] = None,
    energy_cost: Optional[float] = None,
    waiting_cost: Optional[float] = None,
    speed: Optional[float] = None,
) -> Problem:
    """
    Apply runtime overrides. If not provided, keep current values.
    Note: energy_consumption is intentionally fixed at 1.0 by design.
    """
    if charge_rate is not None:
        problem.charge_rate = charge_rate
    if energy_cost is not None:
        problem.energy_cost = energy_cost
    if waiting_cost is not None:
        problem.waiting_cost = waiting_cost
    if speed is not None:
        problem.speed = speed
    return problem
