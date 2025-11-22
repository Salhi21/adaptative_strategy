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
    energy_capacity: float = 100.0  # B_max (kWh) — from file
    energy_consumption: float = 1.0  # kWh/km (fixed per project spec)
    init_soc_ratio: float = 1.0  # α0 (start SoC as fraction of B_max)

    # Costs & charging
    waiting_cost: float = 5.0  # $/hour (fixed per project spec)
    energy_cost: float = 4.22  # $/kWh  (fixed per project spec)
    charge_rate: Optional[float] = None  # kW (global fallback; station may override)
    fixed_charge_time_h: float = 0.5  # 30 minutes overhead per recharge (optional)

    # Demands (capacity). Defaults to zero unless a DEMAND section is added later.
    demands: Dict[int, int] = field(default_factory=dict)

    # Per-station optional overrides (all optional; use if your instances provide them)
    station_charge_rate: Dict[int, float] = field(default_factory=dict)  # kW
    station_energy_price: Dict[int, float] = field(default_factory=dict)  # $/kWh
    station_wait_time: Dict[int, float] = field(default_factory=dict)  # hours per visit
    station_wait_cost: Dict[int, float] = field(default_factory=dict)  # $ per visit
    station_detour_km: Dict[int, float] = field(default_factory=dict)  # extra km on arrival

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

    Expected sections:
      - Header: NAME, VEHICLES, DIMENSION, STATIONS, CAPACITY, ENERGY_CAPACITY
      - NODE_COORD_SECTION: node coordinates
      - DEMAND_SECTION: customer demands
      - STATIONS_COORD_SECTION: station node IDs
      - DEPOT_SECTION: depot node ID

    Note: ENERGY_CONSUMPTION is ignored and forced to 1.0 per project spec.
    """
    # Header values
    name: str = ""
    vehicles: Optional[int] = None
    capacity: Optional[int] = None
    dimension: Optional[int] = None
    stations_cnt: Optional[int] = None
    energy_capacity: Optional[float] = None

    # Data structures
    coords: List[Tuple[float, float]] = [(-1.0, -1.0)]  # pad index 0 (unused)
    demands: Dict[int, int] = {}
    station_nodes: List[int] = []
    depot_node: int = 1  # Default

    # Parsing state
    reading_coords = False
    reading_demands = False
    reading_stations = False
    reading_depot = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # Skip empty lines
            if not line:
                continue

            # End of file
            if line == "EOF":
                break

            # ============ SECTION HEADERS ============
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                reading_demands = False
                reading_stations = False
                reading_depot = False
                continue

            elif line.startswith("DEMAND_SECTION"):
                reading_coords = False
                reading_demands = True
                reading_stations = False
                reading_depot = False
                continue

            elif line.startswith("STATIONS_COORD_SECTION"):
                reading_coords = False
                reading_demands = False
                reading_stations = True
                reading_depot = False
                continue

            elif line.startswith("DEPOT_SECTION"):
                reading_coords = False
                reading_demands = False
                reading_stations = False
                reading_depot = True
                continue

            # ============ HEADER PARSING ============
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
                # Present in some files, but we override to 1.0 (project spec)
                _ = float(line.split(":", 1)[1].strip())

            # ============ DATA SECTIONS ============
            elif reading_coords:
                # Parse: node_id x y
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])

                    # Extend coords list if necessary
                    if idx >= len(coords):
                        coords.extend([(0.0, 0.0)] * (idx - len(coords) + 1))
                    coords[idx] = (x, y)

            elif reading_demands:
                # Parse: node_id demand
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands[node_id] = demand

            elif reading_stations:
                # Parse: station_node_id (one per line until DEPOT_SECTION or -1)
                if line.startswith("DEPOT_SECTION"):
                    reading_stations = False
                    reading_depot = True
                    continue

                if line.lstrip('-').isdigit():
                    station_id = int(line)
                    if station_id == -1:
                        reading_stations = False
                    elif station_id > 0:
                        station_nodes.append(station_id)

            elif reading_depot:
                # Parse: depot_node_id (usually 1, then -1 to end)
                if line.lstrip('-').isdigit():
                    depot_id = int(line)
                    if depot_id > 0:
                        depot_node = depot_id
                    elif depot_id == -1:
                        reading_depot = False  # End of depot section

    # ============ VALIDATIONS ============
    if dimension is None or stations_cnt is None:
        raise ValueError("Missing DIMENSION or STATIONS in instance header.")
    if vehicles is None or capacity is None or energy_capacity is None:
        raise ValueError("Missing VEHICLES, CAPACITY, or ENERGY_CAPACITY in instance header.")
    if len(coords) - 1 != dimension:
        raise ValueError(f"Expected {dimension} coordinates, found {len(coords) - 1}.")

    # ============ BUILD PROBLEM ============
    # Identify customers: all nodes except depot and stations
    all_nodes = set(range(1, dimension + 1))
    station_set = set(station_nodes)
    customers = sorted(all_nodes - station_set - {depot_node})

    problem = Problem(
        name=name,
        vehicles=vehicles,
        capacity=capacity,
        depot=depot_node,
        customers=customers,
        stations=station_nodes,
        coords=coords,
        energy_capacity=energy_capacity,
        # Fixed values (per project spec):
        energy_consumption=1.0,
        waiting_cost=5.0,
        energy_cost=4.22,
    )

    # Precompute distances
    problem.distance_matrix = build_distance_matrix(problem.coords)

    # Set demands (default to 0 for depot and stations)
    problem.demands = demands
    if depot_node not in problem.demands:
        problem.demands[depot_node] = 0
    for station in station_nodes:
        if station not in problem.demands:
            problem.demands[station] = 0

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


# =========================
# Helper functions
# =========================
def is_customer(problem: Problem, node_id: int) -> bool:
    """Check if a node is a customer."""
    return node_id in problem.customers


def is_station(problem: Problem, node_id: int) -> bool:
    """Check if a node is a charging station."""
    return node_id in problem.stations


def is_depot(problem: Problem, node_id: int) -> bool:
    """Check if a node is the depot."""
    return node_id == problem.depot


def print_problem_summary(problem: Problem):
    """Print a summary of the loaded problem."""
    print("=" * 60)
    print(f"Instance: {problem.name}")
    print("=" * 60)
    print(f"Vehicles:          {problem.vehicles}")
    print(f"Dimension:         {problem.n}")
    print(f"Capacity:          {problem.capacity}")
    print(f"Energy Capacity:   {problem.energy_capacity} kWh")
    print(f"Energy Consumption: {problem.energy_consumption} kWh/km")
    print(f"Depot:             {problem.depot}")
    print(f"Customers:         {len(problem.customers)} nodes")
    print(f"Stations:          {len(problem.stations)} nodes")
    print(f"\nStation IDs: {problem.stations}")
    print(f"Depot Coordinates: {problem.coords[problem.depot]}")

    # Sample demands
    if problem.customers:
        print(f"\nSample Customer Demands:")
        for cust in problem.customers[:5]:
            demand = problem.demands.get(cust, 0)
            print(f"  Customer {cust}: {demand}")

    # Sample distances
    if len(problem.customers) >= 2:
        c1, c2 = problem.customers[0], problem.customers[1]
        d_depot_c1 = problem.distance_matrix[problem.depot][c1]
        d_c1_c2 = problem.distance_matrix[c1][c2]
        print(f"\nSample Distances:")
        print(f"  Depot → Customer {c1}: {d_depot_c1:.2f} km")
        print(f"  Customer {c1} → Customer {c2}: {d_c1_c2:.2f} km")


