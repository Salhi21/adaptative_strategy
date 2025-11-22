# evrp/elite.py
from __future__ import annotations
from typing import List, Tuple, Sequence, Iterable, Optional, Union
from random import Random
from .cluster import embed_solution, nearest_centroid_idx, kmeans

# (cost, sol, embedding)
EliteEntry = Tuple[float, List[List[int]], List[float]]

def update_elite_archive(
    elite: List[EliteEntry],
    sol: List[List[int]],
    cost: float,
    dim: int = 32,          # <-- default embedding size
    max_size: int = 100     # <-- default archive cap
):
    emb = embed_solution(sol, dim)
    elite.append((cost, sol, emb))
    elite.sort(key=lambda e: e[0])
    if len(elite) > max_size:
        del elite[max_size:]

def _iter_entries(elite: Union[List[EliteEntry], dict]) -> Iterable[EliteEntry]:
    # Accept both list and dict to be defensive
    return elite.values() if isinstance(elite, dict) else elite


def _normalize_points(points: List[List[float]]) -> List[List[float]]:
    """
    Make all vectors the same length (pad with zeros or truncate).
    We choose the MAX length so we don't throw info away unless necessary.
    """
    if not points:
        return points
    target = max(len(p) for p in points)
    norm = []
    for p in points:
        if len(p) < target:
            norm.append(p + [0.0] * (target - len(p)))
        elif len(p) > target:
            norm.append(p[:target])
        else:
            norm.append(p)
    return norm


def cluster_elite_archive(
    elite: Union[List[EliteEntry], dict],
    k: int = 3,
    rounds: int = 10,
    rng: Optional[Random] = None
) -> List[List[float]]:
    entries = list(_iter_entries(elite))
    if not entries:
        return []

    points = [e[2] for e in entries if e[2] is not None]
    if not points:
        return []

    points = _normalize_points(points)        # <-- enforce uniform dimensionality
    k = min(k, len(points))
    centers, _ = kmeans(points, k, rounds, rng)
    return centers
