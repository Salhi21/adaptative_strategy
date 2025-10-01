# evrp/cluster.py
from __future__ import annotations
from typing import List, Tuple
from random import Random
import math

# --- Basic geometry helpers -------------------------------------------------

def sqdist(a: List[float], b: List[float]) -> float:
    # squared Euclidean distance (assumes len(a) == len(b))
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

def nearest_centroid_idx(p: List[float], centers: List[List[float]]) -> int:
    # return index of the closest center (ties broken by first)
    if not centers:
        return 0
    best_i = 0
    best_d = float("inf")
    for i, c in enumerate(centers):
        d = sqdist(p, c)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

# --- Solution embedding ------------------------------------------------------

def embed_solution(sol: List[List[int]], dim: int) -> List[float]:
    """
    Deterministic, lightweight embedding of a solution into R^dim.
    Flatten all routes into a single sequence of node ids, normalize,
    then pad/truncate to 'dim'. This is simple but works for clustering diversity.
    """
    flat: List[int] = []
    for route in sol or []:
        flat.extend(route or [])

    if not flat:
        return [0.0] * dim

    # normalize ids into [0,1] to avoid huge distances if ids are large
    max_id = max(flat) or 1
    vec = [x / max_id for x in flat]

    # pad/truncate to fixed length
    if len(vec) < dim:
        vec.extend([0.0] * (dim - len(vec)))
    else:
        vec = vec[:dim]
    return vec

# --- K-means clustering ------------------------------------------------------

def kmeans(points: List[List[float]],
           k: int,
           rounds: int = 10,
           rng: Random | None = None) -> Tuple[List[List[float]], List[int]]:
    """
    Return (centers, labels) for k-means on 'points'.
    Robust to rng=None, k > n, empty clusters.
    """
    if not points:
        return [], []

    rng = rng or Random()
    n = len(points)
    k = max(1, min(k, n))  # ensure 1 <= k <= n

    # init centers: sample k distinct points
    centers = [points[i] for i in rng.sample(range(n), k)]
    labels = [0] * n

    for _ in range(max(1, rounds)):
        # assign
        for i, p in enumerate(points):
            labels[i] = nearest_centroid_idx(p, centers)

        # update
        new_centers: List[List[float]] = []
        for c in range(k):
            cluster_idx = [i for i, lab in enumerate(labels) if lab == c]
            if cluster_idx:
                dim = len(points[0])
                mean = [0.0] * dim
                for i in cluster_idx:
                    for d in range(dim):
                        mean[d] += points[i][d]
                inv = 1.0 / len(cluster_idx)
                for d in range(dim):
                    mean[d] *= inv
                new_centers.append(mean)
            else:
                # empty cluster → re-seed with a random point to keep k stable
                new_centers.append(points[rng.randrange(n)])

        if new_centers == centers:
            break
        centers = new_centers

    return centers, labels
