"""Geometry utilities shared by simulator, expert, and shield code."""

from __future__ import annotations

from typing import Tuple

import numpy as np


EPS = 1e-8


def l2_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def clip_by_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vectors to a maximum Euclidean norm without changing zero vectors."""

    vectors = np.asarray(vectors, dtype=np.float64)
    norms = l2_norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, EPS))
    return vectors * scale


def pairwise_deltas(positions: np.ndarray) -> np.ndarray:
    """Return pairwise deltas delta[i, j] = positions[i] - positions[j]."""

    return positions[:, None, :] - positions[None, :, :]


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    deltas = pairwise_deltas(positions)
    return l2_norm(deltas, axis=-1)


def collision_pairs(
    positions: np.ndarray,
    radii: np.ndarray,
    margin: float = 0.0,
    tolerance: float = 1e-7,
) -> Tuple[Tuple[int, int], ...]:
    """Return unordered colliding agent pairs."""

    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n_agents = positions.shape[0]
    if n_agents < 2:
        return ()
    distances = pairwise_distances(positions)
    min_separations = radii[:, None] + radii[None, :] + margin
    colliding = distances < (min_separations - tolerance)
    rows, cols = np.nonzero(np.triu(colliding, k=1))
    return tuple((int(i), int(j)) for i, j in zip(rows, cols))


def separation_violation(positions: np.ndarray, radii: np.ndarray, margin: float = 0.0) -> float:
    """Return the largest pairwise shortfall from the requested separation."""

    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if positions.shape[0] < 2:
        return 0.0
    distances = pairwise_distances(positions)
    min_separations = radii[:, None] + radii[None, :] + margin
    shortfalls = min_separations - distances
    rows, cols = np.triu_indices(positions.shape[0], k=1)
    return float(max(0.0, np.max(shortfalls[rows, cols])))


def total_separation_violation(positions: np.ndarray, radii: np.ndarray, margin: float = 0.0) -> float:
    """Return the sum of all pairwise shortfalls from the requested separation."""

    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if positions.shape[0] < 2:
        return 0.0
    distances = pairwise_distances(positions)
    min_separations = radii[:, None] + radii[None, :] + margin
    rows, cols = np.triu_indices(positions.shape[0], k=1)
    shortfalls = np.maximum(0.0, min_separations[rows, cols] - distances[rows, cols])
    return float(np.sum(shortfalls))


def project_positions_to_bounds(
    positions: np.ndarray,
    radii: np.ndarray,
    world_size: Tuple[float, float],
) -> np.ndarray:
    """Keep circular agents inside axis-aligned world bounds."""

    lower = radii[:, None]
    upper = np.asarray(world_size, dtype=np.float64)[None, :] - radii[:, None]
    return np.minimum(np.maximum(positions, lower), upper)


def stable_unit_vector(vector: np.ndarray, fallback_angle: float = 0.0) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > EPS:
        return vector / norm
    return np.array([np.cos(fallback_angle), np.sin(fallback_angle)], dtype=np.float64)
