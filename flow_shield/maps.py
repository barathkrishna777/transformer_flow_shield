"""Moving AI grid maps and continuous obstacle queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


FREE_TILES = frozenset((".", "G", "S"))
BLOCKED_TILES = frozenset(("@", "O", "T", "W"))
EPS = 1e-9


@dataclass(frozen=True)
class GridMap:
    """Static grid obstacle map embedded in a continuous 2D world.

    Continuous coordinates use cell-sized axis-aligned squares:
    cell ``(row, col)`` covers ``[col, col + 1] x [row, row + 1]`` when
    ``cell_size == 1``. Moving AI row order is preserved; the simulator only
    needs a consistent obstacle geometry, not a display convention.
    """

    width: int
    height: int
    blocked: np.ndarray
    cell_size: float = 1.0
    map_type: str = "octile"
    source_path: Optional[str] = None
    name: str = "grid_map"

    def __post_init__(self) -> None:
        blocked = np.asarray(self.blocked, dtype=bool)
        if blocked.shape != (int(self.height), int(self.width)):
            raise ValueError(
                "blocked grid shape must be (height, width); "
                f"got {blocked.shape} for {(self.height, self.width)}."
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("GridMap width and height must be positive.")
        if self.cell_size <= 0.0:
            raise ValueError("GridMap cell_size must be positive.")
        object.__setattr__(self, "blocked", blocked)
        rows, cols = np.nonzero(blocked)
        object.__setattr__(self, "_blocked_rows", rows.astype(np.int64))
        object.__setattr__(self, "_blocked_cols", cols.astype(np.int64))
        if rows.size:
            centers = np.column_stack(
                [
                    (cols.astype(np.float64) + 0.5) * self.cell_size,
                    (rows.astype(np.float64) + 0.5) * self.cell_size,
                ]
            )
        else:
            centers = np.empty((0, 2), dtype=np.float64)
        object.__setattr__(self, "_blocked_centers", centers)

    @property
    def world_size(self) -> Tuple[float, float]:
        return (float(self.width * self.cell_size), float(self.height * self.cell_size))

    @property
    def blocked_count(self) -> int:
        return int(self._blocked_centers.shape[0])

    @property
    def free_count(self) -> int:
        return int(self.width * self.height - self.blocked_count)

    @property
    def blocked_centers(self) -> np.ndarray:
        return np.asarray(self._blocked_centers, dtype=np.float64)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "type": self.map_type,
            "width": int(self.width),
            "height": int(self.height),
            "cell_size": float(self.cell_size),
            "world_size": list(self.world_size),
            "blocked_count": self.blocked_count,
            "free_count": self.free_count,
            "source_path": self.source_path,
            "coordinate_convention": (
                "cell(row,col) occupies [col,col+1] x [row,row+1] scaled by cell_size"
            ),
        }

    def in_bounds_cell(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_cell_free(self, row: int, col: int) -> bool:
        return self.in_bounds_cell(row, col) and not bool(self.blocked[row, col])

    def cell_center(self, row: int, col: int) -> np.ndarray:
        return np.array(
            [
                (float(col) + 0.5) * self.cell_size,
                (float(row) + 0.5) * self.cell_size,
            ],
            dtype=np.float64,
        )

    def point_to_cell(self, point: np.ndarray) -> Tuple[int, int]:
        point = np.asarray(point, dtype=np.float64)
        col = int(np.floor(point[0] / self.cell_size))
        row = int(np.floor(point[1] / self.cell_size))
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        return row, col

    def contains_circle(self, point: np.ndarray, radius: float, margin: float = 0.0) -> bool:
        point = np.asarray(point, dtype=np.float64)
        clearance = float(radius + margin)
        world = np.asarray(self.world_size, dtype=np.float64)
        return bool(np.all(point >= clearance - EPS) and np.all(point <= world - clearance + EPS))

    def circle_collision_details(
        self,
        point: np.ndarray,
        radius: float,
        margin: float = 0.0,
        tolerance: float = 1e-8,
    ) -> Tuple[Tuple[int, int, float], ...]:
        """Return colliding blocked cells as ``(row, col, penetration)`` tuples."""

        point = np.asarray(point, dtype=np.float64)
        required = float(radius + margin)
        collisions: List[Tuple[int, int, float]] = []
        if not self.contains_circle(point, radius, margin):
            world = np.asarray(self.world_size, dtype=np.float64)
            boundary_shortfall = max(
                required - float(point[0]),
                required - float(point[1]),
                float(point[0]) - (world[0] - required),
                float(point[1]) - (world[1] - required),
            )
            if boundary_shortfall > tolerance:
                collisions.append((-1, -1, float(boundary_shortfall)))
        if self.blocked_count == 0:
            return tuple(collisions)

        mins = np.column_stack([self._blocked_cols, self._blocked_rows]).astype(np.float64)
        mins *= self.cell_size
        maxs = mins + self.cell_size
        closest = np.minimum(np.maximum(point[None, :], mins), maxs)
        distances = np.linalg.norm(point[None, :] - closest, axis=1)
        penetrations = required - distances
        hit_indices = np.nonzero(penetrations > tolerance)[0]
        for index in hit_indices:
            collisions.append(
                (
                    int(self._blocked_rows[index]),
                    int(self._blocked_cols[index]),
                    float(penetrations[index]),
                )
            )
        return tuple(collisions)

    def circle_collides(self, point: np.ndarray, radius: float, margin: float = 0.0) -> bool:
        return bool(self.circle_collision_details(point, radius, margin))

    def max_circle_penetration(
        self,
        point: np.ndarray,
        radius: float,
        margin: float = 0.0,
    ) -> float:
        details = self.circle_collision_details(point, radius, margin)
        if not details:
            return 0.0
        return float(max(penetration for _, _, penetration in details))

    def circle_collisions(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        margin: float = 0.0,
    ) -> Tuple[Tuple[int, int, int], ...]:
        collisions: List[Tuple[int, int, int]] = []
        for agent_index, (point, radius) in enumerate(zip(positions, radii)):
            for row, col, _ in self.circle_collision_details(point, float(radius), margin):
                collisions.append((int(agent_index), int(row), int(col)))
        return tuple(collisions)

    def max_penetration(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        margin: float = 0.0,
    ) -> float:
        if len(positions) == 0:
            return 0.0
        return float(
            max(
                self.max_circle_penetration(point, float(radius), margin)
                for point, radius in zip(positions, radii)
            )
        )

    def motion_hits(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        margin: float = 0.0,
    ) -> bool:
        _, hit, _ = self.constrain_motion(start, end, radius, margin)
        return bool(hit)

    def constrain_motion(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        margin: float = 0.0,
    ) -> Tuple[np.ndarray, bool, float]:
        """Return the farthest point on ``start -> end`` that stays collision-free."""

        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        radius = float(radius)
        margin = float(margin)
        if self.circle_collides(start, radius, margin):
            return start.copy(), True, self.max_circle_penetration(start, radius, margin)
        if np.allclose(start, end):
            return end.copy(), False, 0.0
        delta = end - start
        length = float(np.linalg.norm(delta))
        step = max(min(self.cell_size * 0.25, max(radius + margin, 1e-3) * 0.5), 0.025)
        samples = max(2, int(np.ceil(length / step)))
        last_good = 0.0
        first_bad: Optional[float] = None
        for sample in range(1, samples + 1):
            t = sample / samples
            candidate = start + delta * t
            if self.circle_collides(candidate, radius, margin):
                first_bad = t
                break
            last_good = t
        if first_bad is None:
            return end.copy(), False, 0.0

        lo = last_good
        hi = first_bad
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            candidate = start + delta * mid
            if self.circle_collides(candidate, radius, margin):
                hi = mid
            else:
                lo = mid
        constrained = start + delta * lo
        penetration = self.max_circle_penetration(start + delta * hi, radius, margin)
        return constrained, True, penetration

    def constrain_positions(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        radii: np.ndarray,
        margin: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[int, ...], float]:
        constrained = np.asarray(ends, dtype=np.float64).copy()
        hit_agents: List[int] = []
        max_penetration = 0.0
        for index, (start, end, radius) in enumerate(zip(starts, ends, radii)):
            adjusted, hit, penetration = self.constrain_motion(
                start,
                end,
                float(radius),
                margin=margin,
            )
            constrained[index] = adjusted
            if hit:
                hit_agents.append(int(index))
                max_penetration = max(max_penetration, float(penetration))
        return constrained, tuple(hit_agents), float(max_penetration)

    def nearest_obstacle_tokens(
        self,
        point: np.ndarray,
        max_tokens: int,
        max_range: float,
        radius: float,
        margin: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return nearest blocked-cell centers and clearances for observation tokens."""

        if max_tokens <= 0 or self.blocked_count == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        point = np.asarray(point, dtype=np.float64)
        centers = self.blocked_centers
        center_distances = np.linalg.norm(centers - point[None, :], axis=1)
        if max_range > 0.0:
            candidate_indices = np.nonzero(center_distances <= max_range + self.cell_size)[0]
        else:
            candidate_indices = np.arange(centers.shape[0])
        if candidate_indices.size == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        order = candidate_indices[np.argsort(center_distances[candidate_indices])]
        order = order[:max_tokens]
        clearances = []
        for index in order:
            row = int(self._blocked_rows[index])
            col = int(self._blocked_cols[index])
            mins = np.array([col, row], dtype=np.float64) * self.cell_size
            maxs = mins + self.cell_size
            closest = np.minimum(np.maximum(point, mins), maxs)
            distance = float(np.linalg.norm(point - closest))
            clearances.append(distance - float(radius + margin))
        return centers[order].copy(), np.asarray(clearances, dtype=np.float64)

    def random_free_point(
        self,
        rng: np.random.Generator,
        radius: float,
        margin: float = 0.0,
    ) -> np.ndarray:
        if self.free_count <= 0:
            raise RuntimeError("Cannot sample from a fully blocked map.")
        free_rows, free_cols = np.nonzero(~self.blocked)
        order = rng.permutation(free_rows.shape[0])
        clearance = float(radius + margin)
        for index in order:
            row = int(free_rows[index])
            col = int(free_cols[index])
            lower = np.array([col, row], dtype=np.float64) * self.cell_size + clearance
            upper = np.array([col + 1, row + 1], dtype=np.float64) * self.cell_size - clearance
            if np.any(lower > upper):
                candidate = self.cell_center(row, col)
            else:
                candidate = rng.uniform(lower, upper)
            if not self.circle_collides(candidate, radius, margin):
                return np.asarray(candidate, dtype=np.float64)
        raise RuntimeError(
            "Could not sample a free point with the requested radius/clearance."
        )


def load_moving_ai_map(path: str | Path, cell_size: float = 1.0) -> GridMap:
    """Parse a standard Moving AI ``.map`` file."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header: Dict[str, str] = {}
    map_start: Optional[int] = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "map":
            map_start = index + 1
            break
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid Moving AI map header line: {raw_line!r}")
        header[parts[0].lower()] = parts[1].strip()
    if map_start is None:
        raise ValueError(f"{path} is missing the 'map' grid marker.")
    map_type = header.get("type", "").lower()
    if map_type != "octile":
        raise ValueError(f"Unsupported Moving AI map type {map_type!r}; expected 'octile'.")
    try:
        height = int(header["height"])
        width = int(header["width"])
    except KeyError as exc:
        raise ValueError(f"{path} is missing required Moving AI map header {exc}.") from exc

    grid_lines = [line.rstrip("\n") for line in lines[map_start : map_start + height]]
    if len(grid_lines) != height:
        raise ValueError(f"{path} declares height={height} but has {len(grid_lines)} map rows.")
    blocked = np.zeros((height, width), dtype=bool)
    for row, grid_line in enumerate(grid_lines):
        if len(grid_line) != width:
            raise ValueError(
                f"{path} row {row} has width {len(grid_line)}; expected {width}."
            )
        for col, char in enumerate(grid_line):
            if char in FREE_TILES:
                blocked[row, col] = False
            elif char in BLOCKED_TILES:
                blocked[row, col] = True
            else:
                raise ValueError(
                    f"{path} contains unsupported Moving AI tile {char!r} at row={row}, col={col}."
                )
    return GridMap(
        width=width,
        height=height,
        blocked=blocked,
        cell_size=float(cell_size),
        map_type=map_type,
        source_path=str(path),
        name=path.stem,
    )


def load_obstacle_map(path: Optional[str | Path], cell_size: float = 1.0) -> Optional[GridMap]:
    if path is None:
        return None
    return load_moving_ai_map(path, cell_size=cell_size)


def map_metadata(obstacle_map: Optional[GridMap]) -> Optional[Dict[str, object]]:
    if obstacle_map is None:
        return None
    return obstacle_map.metadata()

