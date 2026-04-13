"""Expert data generation for phase 0 continuous-space MAPF."""

from __future__ import annotations

import heapq
from collections import Counter, OrderedDict
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from .config import DatasetConfig, SimConfig
from .geometry import clip_by_norm, collision_pairs, project_positions_to_bounds
from .maps import GridMap, MovingAIScenarioTask, load_moving_ai_scen, load_obstacle_map
from .simulator import ContinuousWorld, Scenario, sim_config_for_scenario


def straight_line_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    config: SimConfig,
) -> np.ndarray:
    """Empty-map optimal velocity intent: move straight toward each goal."""

    deltas = np.asarray(goals, dtype=np.float64) - np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(deltas, axis=1, keepdims=True)
    desired = deltas / np.maximum(distances, 1e-8)
    speeds = np.minimum(config.max_speed, distances / max(config.dt, 1e-8))
    velocities = desired * speeds
    velocities[distances[:, 0] <= config.goal_tolerance] = 0.0
    return clip_by_norm(velocities, config.max_speed)


def obstacle_aware_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    config: SimConfig,
    static_obstacles: Tuple[Tuple[float, float, float], ...],
) -> np.ndarray:
    """Legacy circular-obstacle potential-field intent for static_obstacles."""

    velocities = straight_line_velocity(positions, goals, config)
    if not static_obstacles:
        return velocities
    corrections = np.zeros_like(velocities)
    influence = 1.25
    for x, y, radius in static_obstacles:
        center = np.array([x, y], dtype=np.float64)
        delta = positions - center[None, :]
        distances = np.linalg.norm(delta, axis=1, keepdims=True)
        clearance = distances - radius - config.agent_radius
        active = clearance < influence
        direction = delta / np.maximum(distances, 1e-8)
        strength = np.maximum(0.0, (influence - clearance) / influence)
        corrections += direction * strength * config.max_speed * active
    return clip_by_norm(velocities + corrections, config.max_speed)


_NEIGHBOR_DELTAS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)),
    (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)),
    (1, 1, np.sqrt(2.0)),
)
_WAIT_DELTA = (0, 0, 1.0)
_CELL_CLEARANCE_CACHE_MAX = 250_000
_ASTAR_PATH_CACHE_MAX = 100_000
_CELL_CLEARANCE_CACHE: "OrderedDict[Tuple[Hashable, ...], bool]" = OrderedDict()
_ASTAR_PATH_CACHE: "OrderedDict[Tuple[Hashable, ...], Optional[Tuple[Tuple[int, int], ...]]]" = OrderedDict()
_ASTAR_CACHE_STATS = Counter()


def normalize_expert_type(expert_type: str | None) -> str:
    normalized = (expert_type or "independent_astar").strip().lower().replace("-", "_")
    aliases = {
        "independent": "independent_astar",
        "astar": "independent_astar",
        "independent_astar": "independent_astar",
        "waypoint_astar": "independent_astar",
        "prioritized": "prioritized_astar",
        "reservation": "prioritized_astar",
        "reservation_astar": "prioritized_astar",
        "prioritized_astar": "prioritized_astar",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported expert_type={expert_type!r}; expected 'independent_astar' "
            "or 'prioritized_astar'."
        )
    return aliases[normalized]


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def _map_cache_key(obstacle_map: GridMap) -> Tuple[Hashable, ...]:
    return (
        id(obstacle_map),
        obstacle_map.source_path,
        obstacle_map.name,
        int(obstacle_map.width),
        int(obstacle_map.height),
        _rounded_float(obstacle_map.cell_size),
        int(obstacle_map.blocked_count),
    )


def _cache_get(cache: OrderedDict, key: Tuple[Hashable, ...]):
    try:
        value = cache.pop(key)
    except KeyError:
        return None, False
    cache[key] = value
    return value, True


def _cache_put(cache: OrderedDict, key: Tuple[Hashable, ...], value, max_size: int) -> None:
    cache[key] = value
    if len(cache) > max_size:
        cache.popitem(last=False)


def clear_astar_caches() -> None:
    """Clear bounded A* and clearance caches, mainly for tests/benchmark hygiene."""

    _CELL_CLEARANCE_CACHE.clear()
    _ASTAR_PATH_CACHE.clear()
    _ASTAR_CACHE_STATS.clear()


def astar_cache_info() -> Dict[str, int]:
    """Return lightweight cache diagnostics for benchmark metadata/debugging."""

    return {
        "cell_clearance_entries": len(_CELL_CLEARANCE_CACHE),
        "astar_path_entries": len(_ASTAR_PATH_CACHE),
        "cell_clearance_hits": int(_ASTAR_CACHE_STATS["cell_clearance_hits"]),
        "cell_clearance_misses": int(_ASTAR_CACHE_STATS["cell_clearance_misses"]),
        "astar_path_hits": int(_ASTAR_CACHE_STATS["astar_path_hits"]),
        "astar_path_misses": int(_ASTAR_CACHE_STATS["astar_path_misses"]),
        "reservation_astar_calls": int(_ASTAR_CACHE_STATS["reservation_astar_calls"]),
        "reservation_astar_failures": int(_ASTAR_CACHE_STATS["reservation_astar_failures"]),
        "reservation_astar_expansion_limit_hits": int(
            _ASTAR_CACHE_STATS["reservation_astar_expansion_limit_hits"]
        ),
        "prioritized_fallback_to_independent": int(
            _ASTAR_CACHE_STATS["prioritized_fallback_to_independent"]
        ),
        "prioritized_total_failures": int(_ASTAR_CACHE_STATS["prioritized_total_failures"]),
    }


def _cell_has_clearance(
    obstacle_map: GridMap,
    row: int,
    col: int,
    radius: float,
    margin: float,
) -> bool:
    if not obstacle_map.is_cell_free(row, col):
        return False
    key = (
        *_map_cache_key(obstacle_map),
        int(row),
        int(col),
        _rounded_float(radius),
        _rounded_float(margin),
    )
    cached, hit = _cache_get(_CELL_CLEARANCE_CACHE, key)
    if hit:
        _ASTAR_CACHE_STATS["cell_clearance_hits"] += 1
        return bool(cached)
    _ASTAR_CACHE_STATS["cell_clearance_misses"] += 1
    value = not obstacle_map.circle_collides(
        obstacle_map.cell_center(row, col),
        radius,
        margin=margin,
    )
    _cache_put(_CELL_CLEARANCE_CACHE, key, bool(value), _CELL_CLEARANCE_CACHE_MAX)
    return bool(value)


def _nearest_clear_cell(
    obstacle_map: GridMap,
    point: np.ndarray,
    radius: float,
    margin: float,
) -> Optional[Tuple[int, int]]:
    start = obstacle_map.point_to_cell(point)
    if _cell_has_clearance(obstacle_map, start[0], start[1], radius, margin):
        return start
    free_rows, free_cols = np.nonzero(~obstacle_map.blocked)
    if free_rows.size == 0:
        return None
    centers = np.column_stack(
        [
            (free_cols.astype(np.float64) + 0.5) * obstacle_map.cell_size,
            (free_rows.astype(np.float64) + 0.5) * obstacle_map.cell_size,
        ]
    )
    distances = np.linalg.norm(centers - np.asarray(point, dtype=np.float64)[None, :], axis=1)
    for index in np.argsort(distances):
        row = int(free_rows[index])
        col = int(free_cols[index])
        if _cell_has_clearance(obstacle_map, row, col, radius, margin):
            return row, col
    return None


def _octile_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diagonal = min(dr, dc)
    straight = max(dr, dc) - diagonal
    return float(diagonal * np.sqrt(2.0) + straight)


def astar_grid_path(
    obstacle_map: GridMap,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    margin: float = 0.0,
) -> Optional[List[Tuple[int, int]]]:
    """Plan a clearance-aware octile-grid path over a Moving AI map."""

    start_cell = _nearest_clear_cell(obstacle_map, start, radius, margin)
    goal_cell = _nearest_clear_cell(obstacle_map, goal, radius, margin)
    if start_cell is None or goal_cell is None:
        return None
    cache_key = (
        *_map_cache_key(obstacle_map),
        start_cell,
        goal_cell,
        _rounded_float(radius),
        _rounded_float(margin),
    )
    cached, hit = _cache_get(_ASTAR_PATH_CACHE, cache_key)
    if hit:
        _ASTAR_CACHE_STATS["astar_path_hits"] += 1
        return list(cached) if cached is not None else None
    _ASTAR_CACHE_STATS["astar_path_misses"] += 1
    if start_cell == goal_cell:
        path = (start_cell,)
        _cache_put(_ASTAR_PATH_CACHE, cache_key, path, _ASTAR_PATH_CACHE_MAX)
        return list(path)

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (_octile_heuristic(start_cell, goal_cell), 0.0, start_cell))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_cost: Dict[Tuple[int, int], float] = {start_cell: 0.0}
    closed = set()

    while open_heap:
        _, cost, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        if cell == goal_cell:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            result = tuple(reversed(path))
            _cache_put(_ASTAR_PATH_CACHE, cache_key, result, _ASTAR_PATH_CACHE_MAX)
            return list(result)
        closed.add(cell)
        row, col = cell
        for dr, dc, step_cost in _NEIGHBOR_DELTAS:
            next_cell = (row + dr, col + dc)
            if not _cell_has_clearance(
                obstacle_map,
                next_cell[0],
                next_cell[1],
                radius,
                margin,
            ):
                continue
            if dr != 0 and dc != 0:
                if not (
                    _cell_has_clearance(obstacle_map, row + dr, col, radius, margin)
                    and _cell_has_clearance(obstacle_map, row, col + dc, radius, margin)
                ):
                    continue
            new_cost = cost + float(step_cost)
            if new_cost + 1e-12 >= best_cost.get(next_cell, np.inf):
                continue
            best_cost[next_cell] = new_cost
            came_from[next_cell] = cell
            priority = new_cost + _octile_heuristic(next_cell, goal_cell)
            heapq.heappush(open_heap, (priority, new_cost, next_cell))
    _cache_put(_ASTAR_PATH_CACHE, cache_key, None, _ASTAR_PATH_CACHE_MAX)
    return None


def _astar_grid_path_with_reservations(
    obstacle_map: GridMap,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    margin: float,
    reserved_vertices: set[Tuple[int, Tuple[int, int]]],
    reserved_edges: set[Tuple[int, Tuple[int, int], Tuple[int, int]]],
    max_time: int,
    expansion_limit: Optional[int] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Time-expanded A* with vertex/edge reservations and wait actions."""

    _ASTAR_CACHE_STATS["reservation_astar_calls"] += 1
    start_cell = _nearest_clear_cell(obstacle_map, start, radius, margin)
    goal_cell = _nearest_clear_cell(obstacle_map, goal, radius, margin)
    if start_cell is None or goal_cell is None:
        _ASTAR_CACHE_STATS["reservation_astar_failures"] += 1
        return None
    if (0, start_cell) in reserved_vertices:
        _ASTAR_CACHE_STATS["reservation_astar_failures"] += 1
        return None

    open_heap: List[Tuple[float, float, int, Tuple[int, int]]] = []
    heapq.heappush(
        open_heap,
        (_octile_heuristic(start_cell, goal_cell), 0.0, 0, start_cell),
    )
    came_from: Dict[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]] = {}
    best_cost: Dict[Tuple[int, Tuple[int, int]], float] = {(0, start_cell): 0.0}
    closed = set()
    moves = _NEIGHBOR_DELTAS + (_WAIT_DELTA,)
    expanded = 0
    if expansion_limit is None:
        expansion_limit = max(1_000, int(max_time) * int(obstacle_map.free_count + 1))

    while open_heap:
        _, cost, time_step, cell = heapq.heappop(open_heap)
        state = (time_step, cell)
        if state in closed:
            continue
        expanded += 1
        if expanded > expansion_limit:
            _ASTAR_CACHE_STATS["reservation_astar_expansion_limit_hits"] += 1
            _ASTAR_CACHE_STATS["reservation_astar_failures"] += 1
            return None
        if cell == goal_cell:
            path = [cell]
            while state in came_from:
                state = came_from[state]
                path.append(state[1])
            return list(reversed(path))
        if time_step >= max_time:
            continue
        closed.add(state)
        row, col = cell
        for dr, dc, step_cost in moves:
            next_cell = (row + dr, col + dc)
            next_time = time_step + 1
            if not _cell_has_clearance(
                obstacle_map,
                next_cell[0],
                next_cell[1],
                radius,
                margin,
            ):
                continue
            if dr != 0 and dc != 0:
                if not (
                    _cell_has_clearance(obstacle_map, row + dr, col, radius, margin)
                    and _cell_has_clearance(obstacle_map, row, col + dc, radius, margin)
                ):
                    continue
            if (next_time, next_cell) in reserved_vertices:
                continue
            if (next_time, cell, next_cell) in reserved_edges:
                continue
            next_state = (next_time, next_cell)
            new_cost = cost + float(step_cost)
            if new_cost + 1e-12 >= best_cost.get(next_state, np.inf):
                continue
            best_cost[next_state] = new_cost
            came_from[next_state] = state
            priority = new_cost + _octile_heuristic(next_cell, goal_cell)
            heapq.heappush(open_heap, (priority, new_cost, next_time, next_cell))
    _ASTAR_CACHE_STATS["reservation_astar_failures"] += 1
    return None


def obstacle_map_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    config: SimConfig,
    obstacle_map: GridMap,
    radii: Optional[np.ndarray] = None,
) -> np.ndarray:
    """A* waypoint-following expert for static Moving AI obstacle maps."""

    positions = np.asarray(positions, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    radii = (
        np.full(positions.shape[0], config.agent_radius, dtype=np.float64)
        if radii is None
        else np.asarray(radii, dtype=np.float64)
    )
    targets = goals.copy()
    waypoint_tolerance = max(config.goal_tolerance * 0.75, config.agent_radius * 1.5)
    for index, (position, goal, radius) in enumerate(zip(positions, goals, radii)):
        if np.linalg.norm(goal - position) <= config.goal_tolerance:
            targets[index] = position
            continue
        path = astar_grid_path(
            obstacle_map,
            position,
            goal,
            float(radius),
            margin=config.safety_margin,
        )
        if not path:
            targets[index] = position
            continue
        waypoints = [obstacle_map.cell_center(row, col) for row, col in path[1:]]
        waypoints.append(goal)
        for waypoint in waypoints:
            if np.linalg.norm(waypoint - position) > waypoint_tolerance:
                targets[index] = waypoint
                break
        else:
            targets[index] = goal
    return straight_line_velocity(positions, targets, config)


def prioritized_obstacle_map_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    config: SimConfig,
    obstacle_map: GridMap,
    radii: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reservation-table A* waypoint expert baseline for obstacle maps.

    This is a lightweight prioritized planner: agents are ordered by remaining
    distance and planned one by one with vertex and edge reservations. It is a
    coordination baseline, not a complete joint MAPF solver.
    """

    positions = np.asarray(positions, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    radii = (
        np.full(positions.shape[0], config.agent_radius, dtype=np.float64)
        if radii is None
        else np.asarray(radii, dtype=np.float64)
    )
    targets = goals.copy()
    reservations: set[Tuple[int, Tuple[int, int]]] = set()
    edge_reservations: set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = set()
    waypoint_tolerance = max(config.goal_tolerance * 0.75, config.agent_radius * 1.5)
    remaining = np.linalg.norm(goals - positions, axis=1)
    order = np.lexsort((np.arange(positions.shape[0]), -remaining))
    max_time = max(
        8,
        int((obstacle_map.width + obstacle_map.height) * 4 + positions.shape[0] * 2),
    )
    expansion_limit = max(1_000, int(max_time) * int(obstacle_map.free_count + 1))
    for index in order:
        index = int(index)
        position = positions[index]
        goal = goals[index]
        radius = float(radii[index])
        if np.linalg.norm(goal - position) <= config.goal_tolerance:
            targets[index] = position
            start_cell = _nearest_clear_cell(obstacle_map, position, radius, config.safety_margin)
            if start_cell is not None:
                reservations.add((0, start_cell))
            continue
        path = _astar_grid_path_with_reservations(
            obstacle_map,
            position,
            goal,
            radius,
            margin=config.safety_margin,
            reserved_vertices=reservations,
            reserved_edges=edge_reservations,
            max_time=max_time,
            expansion_limit=expansion_limit,
        )
        if not path:
            path = astar_grid_path(
                obstacle_map,
                position,
                goal,
                radius,
                margin=config.safety_margin,
            )
            if path:
                _ASTAR_CACHE_STATS["prioritized_fallback_to_independent"] += 1
        if not path:
            _ASTAR_CACHE_STATS["prioritized_total_failures"] += 1
            targets[index] = position
            continue
        for time_step, cell in enumerate(path):
            reservations.add((time_step, cell))
            if time_step > 0:
                prev = path[time_step - 1]
                edge_reservations.add((time_step, cell, prev))
        for time_step in range(len(path), min(max_time, len(path) + 8)):
            reservations.add((time_step, path[-1]))
        waypoints = [obstacle_map.cell_center(row, col) for row, col in path[1:]]
        waypoints.append(goal)
        for waypoint in waypoints:
            if np.linalg.norm(waypoint - position) > waypoint_tolerance:
                targets[index] = waypoint
                break
        else:
            targets[index] = goal
    return straight_line_velocity(positions, targets, config)


def obstacle_map_expert_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    config: SimConfig,
    obstacle_map: GridMap,
    radii: Optional[np.ndarray] = None,
    expert_type: str = "independent_astar",
) -> np.ndarray:
    expert_type = normalize_expert_type(expert_type)
    if expert_type == "prioritized_astar":
        return prioritized_obstacle_map_velocity(
            positions,
            goals,
            config,
            obstacle_map,
            radii=radii,
        )
    return obstacle_map_velocity(positions, goals, config, obstacle_map, radii=radii)


def _sample_points(
    rng: np.random.Generator,
    count: int,
    config: SimConfig,
    min_pair_distance: float,
    forbidden: Optional[np.ndarray] = None,
) -> np.ndarray:
    points: List[np.ndarray] = []
    lower = config.agent_radius
    upper = np.asarray(config.world_size, dtype=np.float64) - config.agent_radius
    forbidden = np.empty((0, 2), dtype=np.float64) if forbidden is None else forbidden
    attempts = 0
    attempt_limit = max(20000, int(count) * 1000)
    while len(points) < count and attempts < attempt_limit:
        attempts += 1
        candidate = rng.uniform(lower, upper, size=2)
        existing = np.vstack([forbidden, np.asarray(points).reshape(-1, 2)])
        if existing.size == 0:
            points.append(candidate)
            continue
        distances = np.linalg.norm(existing - candidate[None, :], axis=1)
        if np.all(distances >= min_pair_distance):
            points.append(candidate)
    if len(points) != count and forbidden.size == 0:
        points = _grid_sample_points(
            rng,
            count,
            lower,
            upper,
            min_pair_distance,
        )
    if len(points) != count:
        raise RuntimeError(
            "Could not sample non-overlapping points; reduce agent count/radius "
            "or increase world_size."
        )
    return np.asarray(points, dtype=np.float64)


def _grid_sample_points(
    rng: np.random.Generator,
    count: int,
    lower: float,
    upper: np.ndarray,
    min_pair_distance: float,
) -> List[np.ndarray]:
    """Fallback sampler for dense scaled empty-map starts."""

    width = np.maximum(upper - lower, 0.0)
    grid_counts = np.floor(width / max(min_pair_distance, 1e-8)).astype(int) + 1
    if int(np.prod(grid_counts)) < count:
        return []
    xs = np.linspace(lower, upper[0], grid_counts[0])
    ys = np.linspace(lower, upper[1], grid_counts[1])
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float64).reshape(2, -1).T
    order = rng.permutation(grid.shape[0])[:count]
    return [grid[index] for index in order]


def _sample_obstacle_free_points(
    rng: np.random.Generator,
    count: int,
    config: SimConfig,
    obstacle_map: GridMap,
    min_pair_distance: float,
    forbidden: Optional[np.ndarray] = None,
) -> np.ndarray:
    points: List[np.ndarray] = []
    forbidden = np.empty((0, 2), dtype=np.float64) if forbidden is None else forbidden
    attempts = 0
    attempt_limit = max(50000, int(count) * 5000)
    clearance_margin = config.safety_margin
    while len(points) < count and attempts < attempt_limit:
        attempts += 1
        candidate = obstacle_map.random_free_point(
            rng,
            radius=config.agent_radius,
            margin=clearance_margin,
        )
        existing = np.vstack([forbidden, np.asarray(points).reshape(-1, 2)])
        if existing.size:
            distances = np.linalg.norm(existing - candidate[None, :], axis=1)
            if not np.all(distances >= min_pair_distance):
                continue
        points.append(candidate)
    if len(points) != count:
        raise RuntimeError(
            "Could not sample non-overlapping obstacle-free points; reduce agent "
            "count/radius or choose a less constrained map."
        )
    return np.asarray(points, dtype=np.float64)


def sample_empty_scenario(
    rng: np.random.Generator,
    num_agents: int,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
) -> Scenario:
    """Sample starts and goals in an empty box without initial overlaps."""

    min_pair_distance = 2.5 * sim_config.agent_radius + sim_config.safety_margin
    starts = _sample_points(rng, num_agents, sim_config, min_pair_distance)
    goals: List[np.ndarray] = []
    attempts = 0
    lower = sim_config.agent_radius
    upper = np.asarray(sim_config.world_size, dtype=np.float64) - sim_config.agent_radius
    goal_attempt_limit = max(30000, int(num_agents) * 2000)
    while len(goals) < num_agents and attempts < goal_attempt_limit:
        attempts += 1
        candidate = rng.uniform(lower, upper, size=2)
        if np.linalg.norm(candidate - starts[len(goals)]) < dataset_config.min_start_goal_distance:
            continue
        existing = np.asarray(goals).reshape(-1, 2) if goals else np.empty((0, 2))
        if existing.size and np.any(np.linalg.norm(existing - candidate[None, :], axis=1) < min_pair_distance):
            continue
        goals.append(candidate)
    if len(goals) != num_agents:
        raise RuntimeError("Could not sample valid goals for scenario.")
    radii = np.full(num_agents, sim_config.agent_radius, dtype=np.float64)
    return Scenario(
        starts=starts,
        goals=np.asarray(goals, dtype=np.float64),
        radii=radii,
        world_size=sim_config.world_size,
    )


def sample_obstacle_map_scenario(
    rng: np.random.Generator,
    num_agents: int,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    obstacle_map: GridMap,
) -> Scenario:
    """Sample starts/goals in free space on a static obstacle map."""

    map_config = sim_config_for_scenario(
        sim_config,
        Scenario(
            starts=np.empty((0, 2), dtype=np.float64),
            goals=np.empty((0, 2), dtype=np.float64),
            radii=np.empty((0,), dtype=np.float64),
            world_size=obstacle_map.world_size,
            obstacle_map=obstacle_map,
        ),
    )
    min_pair_distance = 2.5 * map_config.agent_radius + map_config.safety_margin
    starts = _sample_obstacle_free_points(
        rng,
        num_agents,
        map_config,
        obstacle_map,
        min_pair_distance,
    )
    goals: List[np.ndarray] = []
    attempts = 0
    goal_attempt_limit = max(80000, int(num_agents) * 8000)
    while len(goals) < num_agents and attempts < goal_attempt_limit:
        attempts += 1
        candidate = obstacle_map.random_free_point(
            rng,
            radius=map_config.agent_radius,
            margin=map_config.safety_margin,
        )
        agent_index = len(goals)
        if np.linalg.norm(candidate - starts[agent_index]) < dataset_config.min_start_goal_distance:
            continue
        existing = np.asarray(goals).reshape(-1, 2) if goals else np.empty((0, 2))
        if existing.size and np.any(
            np.linalg.norm(existing - candidate[None, :], axis=1) < min_pair_distance
        ):
            continue
        path = astar_grid_path(
            obstacle_map,
            starts[agent_index],
            candidate,
            map_config.agent_radius,
            margin=map_config.safety_margin,
        )
        if not path:
            continue
        goals.append(candidate)
    if len(goals) != num_agents:
        raise RuntimeError(
            "Could not sample valid obstacle-map goals with a collision-free A* path."
        )
    radii = np.full(num_agents, map_config.agent_radius, dtype=np.float64)
    return Scenario(
        starts=starts,
        goals=np.asarray(goals, dtype=np.float64),
        radii=radii,
        world_size=obstacle_map.world_size,
        obstacle_map=obstacle_map,
    )


SCEN_SKIP_REASONS = (
    "map_dimension_mismatch",
    "start_collision",
    "goal_collision",
    "too_short_start_goal_distance",
    "unreachable_astar",
    "grouping_overlap",
    "insufficient_valid_tasks",
)


def _scen_task_skip_reason(
    task: MovingAIScenarioTask,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    obstacle_map: GridMap,
) -> Optional[str]:
    radius = float(sim_config.agent_radius)
    margin = float(sim_config.safety_margin)
    if task.map_width != obstacle_map.width or task.map_height != obstacle_map.height:
        return "map_dimension_mismatch"
    if np.linalg.norm(task.goal - task.start) < dataset_config.min_start_goal_distance:
        return "too_short_start_goal_distance"
    if obstacle_map.circle_collides(task.start, radius, margin=margin):
        return "start_collision"
    if obstacle_map.circle_collides(task.goal, radius, margin=margin):
        return "goal_collision"
    if astar_grid_path(obstacle_map, task.start, task.goal, radius, margin=margin) is None:
        return "unreachable_astar"
    return None


def _valid_scen_task(
    task: MovingAIScenarioTask,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    obstacle_map: GridMap,
) -> bool:
    return _scen_task_skip_reason(task, sim_config, dataset_config, obstacle_map) is None


def scen_task_diagnostics(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
    obstacle_map: GridMap,
) -> Dict[str, object]:
    """Return Moving AI .scen validity counts before multi-agent grouping."""

    if dataset_config.scen_path is None:
        return {
            "scenario_source": dataset_config.scenario_source,
            "raw_tasks": 0,
            "valid_tasks": 0,
            "skipped_tasks": 0,
            "skip_counts": {reason: 0 for reason in SCEN_SKIP_REASONS},
        }
    raw_tasks = load_moving_ai_scen(
        dataset_config.scen_path,
        cell_size=obstacle_map.cell_size,
        limit=dataset_config.scen_limit,
    )
    skip_counts = Counter({reason: 0 for reason in SCEN_SKIP_REASONS})
    valid = 0
    for task in raw_tasks:
        reason = _scen_task_skip_reason(task, sim_config, dataset_config, obstacle_map)
        if reason is None:
            valid += 1
        else:
            skip_counts[reason] += 1
    return {
        "scenario_source": dataset_config.scenario_source,
        "scen_path": dataset_config.scen_path,
        "scen_limit": dataset_config.scen_limit,
        "raw_tasks": len(raw_tasks),
        "valid_tasks": int(valid),
        "skipped_tasks": len(raw_tasks) - int(valid),
        "skip_counts": dict(skip_counts),
    }


def _group_has_overlap(
    group: List[MovingAIScenarioTask],
    map_config: SimConfig,
    min_pair_distance: float,
) -> bool:
    if len(group) <= 1:
        return False
    starts = np.stack([task.start for task in group], axis=0)
    goals = np.stack([task.goal for task in group], axis=0)
    radii = np.full(len(group), map_config.agent_radius, dtype=np.float64)
    if collision_pairs(starts, radii, margin=map_config.safety_margin):
        return True
    if collision_pairs(goals, radii, margin=map_config.safety_margin):
        return True
    start_distances = np.linalg.norm(starts[:, None, :] - starts[None, :, :], axis=-1)
    goal_distances = np.linalg.norm(goals[:, None, :] - goals[None, :, :], axis=-1)
    np.fill_diagonal(start_distances, np.inf)
    np.fill_diagonal(goal_distances, np.inf)
    return bool(
        np.min(start_distances) < min_pair_distance
        or np.min(goal_distances) < min_pair_distance
    )


def _select_scen_group(
    valid: List[MovingAIScenarioTask],
    available: List[int],
    rng: np.random.Generator,
    num_agents: int,
    map_config: SimConfig,
    min_pair_distance: float,
) -> Optional[List[int]]:
    if len(available) < num_agents:
        return None
    ordered = list(rng.permutation(np.asarray(available, dtype=np.int64)))
    group_indices: List[int] = []
    for candidate_index in ordered:
        trial = group_indices + [int(candidate_index)]
        trial_group = [valid[index] for index in trial]
        if _group_has_overlap(trial_group, map_config, min_pair_distance):
            continue
        group_indices = trial
        if len(group_indices) == num_agents:
            return group_indices
    return None


def sample_scen_scenarios(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
    obstacle_map: GridMap,
) -> List[Scenario]:
    """Build multi-agent scenarios from standard Moving AI .scen tasks."""

    if dataset_config.scen_path is None:
        raise ValueError("scenario_source='scen' requires DatasetConfig.scen_path.")
    map_config = sim_config_for_scenario(
        sim_config,
        Scenario(
            starts=np.empty((0, 2), dtype=np.float64),
            goals=np.empty((0, 2), dtype=np.float64),
            radii=np.empty((0,), dtype=np.float64),
            world_size=obstacle_map.world_size,
            obstacle_map=obstacle_map,
        ),
    )
    raw_tasks = load_moving_ai_scen(
        dataset_config.scen_path,
        cell_size=obstacle_map.cell_size,
        limit=dataset_config.scen_limit,
    )
    valid = [
        task
        for task in raw_tasks
        if _valid_scen_task(task, map_config, dataset_config, obstacle_map)
    ]
    scenarios: List[Scenario] = []
    min_pair_distance = 2.5 * map_config.agent_radius + map_config.safety_margin
    rng = np.random.default_rng(dataset_config.seed)
    available = list(range(len(valid)))
    overlap_skips = 0
    while len(scenarios) < dataset_config.num_scenarios:
        group_indices = _select_scen_group(
            valid,
            available,
            rng,
            dataset_config.num_agents,
            map_config,
            min_pair_distance,
        )
        if group_indices is None:
            break
        group = [valid[index] for index in group_indices]
        for index in group_indices:
            available.remove(index)
        starts = np.stack([task.start for task in group], axis=0)
        goals = np.stack([task.goal for task in group], axis=0)
        radii = np.full(dataset_config.num_agents, map_config.agent_radius, dtype=np.float64)
        if _group_has_overlap(group, map_config, min_pair_distance):
            overlap_skips += 1
            continue
        scenarios.append(
            Scenario(
                starts=starts,
                goals=goals,
                radii=radii,
                world_size=obstacle_map.world_size,
                obstacle_map=obstacle_map,
            )
        )
    if len(scenarios) != dataset_config.num_scenarios:
        diagnostics = scen_task_diagnostics(dataset_config, map_config, obstacle_map)
        skip_counts = dict(diagnostics.get("skip_counts", {}))
        skip_counts["grouping_overlap"] = skip_counts.get("grouping_overlap", 0) + overlap_skips
        if len(valid) < dataset_config.num_agents:
            skip_counts["insufficient_valid_tasks"] = 1
        raise RuntimeError(
            "Could not build enough valid scenarios from .scen tasks; "
            f"requested {dataset_config.num_scenarios}, built {len(scenarios)}, "
            f"valid_tasks={len(valid)}, raw_tasks={len(raw_tasks)}, "
            f"skip_counts={skip_counts}."
        )
    return scenarios


def rollout_expert(
    scenario: Scenario,
    sim_config: SimConfig,
    horizon: int,
    expert_type: str = "independent_astar",
) -> List[dict]:
    """Roll out the phase 0 expert and return pre-step states plus targets."""

    world = ContinuousWorld(scenario, sim_config)
    records: List[dict] = []
    for _ in range(horizon):
        scenario_config = sim_config_for_scenario(sim_config, scenario)
        if scenario.obstacle_map is not None:
            target_velocity = obstacle_map_expert_velocity(
                world.positions,
                world.goals,
                scenario_config,
                scenario.obstacle_map,
                radii=world.radii,
                expert_type=expert_type,
            )
        elif scenario.static_obstacles:
            target_velocity = obstacle_aware_velocity(
                world.positions,
                world.goals,
                scenario_config,
                scenario.static_obstacles,
            )
        else:
            target_velocity = straight_line_velocity(
                world.positions,
                world.goals,
                scenario_config,
            )
        records.append(
            {
                "positions": world.positions.copy(),
                "velocities": world.velocities.copy(),
                "goals": world.goals.copy(),
                "radii": world.radii.copy(),
                "target_velocities": target_velocity.copy(),
                "reached": world.reached_goals().copy(),
            }
        )
        world.step(target_velocity)
        if world.all_reached():
            break
    return records


def generate_scenarios(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
) -> List[Scenario]:
    """Generate phase 0 scenarios."""

    rng = np.random.default_rng(dataset_config.seed)
    scenarios = []
    count_choices = tuple(int(count) for count in dataset_config.agent_count_choices)
    obstacle_map = load_obstacle_map(
        dataset_config.map_path,
        cell_size=dataset_config.map_cell_size,
    )
    scenario_type = dataset_config.scenario_type.strip().lower().replace("-", "_")
    scenario_source = dataset_config.scenario_source.strip().lower().replace("-", "_")
    if scenario_type in {"moving_ai", "moving_ai_map", "map"}:
        scenario_type = "obstacle_map"
    if scenario_type == "obstacle_map" and obstacle_map is None:
        raise ValueError("scenario_type='obstacle_map' requires DatasetConfig.map_path.")
    if scenario_type == "empty" and obstacle_map is not None:
        scenario_type = "obstacle_map"
    if scenario_source == "scen":
        if scenario_type != "obstacle_map" or obstacle_map is None:
            raise ValueError("scenario_source='scen' requires an obstacle_map and map_path.")
        scenarios = sample_scen_scenarios(dataset_config, sim_config, obstacle_map)
    elif scenario_source != "sampled":
        raise ValueError(
            f"Unsupported scenario_source={dataset_config.scenario_source!r}; "
            "expected 'sampled' or 'scen'."
        )

    for _ in range(0 if scenarios else dataset_config.num_scenarios):
        if scenario_type not in {"empty", "obstacle_map"}:
            raise ValueError(
                f"Unsupported scenario_type={dataset_config.scenario_type!r}; "
                "expected 'empty' or 'obstacle_map'."
            )
        num_agents = (
            int(rng.choice(count_choices))
            if count_choices
            else dataset_config.num_agents
        )
        if scenario_type == "obstacle_map":
            assert obstacle_map is not None
            scenario = sample_obstacle_map_scenario(
                rng,
                num_agents,
                sim_config,
                dataset_config,
                obstacle_map,
            )
        else:
            scenario = sample_empty_scenario(
                rng,
                num_agents,
                sim_config,
                dataset_config,
            )
        if collision_pairs(scenario.starts, scenario.radii):
            raise RuntimeError("Generated an initially colliding scenario.")
        if scenario.obstacle_map is not None:
            obstacle_collisions = scenario.obstacle_map.circle_collisions(
                scenario.starts,
                scenario.radii,
                margin=sim_config.safety_margin,
            )
            if obstacle_collisions:
                raise RuntimeError("Generated an initially obstacle-colliding scenario.")
        scenario = Scenario(
            starts=project_positions_to_bounds(
                scenario.starts,
                scenario.radii,
                scenario.world_size,
            ),
            goals=project_positions_to_bounds(
                scenario.goals,
                scenario.radii,
                scenario.world_size,
            ),
            radii=scenario.radii,
            world_size=scenario.world_size,
            static_obstacles=scenario.static_obstacles,
            obstacle_map=scenario.obstacle_map,
        )
        scenarios.append(scenario)
    return scenarios
