"""Collision shield variants for learned continuous-space MAPF intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import SimConfig
from .geometry import (
    clip_by_norm,
    collision_pairs,
    project_positions_to_bounds,
    stable_unit_vector,
    total_separation_violation,
)
from .maps import GridMap


@dataclass(frozen=True)
class ShieldDiagnostics:
    iterations: int = 0
    initial_conflicts: int = 0
    final_conflicts: int = 0
    mean_correction_norm: float = 0.0
    max_correction_norm: float = 0.0
    clipped_agents: int = 0
    inherited_priorities: int = 0
    backtracks: int = 0
    initial_obstacle_conflicts: int = 0
    final_obstacle_conflicts: int = 0
    obstacle_blocked_agents: int = 0
    obstacle_max_penetration: float = 0.0
    variant: str = "unknown"
    resolved: bool = True
    limited: bool = False

    def to_dict(self) -> Dict[str, float | int | str | bool]:
        return {
            "iterations": self.iterations,
            "initial_conflicts": self.initial_conflicts,
            "final_conflicts": self.final_conflicts,
            "mean_correction_norm": self.mean_correction_norm,
            "max_correction_norm": self.max_correction_norm,
            "clipped_agents": self.clipped_agents,
            "inherited_priorities": self.inherited_priorities,
            "backtracks": self.backtracks,
            "initial_obstacle_conflicts": self.initial_obstacle_conflicts,
            "final_obstacle_conflicts": self.final_obstacle_conflicts,
            "obstacle_blocked_agents": self.obstacle_blocked_agents,
            "obstacle_max_penetration": self.obstacle_max_penetration,
            "variant": self.variant,
            "resolved": self.resolved,
            "limited": self.limited,
        }


def _correction_stats(updated: np.ndarray, original: np.ndarray) -> Tuple[float, float]:
    norms = np.linalg.norm(updated - original, axis=1)
    if norms.size == 0:
        return 0.0, 0.0
    return float(np.mean(norms)), float(np.max(norms))


def _changed_agent_count(updated: np.ndarray, original: np.ndarray) -> int:
    if updated.size == 0:
        return 0
    return int(np.sum(np.linalg.norm(updated - original, axis=1) > 1e-8))


class BaseShield:
    """Base API for one-step continuous collision shields."""

    name = "base"

    def __init__(self, config: SimConfig):
        self.config = config

    def priority_scores(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        external_priorities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if external_priorities is not None:
            return np.asarray(external_priorities, dtype=np.float64)
        remaining = np.linalg.norm(goals - positions, axis=1)
        tie_break = np.linspace(0.0, 1e-6, positions.shape[0], dtype=np.float64)
        return remaining + tie_break

    def predicted_positions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> np.ndarray:
        clipped = clip_by_norm(velocities, self.config.max_speed)
        next_positions = positions + clipped * self.config.dt
        world_size = self._world_size(obstacle_map)
        next_positions = project_positions_to_bounds(next_positions, radii, world_size)
        if obstacle_map is not None:
            next_positions, _, _ = obstacle_map.constrain_positions(
                positions,
                next_positions,
                radii,
                margin=self.config.safety_margin,
            )
        return next_positions

    def raw_predicted_positions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> np.ndarray:
        clipped = clip_by_norm(velocities, self.config.max_speed)
        next_positions = positions + clipped * self.config.dt
        return project_positions_to_bounds(next_positions, radii, self._world_size(obstacle_map))

    def _world_size(self, obstacle_map: Optional[GridMap] = None) -> Tuple[float, float]:
        if obstacle_map is not None:
            return obstacle_map.world_size
        return self.config.world_size

    def predicted_collision_pairs(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        margin: Optional[float] = None,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        safety_margin = self.config.safety_margin if margin is None else margin
        return collision_pairs(
            self.predicted_positions(positions, velocities, radii, obstacle_map=obstacle_map),
            radii,
            margin=safety_margin,
        )

    def predicted_obstacle_collisions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
        margin: Optional[float] = None,
    ) -> Tuple[Tuple[int, int, int], ...]:
        if obstacle_map is None:
            return ()
        safety_margin = self.config.safety_margin if margin is None else margin
        next_positions = self.raw_predicted_positions(
            positions,
            velocities,
            radii,
            obstacle_map=obstacle_map,
        )
        return obstacle_map.circle_collisions(next_positions, radii, margin=safety_margin)

    def _respect_bounds_and_obstacles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int, float]:
        next_positions = positions + velocities * self.config.dt
        projected = project_positions_to_bounds(
            next_positions,
            radii,
            self._world_size(obstacle_map),
        )
        blocked_agents: Tuple[int, ...] = ()
        max_penetration = 0.0
        if obstacle_map is not None:
            projected, blocked_agents, max_penetration = obstacle_map.constrain_positions(
                positions,
                projected,
                radii,
                margin=self.config.safety_margin,
            )
        updated = (projected - positions) / max(self.config.dt, 1e-8)
        return updated, len(blocked_agents), float(max_penetration)

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        priorities: Optional[np.ndarray] = None,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        raise NotImplementedError

    def _diagnostics(
        self,
        original: np.ndarray,
        updated: np.ndarray,
        initial_conflicts: int,
        final_conflicts: int,
        iterations: int = 0,
        clipped_agents: int = 0,
        inherited_priorities: int = 0,
        backtracks: int = 0,
        initial_obstacle_conflicts: int = 0,
        final_obstacle_conflicts: int = 0,
        obstacle_blocked_agents: int = 0,
        obstacle_max_penetration: float = 0.0,
    ) -> ShieldDiagnostics:
        mean_correction, max_correction = _correction_stats(updated, original)
        return ShieldDiagnostics(
            iterations=iterations,
            initial_conflicts=initial_conflicts,
            final_conflicts=final_conflicts,
            mean_correction_norm=mean_correction,
            max_correction_norm=max_correction,
            clipped_agents=clipped_agents,
            inherited_priorities=inherited_priorities,
            backtracks=backtracks,
            initial_obstacle_conflicts=initial_obstacle_conflicts,
            final_obstacle_conflicts=final_obstacle_conflicts,
            obstacle_blocked_agents=obstacle_blocked_agents,
            obstacle_max_penetration=obstacle_max_penetration,
            variant=self.name,
            resolved=final_conflicts == 0,
            limited=final_conflicts > 0,
        )


class NoShield(BaseShield):
    """Explicit no-op baseline that still reports predicted conflicts."""

    name = "none"

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        priorities: Optional[np.ndarray] = None,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        del goals, priorities
        positions = np.asarray(positions, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        updated = np.asarray(velocities, dtype=np.float64).copy()
        pair_conflicts = len(
            self.predicted_collision_pairs(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        obstacle_conflicts = len(
            self.predicted_obstacle_collisions(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        conflicts = pair_conflicts + obstacle_conflicts
        return updated, self._diagnostics(
            updated,
            updated,
            initial_conflicts=conflicts,
            final_conflicts=conflicts,
            initial_obstacle_conflicts=obstacle_conflicts,
            final_obstacle_conflicts=obstacle_conflicts,
        )


class VelocityClipShield(BaseShield):
    """Only clip velocity commands to the simulator speed limit."""

    name = "velocity_clip"

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        priorities: Optional[np.ndarray] = None,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        del goals, priorities
        positions = np.asarray(positions, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        original = np.asarray(velocities, dtype=np.float64).copy()
        updated = clip_by_norm(original, self.config.max_speed)
        initial_pair_conflicts = len(
            self.predicted_collision_pairs(
                positions,
                original,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        initial_obstacle_conflicts = len(
            self.predicted_obstacle_collisions(
                positions,
                original,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        updated, obstacle_blocked_agents, obstacle_penetration = (
            self._respect_bounds_and_obstacles(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        final_pair_conflicts = len(
            self.predicted_collision_pairs(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        final_obstacle_conflicts = len(
            self.predicted_obstacle_collisions(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        return updated, self._diagnostics(
            original,
            updated,
            initial_conflicts=initial_pair_conflicts + initial_obstacle_conflicts,
            final_conflicts=final_pair_conflicts + final_obstacle_conflicts,
            clipped_agents=_changed_agent_count(updated, original),
            initial_obstacle_conflicts=initial_obstacle_conflicts,
            final_obstacle_conflicts=final_obstacle_conflicts,
            obstacle_blocked_agents=obstacle_blocked_agents,
            obstacle_max_penetration=obstacle_penetration,
        )


class IterativeProjectionShield(BaseShield):
    """Predict one step ahead and project conflicting pairs apart."""

    name = "iterative_projection"

    def __init__(
        self,
        config: SimConfig,
        mode: str = "priority",
        max_iterations: int = 8,
        damping: float = 1.0,
    ):
        if mode not in {"priority", "pairwise"}:
            raise ValueError("mode must be 'priority' or 'pairwise'.")
        super().__init__(config)
        self.mode = mode
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        priorities: Optional[np.ndarray] = None,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, ShieldDiagnostics]:
        positions = np.asarray(positions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        original = np.asarray(velocities, dtype=np.float64).copy()
        velocities = clip_by_norm(original, self.config.max_speed)
        clipped_agents = _changed_agent_count(velocities, original)
        scores = self.priority_scores(positions, goals, priorities)
        scores, inherited = self._effective_priority_scores(
            positions,
            velocities,
            goals,
            radii,
            scores,
            obstacle_map,
        )

        initial_pair_conflicts = len(
            self.predicted_collision_pairs(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        initial_obstacle_conflicts = len(
            self.predicted_obstacle_collisions(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        obstacle_blocked_agents = 0
        obstacle_max_penetration = 0.0
        velocities, blocked, penetration = self._respect_bounds_and_obstacles(
            positions,
            velocities,
            radii,
            obstacle_map=obstacle_map,
        )
        obstacle_blocked_agents += blocked
        obstacle_max_penetration = max(obstacle_max_penetration, penetration)
        velocities, iterations_used = self._project_velocities(
            positions,
            velocities,
            radii,
            scores,
            self.max_iterations,
            obstacle_map,
        )
        velocities, backtracks, extra_iterations, blocked, penetration = self._after_projection(
            positions,
            velocities,
            radii,
            scores,
            obstacle_map,
        )
        obstacle_blocked_agents += blocked
        obstacle_max_penetration = max(obstacle_max_penetration, penetration)
        iterations_used += extra_iterations
        final_pair_conflicts = len(
            self.predicted_collision_pairs(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        final_obstacle_conflicts = len(
            self.predicted_obstacle_collisions(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
        )
        diagnostics = self._diagnostics(
            original,
            velocities,
            initial_conflicts=initial_pair_conflicts + initial_obstacle_conflicts,
            final_conflicts=final_pair_conflicts + final_obstacle_conflicts,
            iterations=iterations_used,
            clipped_agents=clipped_agents,
            inherited_priorities=inherited,
            backtracks=backtracks,
            initial_obstacle_conflicts=initial_obstacle_conflicts,
            final_obstacle_conflicts=final_obstacle_conflicts,
            obstacle_blocked_agents=obstacle_blocked_agents,
            obstacle_max_penetration=obstacle_max_penetration,
        )
        return velocities, diagnostics

    def _effective_priority_scores(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int]:
        del positions, velocities, goals, radii, obstacle_map
        return scores, 0

    def _after_projection(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int, int, int, float]:
        del positions, radii, scores, obstacle_map
        return velocities, 0, 0, 0, 0.0

    def _project_velocities(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        max_iterations: int,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int]:
        iterations_used = 0
        for iteration in range(max(0, int(max_iterations))):
            conflicts = self.predicted_collision_pairs(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
            if not conflicts:
                return velocities, iteration
            iterations_used = iteration + 1
            for i, j in conflicts:
                velocities = self._resolve_pair(positions, velocities, radii, scores, i, j)
            velocities, _, _ = self._respect_bounds_and_obstacles(
                positions,
                velocities,
                radii,
                obstacle_map=obstacle_map,
            )
            velocities = clip_by_norm(velocities, self.config.max_speed)
        return velocities, iterations_used

    def _resolve_pair(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        priority_scores: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        updated = velocities.copy()
        predicted = positions + updated * self.config.dt
        rel_next = predicted[i] - predicted[j]
        fallback = float((i + 1) * 1.61803398875 + (j + 1) * 0.61803398875)
        normal = stable_unit_vector(rel_next, fallback_angle=fallback)
        min_sep = radii[i] + radii[j] + self.config.safety_margin
        target_next = normal * min_sep
        delta_next = (target_next - rel_next) * self.damping
        delta_velocity_rel = delta_next / max(self.config.dt, 1e-8)

        if self.mode == "pairwise":
            updated[i] += 0.5 * delta_velocity_rel
            updated[j] -= 0.5 * delta_velocity_rel
        elif priority_scores[i] >= priority_scores[j]:
            updated[j] -= delta_velocity_rel
        else:
            updated[i] += delta_velocity_rel
        return updated

    def _respect_bounds(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        updated, _, _ = self._respect_bounds_and_obstacles(positions, velocities, radii)
        return updated


class PairwiseProjectionShield(IterativeProjectionShield):
    """Project both agents in a predicted conflict by equal and opposite edits."""

    name = "pairwise"

    def __init__(
        self,
        config: SimConfig,
        max_iterations: int = 8,
        damping: float = 1.0,
    ):
        super().__init__(
            config,
            mode="pairwise",
            max_iterations=max_iterations,
            damping=damping,
        )


class PriorityYieldingShield(IterativeProjectionShield):
    """Project lower-priority agents while higher-priority agents keep intent."""

    name = "priority"

    def __init__(
        self,
        config: SimConfig,
        max_iterations: int = 8,
        damping: float = 1.0,
    ):
        super().__init__(
            config,
            mode="priority",
            max_iterations=max_iterations,
            damping=damping,
        )


class PIBTInspiredShield(IterativeProjectionShield):
    """Single-step PIBT-inspired priority inheritance and local backtracking.

    This is intentionally a prototype: it does not reserve multi-step paths or
    prove liveness. It exposes unresolved ``final_conflicts`` in diagnostics
    rather than hiding cases where the local heuristic is too weak.
    """

    name = "pibt"

    def __init__(
        self,
        config: SimConfig,
        max_iterations: int = 8,
        damping: float = 1.0,
        max_backtracks: Optional[int] = None,
    ):
        super().__init__(
            config,
            mode="priority",
            max_iterations=max_iterations,
            damping=damping,
        )
        self.max_backtracks = max_backtracks
        self.inheritance_gap = 1e-4

    def _effective_priority_scores(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int]:
        del goals
        inherited = np.asarray(scores, dtype=np.float64).copy()
        events = 0
        conflicts = self.predicted_collision_pairs(
            positions,
            velocities,
            radii,
            obstacle_map=obstacle_map,
        )
        for _ in range(max(1, positions.shape[0])):
            changed = False
            for i, j in conflicts:
                if inherited[i] >= inherited[j]:
                    high, low = i, j
                else:
                    high, low = j, i
                target = inherited[high] - self.inheritance_gap
                if inherited[low] + 1e-9 < target:
                    inherited[low] = target
                    events += 1
                    changed = True
            if not changed:
                break
        return inherited, events

    def _after_projection(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int, int, int, float]:
        if not self.predicted_collision_pairs(
            positions,
            velocities,
            radii,
            obstacle_map=obstacle_map,
        ):
            return velocities, 0, 0, 0, 0.0
        velocities, backtracks = self._backtrack_conflicts(
            positions,
            velocities,
            radii,
            scores,
            obstacle_map,
        )
        if backtracks == 0:
            return velocities, 0, 0, 0, 0.0
        extra_budget = max(1, self.max_iterations // 2)
        velocities, extra_iterations = self._project_velocities(
            positions,
            velocities,
            radii,
            scores,
            extra_budget,
            obstacle_map,
        )
        velocities, blocked, penetration = self._respect_bounds_and_obstacles(
            positions,
            velocities,
            radii,
            obstacle_map=obstacle_map,
        )
        return velocities, backtracks, extra_iterations, blocked, penetration

    def _backtrack_conflicts(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        scores: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> Tuple[np.ndarray, int]:
        updated = velocities.copy()
        budget = self.max_backtracks
        if budget is None:
            budget = max(1, positions.shape[0] * 2)
        backtracks = 0
        for _ in range(int(budget)):
            conflicts = self.predicted_collision_pairs(
                positions,
                updated,
                radii,
                obstacle_map=obstacle_map,
            )
            if not conflicts:
                break
            accepted = False
            for i, j in conflicts:
                lower = j if scores[i] >= scores[j] else i
                for agents in ((lower,), (i, j)):
                    candidate = updated.copy()
                    candidate[list(agents)] = 0.0
                    if self._candidate_improves(
                        positions,
                        candidate,
                        updated,
                        radii,
                        obstacle_map,
                    ):
                        updated = candidate
                        backtracks += 1
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break
        return updated, backtracks

    def _candidate_improves(
        self,
        positions: np.ndarray,
        candidate: np.ndarray,
        current: np.ndarray,
        radii: np.ndarray,
        obstacle_map: Optional[GridMap] = None,
    ) -> bool:
        candidate_pairs = self.predicted_collision_pairs(
            positions,
            candidate,
            radii,
            obstacle_map=obstacle_map,
        )
        current_pairs = self.predicted_collision_pairs(
            positions,
            current,
            radii,
            obstacle_map=obstacle_map,
        )
        if len(candidate_pairs) < len(current_pairs):
            return True
        if len(candidate_pairs) > len(current_pairs):
            return False
        candidate_violation = total_separation_violation(
            self.predicted_positions(positions, candidate, radii, obstacle_map=obstacle_map),
            radii,
            margin=self.config.safety_margin,
        )
        current_violation = total_separation_violation(
            self.predicted_positions(positions, current, radii, obstacle_map=obstacle_map),
            radii,
            margin=self.config.safety_margin,
        )
        return candidate_violation < current_violation - 1e-9


class CollisionShield(IterativeProjectionShield):
    """Backward-compatible Phase 1 shield wrapper.

    ``mode="pairwise"`` maps to :class:`PairwiseProjectionShield` behavior and
    ``mode="priority"`` maps to :class:`PriorityYieldingShield` behavior.
    """

    def __init__(
        self,
        config: SimConfig,
        mode: str = "priority",
        max_iterations: int = 8,
        damping: float = 1.0,
    ):
        super().__init__(
            config,
            mode=mode,
            max_iterations=max_iterations,
            damping=damping,
        )
        self.name = "pairwise" if mode == "pairwise" else "priority"


DEFAULT_SHIELD_VARIANTS = ("none", "velocity_clip", "pairwise", "priority", "pibt")

_SHIELD_ALIASES = {
    "none": "none",
    "no": "none",
    "no_shield": "none",
    "baseline": "none",
    "velocity": "velocity_clip",
    "velocity_clip": "velocity_clip",
    "velocity_clipping": "velocity_clip",
    "clip": "velocity_clip",
    "pairwise": "pairwise",
    "pairwise_projection": "pairwise",
    "projection": "pairwise",
    "priority": "priority",
    "priority_yielding": "priority",
    "yielding": "priority",
    "pibt": "pibt",
    "pibt_inspired": "pibt",
    "pibt_prototype": "pibt",
}


def canonical_shield_variant(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in _SHIELD_ALIASES:
        raise ValueError(
            f"Unknown shield variant {name!r}; expected one of {DEFAULT_SHIELD_VARIANTS}."
        )
    return _SHIELD_ALIASES[key]


def make_shield(
    name: str,
    config: SimConfig,
    max_iterations: int = 8,
    damping: float = 1.0,
) -> BaseShield:
    variant = canonical_shield_variant(name)
    if variant == "none":
        return NoShield(config)
    if variant == "velocity_clip":
        return VelocityClipShield(config)
    if variant == "pairwise":
        return PairwiseProjectionShield(
            config,
            max_iterations=max_iterations,
            damping=damping,
        )
    if variant == "priority":
        return PriorityYieldingShield(
            config,
            max_iterations=max_iterations,
            damping=damping,
        )
    if variant == "pibt":
        return PIBTInspiredShield(
            config,
            max_iterations=max_iterations,
            damping=damping,
        )
    raise AssertionError(f"Unhandled shield variant {variant!r}.")
