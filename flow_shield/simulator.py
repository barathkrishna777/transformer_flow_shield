"""Continuous 2D simulator for circular multi-agent path finding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .config import SimConfig
from .geometry import clip_by_norm, collision_pairs, project_positions_to_bounds, separation_violation
from .maps import GridMap, map_metadata


@dataclass(frozen=True)
class Scenario:
    """A phase 0/1 MAPF instance."""

    starts: np.ndarray
    goals: np.ndarray
    radii: np.ndarray
    world_size: Tuple[float, float]
    static_obstacles: Tuple[Tuple[float, float, float], ...] = ()
    obstacle_map: Optional[GridMap] = None

    @property
    def num_agents(self) -> int:
        return int(self.starts.shape[0])

    @property
    def has_obstacles(self) -> bool:
        return bool(self.static_obstacles) or self.obstacle_map is not None

    def map_metadata(self) -> Optional[Dict[str, object]]:
        return map_metadata(self.obstacle_map)

    def copy(self) -> "Scenario":
        return Scenario(
            starts=np.array(self.starts, dtype=np.float64, copy=True),
            goals=np.array(self.goals, dtype=np.float64, copy=True),
            radii=np.array(self.radii, dtype=np.float64, copy=True),
            world_size=tuple(self.world_size),
            static_obstacles=tuple(self.static_obstacles),
            obstacle_map=self.obstacle_map,
        )


def sim_config_for_scenario(config: SimConfig, scenario: Scenario) -> SimConfig:
    """Use scenario-local bounds when a map supplies the continuous world size."""

    if tuple(config.world_size) == tuple(scenario.world_size):
        return config
    return replace(config, world_size=tuple(scenario.world_size))


class ContinuousWorld:
    """A minimal continuous-space environment with circular agents."""

    def __init__(self, scenario: Scenario, config: SimConfig):
        self.scenario = scenario.copy()
        self.config = config
        self.positions = np.array(self.scenario.starts, dtype=np.float64, copy=True)
        self.velocities = np.zeros_like(self.positions)
        self.goals = np.array(self.scenario.goals, dtype=np.float64, copy=True)
        self.radii = np.array(self.scenario.radii, dtype=np.float64, copy=True)
        self.obstacle_map = self.scenario.obstacle_map
        self.step_count = 0

    @property
    def num_agents(self) -> int:
        return self.positions.shape[0]

    def reset(self) -> None:
        self.positions = np.array(self.scenario.starts, dtype=np.float64, copy=True)
        self.velocities = np.zeros_like(self.positions)
        self.step_count = 0

    def reached_goals(self) -> np.ndarray:
        return np.linalg.norm(self.goals - self.positions, axis=1) <= self.config.goal_tolerance

    def all_reached(self) -> bool:
        return bool(np.all(self.reached_goals()))

    def collision_pairs(self, margin: float = 0.0) -> Tuple[Tuple[int, int], ...]:
        return collision_pairs(self.positions, self.radii, margin=margin)

    def step(self, commanded_velocities: np.ndarray) -> Dict[str, object]:
        """Advance the world by one step using velocity commands."""

        commanded_velocities = np.asarray(commanded_velocities, dtype=np.float64)
        clipped = clip_by_norm(commanded_velocities, self.config.max_speed)
        next_positions = self.positions + clipped * self.config.dt
        next_positions = project_positions_to_bounds(
            next_positions,
            self.radii,
            self.scenario.world_size,
        )
        obstacle_hits: Tuple[int, ...] = ()
        obstacle_penetration = 0.0
        if self.obstacle_map is not None:
            next_positions, obstacle_hits, obstacle_penetration = (
                self.obstacle_map.constrain_positions(
                    self.positions,
                    next_positions,
                    self.radii,
                    margin=0.0,
                )
            )
        actual_velocities = (next_positions - self.positions) / self.config.dt
        self.positions = next_positions
        self.velocities = actual_velocities
        self.step_count += 1
        pairs = self.collision_pairs()
        obstacle_collisions = self.obstacle_collisions()
        return {
            "step": self.step_count,
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "collision_pairs": pairs,
            "obstacle_collisions": obstacle_collisions,
            "obstacle_motion_hits": obstacle_hits,
            "obstacle_motion_penetration": float(obstacle_penetration),
            "obstacle_separation_violation": self.obstacle_separation_violation(
                margin=self.config.safety_margin
            ),
            "reached": self.reached_goals().copy(),
        }

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "goals": self.goals.copy(),
            "radii": self.radii.copy(),
        }

    def obstacle_collisions(self, margin: float = 0.0) -> Tuple[Tuple[int, int, int], ...]:
        if self.obstacle_map is None:
            return ()
        return self.obstacle_map.circle_collisions(self.positions, self.radii, margin=margin)

    def obstacle_separation_violation(self, margin: float = 0.0) -> float:
        if self.obstacle_map is None:
            return 0.0
        return self.obstacle_map.max_penetration(self.positions, self.radii, margin=margin)


def stack_trajectories(records: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convert a sequence of simulator records into dense arrays."""

    materialized = list(records)
    if not materialized:
        return {
            "positions": np.empty((0, 0, 2), dtype=np.float64),
            "velocities": np.empty((0, 0, 2), dtype=np.float64),
            "reached": np.empty((0, 0), dtype=bool),
            "obstacle_separation_violation": np.empty((0,), dtype=np.float64),
        }
    return {
        "positions": np.stack([record["positions"] for record in materialized], axis=0),
        "velocities": np.stack([record["velocities"] for record in materialized], axis=0),
        "reached": np.stack([record["reached"] for record in materialized], axis=0),
        "obstacle_separation_violation": np.asarray(
            [
                float(record.get("obstacle_separation_violation", 0.0))
                for record in materialized
            ],
            dtype=np.float64,
        ),
    }


def rollout(
    scenario: Scenario,
    config: SimConfig,
    policy,
    shield: Optional[object] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    """Run a closed-loop rollout for a policy and optional shield."""

    world = ContinuousWorld(scenario, config)
    records = []
    shield_diagnostics = []
    shield_mean_corrections = []
    shield_max_corrections = []
    shield_correction_needed = []
    shield_obstacle_interventions = []
    shield_pairwise_interventions = []
    separation_violations = []
    obstacle_separation_violations = []
    collision_count = 0
    pair_collision_count = 0
    obstacle_collision_count = 0
    obstacle_collision_steps = 0
    obstacle_motion_hit_count = 0
    first_success_step = None
    steps = max_steps if max_steps is not None else config.max_steps
    start_time = perf_counter()
    progress_window = min(20, max(2, steps // 4 if steps else 2))

    for _ in range(steps):
        commands = policy(world.positions, world.velocities, world.goals, world.radii)
        if shield is not None:
            commands, diagnostics = shield.apply(
                positions=world.positions,
                velocities=commands,
                goals=world.goals,
                radii=world.radii,
                obstacle_map=world.obstacle_map,
            )
            if hasattr(diagnostics, "to_dict"):
                shield_diagnostics.append(diagnostics.to_dict())
            shield_mean_corrections.append(
                float(getattr(diagnostics, "mean_correction_norm", 0.0))
            )
            max_correction = float(getattr(diagnostics, "max_correction_norm", 0.0))
            shield_max_corrections.append(max_correction)
            initial_obstacle = int(getattr(diagnostics, "initial_obstacle_conflicts", 0))
            initial_total = int(getattr(diagnostics, "initial_conflicts", 0))
            obstacle_blocked = int(getattr(diagnostics, "obstacle_blocked_agents", 0))
            shield_correction_needed.append(float(max_correction > 1e-8))
            shield_obstacle_interventions.append(
                float(initial_obstacle > 0 or obstacle_blocked > 0)
            )
            shield_pairwise_interventions.append(
                float(max(0, initial_total - initial_obstacle) > 0)
            )
        record = world.step(commands)
        records.append(record)
        separation_violations.append(
            separation_violation(
                record["positions"],
                world.radii,
                margin=config.safety_margin,
            )
        )
        pairs = record["collision_pairs"]
        if pairs:
            collision_count += 1
            pair_collision_count += len(pairs)
        obstacle_collisions = record.get("obstacle_collisions", ())
        if obstacle_collisions:
            obstacle_collision_steps += 1
            obstacle_collision_count += len(obstacle_collisions)
        obstacle_motion_hit_count += len(record.get("obstacle_motion_hits", ()))
        obstacle_separation_violations.append(
            float(record.get("obstacle_separation_violation", 0.0))
        )
        if first_success_step is None and bool(np.all(record["reached"])):
            first_success_step = int(record["step"])
            break

    trajectory = stack_trajectories(records)
    final_positions = trajectory["positions"][-1] if len(records) else world.positions
    final_distances = np.linalg.norm(world.goals - final_positions, axis=1)
    fraction_agents_within_goal_tolerance = float(
        np.mean(final_distances <= config.goal_tolerance)
    )
    if trajectory["positions"].shape[0] > progress_window:
        recent = trajectory["positions"][-progress_window:]
        recent_displacement = np.linalg.norm(recent[-1] - recent[0], axis=1)
        unfinished = final_distances > config.goal_tolerance
        no_progress_agents = recent_displacement < config.goal_tolerance * 0.25
        no_progress_deadlock = bool(np.any(no_progress_agents & unfinished))
        mean_recent_progress = float(np.mean(recent_displacement))
    else:
        no_progress_deadlock = False
        mean_recent_progress = 0.0
    failure_flags = {
        "max_step_or_horizon": first_success_step is None,
        "deadlock_no_progress": bool(no_progress_deadlock),
        "pairwise_collision": pair_collision_count > 0,
        "obstacle_collision": obstacle_collision_count > 0,
        "obstacle_motion_intervention": obstacle_motion_hit_count > 0,
        "shield_obstacle_intervention": bool(
            np.any(shield_obstacle_interventions) if shield_obstacle_interventions else False
        ),
        "shield_pairwise_intervention": bool(
            np.any(shield_pairwise_interventions) if shield_pairwise_interventions else False
        ),
    }
    if first_success_step is not None:
        termination_reason = "reached_all_goals"
    elif no_progress_deadlock:
        termination_reason = "deadlock_no_progress"
    elif pair_collision_count > 0:
        termination_reason = "pairwise_collision"
    elif obstacle_collision_count > 0 or obstacle_motion_hit_count > 0:
        termination_reason = "obstacle_collision_or_intervention"
    else:
        termination_reason = "max_step_or_horizon"
    wall_time_seconds = max(perf_counter() - start_time, 1e-12)
    agent_steps = len(records) * scenario.num_agents
    return {
        "trajectory": trajectory,
        "collision_steps": collision_count,
        "pair_collisions": pair_collision_count,
        "obstacle_collision_steps": obstacle_collision_steps,
        "obstacle_collisions": obstacle_collision_count,
        "obstacle_motion_hits": obstacle_motion_hit_count,
        "success": first_success_step is not None,
        "termination_reason": termination_reason,
        "failure_flags": failure_flags,
        "time_to_goal": (
            first_success_step * config.dt
            if first_success_step is not None
            else steps * config.dt
        ),
        "steps": len(records),
        "mean_final_distance_to_goal": float(np.mean(final_distances)),
        "max_final_distance_to_goal": float(np.max(final_distances)),
        "fraction_agents_within_goal_tolerance": fraction_agents_within_goal_tolerance,
        "no_progress_deadlock": bool(no_progress_deadlock),
        "mean_recent_progress": mean_recent_progress,
        "mean_min_separation_violation": (
            float(np.mean(separation_violations)) if separation_violations else 0.0
        ),
        "max_min_separation_violation": (
            float(np.max(separation_violations)) if separation_violations else 0.0
        ),
        "mean_obstacle_separation_violation": (
            float(np.mean(obstacle_separation_violations))
            if obstacle_separation_violations
            else 0.0
        ),
        "max_obstacle_separation_violation": (
            float(np.max(obstacle_separation_violations))
            if obstacle_separation_violations
            else 0.0
        ),
        "mean_shield_correction_norm": (
            float(np.mean(shield_mean_corrections)) if shield_mean_corrections else 0.0
        ),
        "max_shield_correction_norm": (
            float(np.max(shield_max_corrections)) if shield_max_corrections else 0.0
        ),
        "correction_needed_rate": (
            float(np.mean(shield_correction_needed)) if shield_correction_needed else 0.0
        ),
        "mean_correction_target_norm": (
            float(np.mean(shield_mean_corrections)) if shield_mean_corrections else 0.0
        ),
        "max_correction_target_norm": (
            float(np.max(shield_max_corrections)) if shield_max_corrections else 0.0
        ),
        "obstacle_intervention_rate": (
            float(np.mean(shield_obstacle_interventions))
            if shield_obstacle_interventions
            else 0.0
        ),
        "pairwise_intervention_rate": (
            float(np.mean(shield_pairwise_interventions))
            if shield_pairwise_interventions
            else 0.0
        ),
        "shield_diagnostics": shield_diagnostics,
        "wall_time_seconds": float(wall_time_seconds),
        "steps_per_second": float(len(records) / wall_time_seconds),
        "agents_per_second": float(agent_steps / wall_time_seconds),
        "num_agents": int(scenario.num_agents),
    }
