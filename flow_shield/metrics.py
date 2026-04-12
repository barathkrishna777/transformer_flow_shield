"""Evaluation metrics for phase 1/2 experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def smoothness(velocities: np.ndarray) -> float:
    if velocities.shape[0] < 2:
        return 0.0
    accelerations = np.diff(velocities, axis=0)
    return float(np.mean(np.linalg.norm(accelerations, axis=-1)))


def deadlock_detected(positions: np.ndarray, goals: np.ndarray, tolerance: float, window: int = 20) -> bool:
    if positions.shape[0] <= window:
        return False
    recent = positions[-window:]
    displacement = np.linalg.norm(recent[-1] - recent[0], axis=1)
    remaining = np.linalg.norm(goals - recent[-1], axis=1)
    stuck = displacement < tolerance * 0.25
    unfinished = remaining > tolerance
    return bool(np.any(stuck & unfinished))


def aggregate_rollouts(rollouts: Iterable[Dict[str, object]], goals: List[np.ndarray], tolerance: float) -> Dict[str, float]:
    materialized = list(rollouts)
    if not materialized:
        return {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "pair_collisions_per_run": 0.0,
            "obstacle_collision_rate": 0.0,
            "obstacle_collisions_per_run": 0.0,
            "mean_min_separation_violation": 0.0,
            "max_min_separation_violation": 0.0,
            "mean_obstacle_separation_violation": 0.0,
            "max_obstacle_separation_violation": 0.0,
            "mean_shield_correction_norm": 0.0,
            "max_shield_correction_norm": 0.0,
            "mean_time_to_goal": 0.0,
            "mean_smoothness": 0.0,
            "deadlock_rate": 0.0,
            "steps_per_second": 0.0,
            "agents_per_second": 0.0,
            "mean_agents_per_run": 0.0,
        }
    successes = [float(result["success"]) for result in materialized]
    total_steps = sum(int(result["steps"]) for result in materialized)
    total_wall_time = sum(float(result.get("wall_time_seconds", 0.0)) for result in materialized)
    total_agent_steps = sum(
        int(result["steps"]) * int(result.get("num_agents", result["trajectory"]["positions"].shape[1]))
        for result in materialized
    )
    collision_steps = sum(int(result["collision_steps"]) for result in materialized)
    pair_collisions = [float(result["pair_collisions"]) for result in materialized]
    obstacle_collision_steps = sum(
        int(result.get("obstacle_collision_steps", 0)) for result in materialized
    )
    obstacle_collisions = [
        float(result.get("obstacle_collisions", 0.0)) for result in materialized
    ]
    times = [float(result["time_to_goal"]) for result in materialized]
    smooth = [smoothness(result["trajectory"]["velocities"]) for result in materialized]
    mean_separation_violations = [
        float(result.get("mean_min_separation_violation", 0.0))
        for result in materialized
    ]
    max_separation_violations = [
        float(result.get("max_min_separation_violation", 0.0))
        for result in materialized
    ]
    mean_obstacle_violations = [
        float(result.get("mean_obstacle_separation_violation", 0.0))
        for result in materialized
    ]
    max_obstacle_violations = [
        float(result.get("max_obstacle_separation_violation", 0.0))
        for result in materialized
    ]
    mean_shield_corrections = [
        float(result.get("mean_shield_correction_norm", 0.0))
        for result in materialized
    ]
    max_shield_corrections = [
        float(result.get("max_shield_correction_norm", 0.0))
        for result in materialized
    ]
    deadlocks = []
    for result, goal_array in zip(materialized, goals):
        trajectory = result["trajectory"]
        deadlocks.append(
            float(
                deadlock_detected(
                    trajectory["positions"],
                    goal_array,
                    tolerance=tolerance,
                )
            )
        )
    return {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(collision_steps / max(total_steps, 1)),
        "pair_collisions_per_run": float(np.mean(pair_collisions)),
        "obstacle_collision_rate": float(obstacle_collision_steps / max(total_steps, 1)),
        "obstacle_collisions_per_run": float(np.mean(obstacle_collisions)),
        "mean_min_separation_violation": float(np.mean(mean_separation_violations)),
        "max_min_separation_violation": float(np.max(max_separation_violations)),
        "mean_obstacle_separation_violation": float(np.mean(mean_obstacle_violations)),
        "max_obstacle_separation_violation": float(np.max(max_obstacle_violations)),
        "mean_shield_correction_norm": float(np.mean(mean_shield_corrections)),
        "max_shield_correction_norm": float(np.max(max_shield_corrections)),
        "mean_time_to_goal": float(np.mean(times)),
        "mean_smoothness": float(np.mean(smooth)),
        "deadlock_rate": float(np.mean(deadlocks)),
        "steps_per_second": float(total_steps / max(total_wall_time, 1e-12)),
        "agents_per_second": float(total_agent_steps / max(total_wall_time, 1e-12)),
        "mean_agents_per_run": float(
            np.mean([float(result.get("num_agents", result["trajectory"]["positions"].shape[1])) for result in materialized])
        ),
    }
