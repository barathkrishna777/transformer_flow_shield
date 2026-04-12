"""Observation encoding and dataset materialization for phase 0/1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import DatasetConfig, SimConfig
from .expert import generate_scenarios, rollout_expert
from .maps import GridMap, map_metadata
from .simulator import Scenario, sim_config_for_scenario


FEATURE_DIM = 10


@dataclass
class TrajectoryDataset:
    observations: np.ndarray
    masks: np.ndarray
    targets: np.ndarray
    dataset_config: Dict[str, object]
    sim_config: Dict[str, object]

    @property
    def num_samples(self) -> int:
        return int(self.observations.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            masks=self.masks,
            targets=self.targets,
            dataset_config=json.dumps(self.dataset_config),
            sim_config=json.dumps(self.sim_config),
        )


def load_dataset(path: str | Path) -> TrajectoryDataset:
    data = np.load(path, allow_pickle=False)
    return TrajectoryDataset(
        observations=data["observations"],
        masks=data["masks"],
        targets=data["targets"],
        dataset_config=json.loads(str(data["dataset_config"])),
        sim_config=json.loads(str(data["sim_config"])),
    )


def _world_scale(sim_config: SimConfig) -> float:
    return float(max(sim_config.world_size))


def encode_agent_observation(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    radii: np.ndarray,
    agent_index: int,
    max_neighbors: int,
    sim_config: SimConfig,
    obstacle_map: GridMap | None = None,
    max_obstacle_tokens: int = 0,
    obstacle_context_range: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode self and nearest-neighbor tokens for one agent."""

    n_agents = positions.shape[0]
    scale = _world_scale(sim_config)
    speed = max(sim_config.max_speed, 1e-8)
    obstacle_token_count = max(0, int(max_obstacle_tokens))
    token_count = max_neighbors + 1 + obstacle_token_count
    tokens = np.zeros((token_count, FEATURE_DIM), dtype=np.float64)
    mask = np.zeros(token_count, dtype=bool)

    def fill_token(slot: int, other: int, is_self: bool) -> None:
        rel_pos = (positions[other] - positions[agent_index]) / scale
        if is_self:
            rel_vel = velocities[agent_index] / speed
        else:
            rel_vel = (velocities[other] - velocities[agent_index]) / speed
        goal_delta = (goals[other] - positions[other]) / scale
        goal_distance = float(np.linalg.norm(goals[other] - positions[other]) / scale)
        neighbor_distance = float(np.linalg.norm(positions[other] - positions[agent_index]) / scale)
        tokens[slot] = np.array(
            [
                rel_pos[0],
                rel_pos[1],
                rel_vel[0],
                rel_vel[1],
                goal_delta[0],
                goal_delta[1],
                goal_distance,
                neighbor_distance,
                radii[other] / scale,
                1.0 if is_self else 0.0,
            ],
            dtype=np.float64,
        )
        mask[slot] = True

    fill_token(0, agent_index, True)
    if n_agents > 1:
        distances = np.linalg.norm(positions - positions[agent_index], axis=1)
        neighbor_order = np.argsort(distances)
        slot = 1
        neighbor_limit = max_neighbors + 1
        for other in neighbor_order:
            if other == agent_index:
                continue
            fill_token(slot, int(other), False)
            slot += 1
            if slot >= neighbor_limit:
                break
    if obstacle_map is not None and obstacle_token_count > 0:
        centers, clearances = obstacle_map.nearest_obstacle_tokens(
            positions[agent_index],
            obstacle_token_count,
            obstacle_context_range,
            float(radii[agent_index]),
            margin=sim_config.safety_margin,
        )
        start_slot = max_neighbors + 1
        for offset, (center, clearance) in enumerate(zip(centers, clearances)):
            slot = start_slot + offset
            if slot >= token_count:
                break
            rel_center = (center - positions[agent_index]) / scale
            center_distance = float(
                np.linalg.norm(center - positions[agent_index]) / scale
            )
            tokens[slot] = np.array(
                [
                    rel_center[0],
                    rel_center[1],
                    0.0,
                    0.0,
                    rel_center[0],
                    rel_center[1],
                    float(clearance / scale),
                    center_distance,
                    float(radii[agent_index] / scale),
                    -1.0,
                ],
                dtype=np.float64,
            )
            mask[slot] = True
    return tokens, mask


def encode_joint_observation(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    radii: np.ndarray,
    max_neighbors: int,
    sim_config: SimConfig,
    obstacle_map: GridMap | None = None,
    max_obstacle_tokens: int = 0,
    obstacle_context_range: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    observations = []
    masks = []
    for agent_index in range(positions.shape[0]):
        obs, mask = encode_agent_observation(
            positions,
            velocities,
            goals,
            radii,
            agent_index,
            max_neighbors,
            sim_config,
            obstacle_map=obstacle_map,
            max_obstacle_tokens=max_obstacle_tokens,
            obstacle_context_range=obstacle_context_range,
        )
        observations.append(obs)
        masks.append(mask)
    return np.stack(observations, axis=0), np.stack(masks, axis=0)


def build_dataset(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
) -> TrajectoryDataset:
    """Generate expert-supervised samples for phase 0/1."""

    observations: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    scenarios = generate_scenarios(dataset_config, sim_config)

    for scenario in scenarios:
        scenario_sim_config = sim_config_for_scenario(sim_config, scenario)
        for record in rollout_expert(scenario, sim_config, dataset_config.horizon):
            joint_obs, joint_mask = encode_joint_observation(
                record["positions"],
                record["velocities"],
                record["goals"],
                record["radii"],
                dataset_config.max_neighbors,
                scenario_sim_config,
                obstacle_map=scenario.obstacle_map,
                max_obstacle_tokens=dataset_config.max_obstacle_tokens,
                obstacle_context_range=dataset_config.obstacle_context_range,
            )
            for agent_index in range(scenario.num_agents):
                if (
                    not dataset_config.include_reached_agents
                    and bool(record["reached"][agent_index])
                ):
                    continue
                observations.append(joint_obs[agent_index])
                masks.append(joint_mask[agent_index])
                targets.append(record["target_velocities"][agent_index])
                if (
                    dataset_config.max_samples is not None
                    and len(observations) >= dataset_config.max_samples
                ):
                    break
            if (
                dataset_config.max_samples is not None
                and len(observations) >= dataset_config.max_samples
            ):
                break
        if (
            dataset_config.max_samples is not None
            and len(observations) >= dataset_config.max_samples
        ):
            break

    if not observations:
        raise RuntimeError("Dataset generation produced no samples.")

    return TrajectoryDataset(
        observations=np.asarray(observations, dtype=np.float64),
        masks=np.asarray(masks, dtype=bool),
        targets=np.asarray(targets, dtype=np.float64),
        dataset_config=dataset_config.to_dict(),
        sim_config=sim_config.to_dict(),
    )


def scenario_to_jsonable(scenario: Scenario) -> Dict[str, object]:
    return {
        "starts": scenario.starts.tolist(),
        "goals": scenario.goals.tolist(),
        "radii": scenario.radii.tolist(),
        "world_size": list(scenario.world_size),
        "static_obstacles": [list(obstacle) for obstacle in scenario.static_obstacles],
        "obstacle_map": map_metadata(scenario.obstacle_map),
    }
