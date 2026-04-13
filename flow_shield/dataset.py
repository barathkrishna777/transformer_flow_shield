"""Observation encoding and dataset materialization for phase 0/1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DatasetConfig, SimConfig
from .expert import astar_grid_path, generate_scenarios, rollout_expert
from .maps import GridMap, map_metadata
from .shield import make_shield
from .simulator import Scenario, sim_config_for_scenario


OBSERVATION_LEGACY = "legacy"
OBSERVATION_OBSTACLE_WAYPOINT_V2 = "obstacle_waypoint_v2"
FEATURE_DIM = 10
WAYPOINT_FEATURE_DIM = 18

FEATURE_NAMES = (
    "rel_pos_x",
    "rel_pos_y",
    "rel_vel_x",
    "rel_vel_y",
    "goal_delta_x",
    "goal_delta_y",
    "goal_distance",
    "neighbor_distance",
    "radius",
    "token_type",
)
WAYPOINT_FEATURE_NAMES = FEATURE_NAMES + (
    "next_waypoint_dir_x",
    "next_waypoint_dir_y",
    "second_waypoint_dir_x",
    "second_waypoint_dir_y",
    "remaining_path_length",
    "path_available",
    "direct_goal_dir_x",
    "direct_goal_dir_y",
)


def normalize_observation_version(version: str | None) -> str:
    if version is None:
        return OBSERVATION_LEGACY
    normalized = str(version).strip().lower().replace("-", "_")
    if normalized in {"", "legacy", "v1", "phase1", "obstacle_legacy"}:
        return OBSERVATION_LEGACY
    if normalized in {"obstacle_waypoint_v2", "waypoint_v2", "path_v2"}:
        return OBSERVATION_OBSTACLE_WAYPOINT_V2
    raise ValueError(
        f"Unsupported observation_version={version!r}; expected 'legacy' or "
        "'obstacle_waypoint_v2'."
    )


def observation_feature_dim(version: str | None) -> int:
    return WAYPOINT_FEATURE_DIM if normalize_observation_version(version) == OBSERVATION_OBSTACLE_WAYPOINT_V2 else FEATURE_DIM


def observation_metadata(version: str | None, max_neighbors: int, max_obstacle_tokens: int) -> Dict[str, object]:
    version = normalize_observation_version(version)
    return {
        "observation_version": version,
        "feature_dim": observation_feature_dim(version),
        "feature_names": (
            list(WAYPOINT_FEATURE_NAMES)
            if version == OBSERVATION_OBSTACLE_WAYPOINT_V2
            else list(FEATURE_NAMES)
        ),
        "token_layout": {
            "self_token": 0,
            "neighbor_tokens": [1, int(max_neighbors)],
            "obstacle_tokens": [
                int(max_neighbors) + 1,
                int(max_neighbors) + int(max_obstacle_tokens),
            ],
        },
    }


@dataclass
class TrajectoryDataset:
    observations: np.ndarray
    masks: np.ndarray
    targets: np.ndarray
    dataset_config: Dict[str, object]
    sim_config: Dict[str, object]
    auxiliary_targets: Optional[Dict[str, np.ndarray]] = None

    @property
    def num_samples(self) -> int:
        return int(self.observations.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "observations": self.observations,
            "masks": self.masks,
            "targets": self.targets,
            "dataset_config": np.array(json.dumps(self.dataset_config)),
            "sim_config": np.array(json.dumps(self.sim_config)),
            "observation_metadata": np.array(
                json.dumps(self.dataset_config.get("observation_metadata", {}))
            ),
        }
        if self.auxiliary_targets:
            arrays["auxiliary_target_names"] = np.array(
                json.dumps(sorted(self.auxiliary_targets.keys()))
            )
            for name, values in self.auxiliary_targets.items():
                arrays[f"aux_{name}"] = np.asarray(values)
        np.savez_compressed(path, **arrays)


def load_dataset(path: str | Path) -> TrajectoryDataset:
    data = np.load(path, allow_pickle=False)
    dataset_config = json.loads(str(data["dataset_config"]))
    if "observation_version" not in dataset_config:
        dataset_config["observation_version"] = OBSERVATION_LEGACY
    if "observation_metadata" not in dataset_config:
        dataset_config["observation_metadata"] = observation_metadata(
            dataset_config.get("observation_version"),
            int(dataset_config.get("max_neighbors", 0)),
            int(dataset_config.get("max_obstacle_tokens", 0)),
        )
    if "expert_type" not in dataset_config:
        dataset_config["expert_type"] = "independent_astar"
    if "include_auxiliary_targets" not in dataset_config:
        dataset_config["include_auxiliary_targets"] = False
    auxiliary_targets = None
    if "auxiliary_target_names" in data.files:
        auxiliary_targets = {}
        for name in json.loads(str(data["auxiliary_target_names"])):
            key = f"aux_{name}"
            if key in data.files:
                auxiliary_targets[str(name)] = data[key]
    return TrajectoryDataset(
        observations=data["observations"],
        masks=data["masks"],
        targets=data["targets"],
        dataset_config=dataset_config,
        sim_config=json.loads(str(data["sim_config"])),
        auxiliary_targets=auxiliary_targets,
    )


def auxiliary_target_metadata(auxiliary_targets: Optional[Dict[str, np.ndarray]]) -> Dict[str, object]:
    """Describe optional Phase 5 correction supervision arrays."""

    if not auxiliary_targets:
        return {
            "enabled": False,
            "target_names": [],
            "schema_version": "phase5_auxiliary_targets_v1",
        }
    return {
        "enabled": True,
        "schema_version": "phase5_auxiliary_targets_v1",
        "target_names": sorted(auxiliary_targets.keys()),
        "shapes": {
            name: list(np.asarray(values).shape)
            for name, values in sorted(auxiliary_targets.items())
        },
        "notes": [
            "Auxiliary targets are shield-correction diagnostics for co-design research foundations.",
            "They are optional and are not consumed by the current NumPy velocity-only trainers.",
        ],
    }


def _shield_auxiliary_targets_for_record(
    record: Dict[str, np.ndarray],
    scenario: Scenario,
    sim_config: SimConfig,
) -> Dict[str, np.ndarray]:
    """Compute per-agent low-risk shield correction targets for one expert step."""

    scenario_config = sim_config_for_scenario(sim_config, scenario)
    shield = make_shield("priority", scenario_config, max_iterations=8)
    target_velocities = np.asarray(record["target_velocities"], dtype=np.float64)
    corrected, diagnostics = shield.apply(
        positions=record["positions"],
        velocities=target_velocities,
        goals=record["goals"],
        radii=record["radii"],
        obstacle_map=scenario.obstacle_map,
    )
    correction = corrected - target_velocities
    correction_norm = np.linalg.norm(correction, axis=1)
    unsafe = correction_norm > 1e-8
    obstacle_intervention = np.full(
        target_velocities.shape[0],
        float(
            getattr(diagnostics, "initial_obstacle_conflicts", 0) > 0
            or getattr(diagnostics, "obstacle_blocked_agents", 0) > 0
        ),
        dtype=np.float64,
    )
    pair_initial = int(getattr(diagnostics, "initial_conflicts", 0)) - int(
        getattr(diagnostics, "initial_obstacle_conflicts", 0)
    )
    pairwise_intervention = np.full(
        target_velocities.shape[0],
        float(pair_initial > 0),
        dtype=np.float64,
    )
    return {
        "correction_vector": correction.astype(np.float64),
        "correction_norm": correction_norm.astype(np.float64),
        "correction_needed": unsafe.astype(np.float64),
        "unsafe_command": unsafe.astype(np.float64),
        "obstacle_intervention": obstacle_intervention,
        "pairwise_intervention": pairwise_intervention,
    }


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
    observation_version: str = OBSERVATION_LEGACY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode self and nearest-neighbor tokens for one agent."""

    n_agents = positions.shape[0]
    scale = _world_scale(sim_config)
    speed = max(sim_config.max_speed, 1e-8)
    observation_version = normalize_observation_version(observation_version)
    feature_dim = observation_feature_dim(observation_version)
    obstacle_token_count = max(0, int(max_obstacle_tokens))
    token_count = max_neighbors + 1 + obstacle_token_count
    tokens = np.zeros((token_count, feature_dim), dtype=np.float64)
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
            ]
            + [0.0] * (feature_dim - FEATURE_DIM),
            dtype=np.float64,
        )
        mask[slot] = True

    def fill_waypoint_context(slot: int) -> None:
        if observation_version != OBSERVATION_OBSTACLE_WAYPOINT_V2 or slot != 0:
            return
        goal_delta = goals[agent_index] - positions[agent_index]
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance > 1e-8:
            tokens[slot, 16:18] = goal_delta / goal_distance
        if obstacle_map is None or goal_distance <= sim_config.goal_tolerance:
            return
        path = astar_grid_path(
            obstacle_map,
            positions[agent_index],
            goals[agent_index],
            float(radii[agent_index]),
            margin=sim_config.safety_margin,
        )
        if not path:
            return
        waypoints = [obstacle_map.cell_center(row, col) for row, col in path[1:]]
        waypoints.append(goals[agent_index])
        useful = [
            waypoint
            for waypoint in waypoints
            if np.linalg.norm(waypoint - positions[agent_index])
            > max(sim_config.goal_tolerance * 0.75, float(radii[agent_index]) * 1.5)
        ]
        if not useful:
            useful = [goals[agent_index]]
        first_delta = useful[0] - positions[agent_index]
        first_distance = float(np.linalg.norm(first_delta))
        if first_distance > 1e-8:
            tokens[slot, 10:12] = first_delta / first_distance
        if len(useful) > 1:
            second_delta = useful[1] - useful[0]
            second_distance = float(np.linalg.norm(second_delta))
            if second_distance > 1e-8:
                tokens[slot, 12:14] = second_delta / second_distance
        remaining = first_distance
        if len(useful) > 1:
            remaining += sum(
                float(np.linalg.norm(useful[index + 1] - useful[index]))
                for index in range(len(useful) - 1)
            )
        tokens[slot, 14] = remaining / scale
        tokens[slot, 15] = 1.0

    fill_token(0, agent_index, True)
    fill_waypoint_context(0)
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
                ]
                + [0.0] * (feature_dim - FEATURE_DIM),
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
    observation_version: str = OBSERVATION_LEGACY,
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
            observation_version=observation_version,
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
    auxiliary_buffers: Dict[str, List[np.ndarray]] = {}
    scenarios = generate_scenarios(dataset_config, sim_config)

    for scenario in scenarios:
        scenario_sim_config = sim_config_for_scenario(sim_config, scenario)
        for record in rollout_expert(
            scenario,
            sim_config,
            dataset_config.horizon,
            expert_type=dataset_config.expert_type,
        ):
            record_auxiliary = (
                _shield_auxiliary_targets_for_record(record, scenario, sim_config)
                if dataset_config.include_auxiliary_targets
                else {}
            )
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
                observation_version=dataset_config.observation_version,
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
                for name, values in record_auxiliary.items():
                    auxiliary_buffers.setdefault(name, []).append(values[agent_index])
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

    auxiliary_targets = (
        {
            name: np.asarray(values, dtype=np.float64)
            for name, values in auxiliary_buffers.items()
        }
        if auxiliary_buffers
        else None
    )
    dataset_metadata = {
        **dataset_config.to_dict(),
        "observation_metadata": observation_metadata(
            dataset_config.observation_version,
            dataset_config.max_neighbors,
            dataset_config.max_obstacle_tokens,
        ),
        "auxiliary_target_metadata": auxiliary_target_metadata(auxiliary_targets),
    }
    return TrajectoryDataset(
        observations=np.asarray(observations, dtype=np.float64),
        masks=np.asarray(masks, dtype=bool),
        targets=np.asarray(targets, dtype=np.float64),
        dataset_config=dataset_metadata,
        sim_config=sim_config.to_dict(),
        auxiliary_targets=auxiliary_targets,
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
