"""Configuration objects for continuous MAPF experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class SimConfig:
    """Continuous 2D circular-agent simulator settings."""

    world_size: Tuple[float, float] = (10.0, 10.0)
    dt: float = 0.1
    agent_radius: float = 0.18
    max_speed: float = 1.2
    goal_tolerance: float = 0.25
    max_steps: int = 160
    safety_margin: float = 0.03

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    """Expert rollout and observation encoding settings."""

    num_scenarios: int = 128
    num_agents: int = 8
    horizon: int = 80
    max_neighbors: int = 8
    min_start_goal_distance: float = 3.0
    scenario_type: str = "empty"
    map_path: Optional[str] = None
    map_cell_size: float = 1.0
    max_obstacle_tokens: int = 0
    obstacle_context_range: float = 4.0
    seed: int = 7
    include_reached_agents: bool = False
    max_samples: Optional[int] = None
    agent_count_choices: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Policy training settings."""

    feature_dim: int = 10
    d_model: int = 32
    policy_type: str = "numpy_attention"
    num_heads: int = 1
    num_layers: int = 1
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    validation_split: float = 0.1
    ridge_lambda: float = 1e-3
    seed: int = 11

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase4Config:
    """Scaled training/evaluation defaults for phase 4."""

    train_scenarios: int = 48
    eval_scenarios: int = 12
    num_agents: int = 32
    world_size: float = 32.0
    horizon: int = 120
    max_steps: int = 240
    max_neighbors: int = 16
    min_start_goal_distance: float = 6.0
    max_samples: Optional[int] = 200_000
    seed: int = 4007
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    batch_size: int = 2048
    ridge_lambda: float = 1e-3
    shield_variants: Tuple[str, ...] = (
        "none",
        "velocity_clip",
        "pairwise",
        "priority",
        "pibt",
    )
    scenario_type: str = "empty"
    map_path: Optional[str] = None
    map_cell_size: float = 1.0
    max_obstacle_tokens: int = 0
    obstacle_context_range: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def phase4_component_configs(
    config: Phase4Config,
) -> Tuple[SimConfig, DatasetConfig, DatasetConfig, ModelConfig]:
    """Build concrete sim/data/model configs from the phase 4 bundle."""

    sim_config = SimConfig(
        world_size=(config.world_size, config.world_size),
        max_steps=config.max_steps,
    )
    train_dataset_config = DatasetConfig(
        num_scenarios=config.train_scenarios,
        num_agents=config.num_agents,
        horizon=config.horizon,
        max_neighbors=config.max_neighbors,
        min_start_goal_distance=config.min_start_goal_distance,
        scenario_type=config.scenario_type,
        map_path=config.map_path,
        map_cell_size=config.map_cell_size,
        max_obstacle_tokens=config.max_obstacle_tokens,
        obstacle_context_range=config.obstacle_context_range,
        seed=config.seed,
        max_samples=config.max_samples,
    )
    eval_dataset_config = DatasetConfig(
        num_scenarios=config.eval_scenarios,
        num_agents=config.num_agents,
        horizon=config.horizon,
        max_neighbors=config.max_neighbors,
        min_start_goal_distance=config.min_start_goal_distance,
        scenario_type=config.scenario_type,
        map_path=config.map_path,
        map_cell_size=config.map_cell_size,
        max_obstacle_tokens=config.max_obstacle_tokens,
        obstacle_context_range=config.obstacle_context_range,
        seed=config.seed + 10_000,
        max_samples=None,
    )
    model_config = ModelConfig(
        d_model=config.d_model,
        policy_type="numpy_transformer",
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        batch_size=config.batch_size,
        ridge_lambda=config.ridge_lambda,
        seed=config.seed + 17,
    )
    return sim_config, train_dataset_config, eval_dataset_config, model_config
