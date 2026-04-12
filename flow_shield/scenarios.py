"""Adversarial continuous-space scenarios for phase 2 shield ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import SimConfig
from .geometry import project_positions_to_bounds
from .simulator import Scenario


@dataclass(frozen=True)
class NamedScenario:
    name: str
    scenario: Scenario


def _radii(count: int, config: SimConfig) -> np.ndarray:
    return np.full(count, config.agent_radius, dtype=np.float64)


def _scenario(starts: np.ndarray, goals: np.ndarray, config: SimConfig) -> Scenario:
    radii = _radii(starts.shape[0], config)
    return Scenario(
        starts=project_positions_to_bounds(starts, radii, config.world_size),
        goals=project_positions_to_bounds(goals, radii, config.world_size),
        radii=radii,
        world_size=config.world_size,
    )


def two_agent_swap(config: SimConfig) -> NamedScenario:
    world = np.asarray(config.world_size, dtype=np.float64)
    center = world * 0.5
    half_span = max(1.0, min(world[0] * 0.28, 3.0))
    starts = np.array(
        [
            [center[0] - half_span, center[1]],
            [center[0] + half_span, center[1]],
        ],
        dtype=np.float64,
    )
    goals = starts[::-1].copy()
    return NamedScenario("two_agent_swap", _scenario(starts, goals, config))


def multi_agent_crossing(config: SimConfig) -> NamedScenario:
    world = np.asarray(config.world_size, dtype=np.float64)
    center = world * 0.5
    margin = max(config.agent_radius + 0.4, min(world) * 0.12)
    starts = np.array(
        [
            [margin, center[1]],
            [world[0] - margin, center[1]],
            [center[0], margin],
            [center[0], world[1] - margin],
            [margin, margin],
            [world[0] - margin, world[1] - margin],
        ],
        dtype=np.float64,
    )
    goals = np.array(
        [
            [world[0] - margin, center[1]],
            [margin, center[1]],
            [center[0], world[1] - margin],
            [center[0], margin],
            [world[0] - margin, world[1] - margin],
            [margin, margin],
        ],
        dtype=np.float64,
    )
    return NamedScenario("multi_agent_crossing", _scenario(starts, goals, config))


def dense_circle_swap(config: SimConfig, num_agents: int = 8) -> NamedScenario:
    count = max(4, int(num_agents))
    world = np.asarray(config.world_size, dtype=np.float64)
    center = world * 0.5
    radius = max(1.0, min(world) * 0.32)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    starts = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )
    goals = np.column_stack(
        [
            center[0] + radius * np.cos(angles + np.pi),
            center[1] + radius * np.sin(angles + np.pi),
        ]
    )
    return NamedScenario("dense_circle_swap", _scenario(starts, goals, config))


def corridor_bottleneck(config: SimConfig, num_agents: int = 6) -> NamedScenario:
    count = max(4, int(num_agents))
    if count % 2 == 1:
        count += 1
    half = count // 2
    world = np.asarray(config.world_size, dtype=np.float64)
    center_y = world[1] * 0.5
    lane_spacing = max(2.5 * config.agent_radius + config.safety_margin, 0.45)
    y_offsets = (np.arange(half, dtype=np.float64) - (half - 1) * 0.5) * lane_spacing
    left_x = max(config.agent_radius + 0.4, world[0] * 0.16)
    right_x = world[0] - left_x
    starts_left = np.column_stack([np.full(half, left_x), center_y + y_offsets])
    starts_right = np.column_stack([np.full(half, right_x), center_y + y_offsets[::-1]])
    goals_left = np.column_stack([np.full(half, right_x), center_y + y_offsets[::-1]])
    goals_right = np.column_stack([np.full(half, left_x), center_y + y_offsets])
    starts = np.vstack([starts_left, starts_right])
    goals = np.vstack([goals_left, goals_right])
    return NamedScenario("corridor_bottleneck", _scenario(starts, goals, config))


def phase2_adversarial_scenarios(
    config: SimConfig,
    circle_agents: int = 8,
    bottleneck_agents: int = 6,
) -> List[NamedScenario]:
    return [
        two_agent_swap(config),
        multi_agent_crossing(config),
        dense_circle_swap(config, num_agents=circle_agents),
        corridor_bottleneck(config, num_agents=bottleneck_agents),
    ]
