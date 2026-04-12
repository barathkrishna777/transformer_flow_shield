import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flow_shield.config import DatasetConfig, ModelConfig, SimConfig
from flow_shield.dataset import build_dataset
from flow_shield.expert import astar_grid_path, generate_scenarios, obstacle_map_velocity
from flow_shield.experiment import evaluate_phase3_ablations, train_phase4_model
from flow_shield.maps import GridMap, load_moving_ai_map
from flow_shield.shield import DEFAULT_SHIELD_VARIANTS, make_shield
from flow_shield.simulator import ContinuousWorld, Scenario


FIXTURE_MAP = Path(__file__).resolve().parent / "fixtures" / "tiny_obstacle.map"


class PhaseThreeObstacleMapTests(unittest.TestCase):
    def test_moving_ai_map_parser(self):
        grid = load_moving_ai_map(FIXTURE_MAP)
        self.assertEqual(grid.map_type, "octile")
        self.assertEqual(grid.width, 7)
        self.assertEqual(grid.height, 7)
        self.assertEqual(grid.world_size, (7.0, 7.0))
        self.assertTrue(grid.blocked[1, 2])
        self.assertFalse(grid.blocked[0, 0])
        self.assertGreater(grid.free_count, grid.blocked_count)

    def test_obstacle_free_start_goal_sampling(self):
        sim = SimConfig(agent_radius=0.18, safety_margin=0.02)
        data = DatasetConfig(
            num_scenarios=3,
            num_agents=3,
            horizon=4,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=3,
            min_start_goal_distance=2.0,
            seed=23,
        )
        scenarios = generate_scenarios(data, sim)
        self.assertEqual(len(scenarios), 3)
        for scenario in scenarios:
            self.assertIsNotNone(scenario.obstacle_map)
            self.assertEqual(scenario.world_size, (7.0, 7.0))
            self.assertEqual(
                scenario.obstacle_map.circle_collisions(
                    scenario.starts,
                    scenario.radii,
                    margin=sim.safety_margin,
                ),
                (),
            )
            for start, goal in zip(scenario.starts, scenario.goals):
                self.assertGreaterEqual(np.linalg.norm(goal - start), 2.0)

    def test_simulator_projects_motion_before_blocked_cell(self):
        blocked = np.zeros((3, 3), dtype=bool)
        blocked[1, 1] = True
        grid = GridMap(width=3, height=3, blocked=blocked, name="center_block")
        sim = SimConfig(world_size=(3.0, 3.0), dt=1.0, agent_radius=0.2, max_speed=2.0)
        scenario = Scenario(
            starts=np.array([[0.5, 1.5]], dtype=np.float64),
            goals=np.array([[2.5, 1.5]], dtype=np.float64),
            radii=np.array([0.2], dtype=np.float64),
            world_size=grid.world_size,
            obstacle_map=grid,
        )
        world = ContinuousWorld(scenario, sim)
        record = world.step(np.array([[2.0, 0.0]], dtype=np.float64))
        self.assertLessEqual(record["positions"][0, 0], 0.80001)
        self.assertEqual(record["obstacle_collisions"], ())
        self.assertEqual(record["obstacle_motion_hits"], (0,))

    def test_obstacle_aware_expert_waypoint_velocity(self):
        grid = load_moving_ai_map(FIXTURE_MAP)
        sim = SimConfig(world_size=grid.world_size, agent_radius=0.18, max_speed=1.0)
        start = np.array([[0.5, 2.5]], dtype=np.float64)
        goal = np.array([[5.5, 2.5]], dtype=np.float64)
        path = astar_grid_path(grid, start[0], goal[0], sim.agent_radius, sim.safety_margin)
        self.assertIsNotNone(path)
        velocity = obstacle_map_velocity(start, goal, sim, grid)
        self.assertEqual(velocity.shape, (1, 2))
        self.assertGreater(np.linalg.norm(velocity[0]), 0.0)

    def test_dataset_generation_on_tiny_obstacle_map(self):
        sim = SimConfig(agent_radius=0.18, max_steps=10)
        data = DatasetConfig(
            num_scenarios=2,
            num_agents=3,
            horizon=5,
            max_neighbors=2,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=4,
            min_start_goal_distance=2.0,
            seed=41,
            max_samples=60,
        )
        dataset = build_dataset(data, sim)
        self.assertGreater(dataset.num_samples, 0)
        self.assertEqual(dataset.observations.shape[1:], (7, 10))
        self.assertEqual(dataset.masks.shape[1], 7)
        self.assertTrue(np.any(dataset.observations[:, 3:, 9] < 0.0))

    def test_model_train_eval_and_shield_variants_on_obstacle_map(self):
        sim = SimConfig(agent_radius=0.18, max_steps=8)
        train_data = DatasetConfig(
            num_scenarios=2,
            num_agents=3,
            horizon=5,
            max_neighbors=2,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=3,
            min_start_goal_distance=2.0,
            seed=53,
            max_samples=80,
        )
        dataset = build_dataset(train_data, sim)
        model_config = ModelConfig(
            d_model=16,
            policy_type="numpy_transformer",
            num_heads=2,
            num_layers=1,
            batch_size=64,
            seed=59,
        )
        model, _ = train_phase4_model(dataset, sim, model_config)
        eval_data = DatasetConfig(
            num_scenarios=1,
            num_agents=3,
            horizon=5,
            max_neighbors=2,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=3,
            min_start_goal_distance=2.0,
            seed=1053,
        )
        result = evaluate_phase3_ablations(
            model,
            sim,
            eval_data,
            variants=DEFAULT_SHIELD_VARIANTS,
            max_iterations=3,
        )
        self.assertEqual(tuple(result["variant_order"]), DEFAULT_SHIELD_VARIANTS)
        for variant in DEFAULT_SHIELD_VARIANTS:
            self.assertIn("obstacle_collision_rate", result["metrics"][variant])
            self.assertTrue(np.isfinite(result["metrics"][variant]["success_rate"]))

    def test_shield_blocks_obstacle_entering_velocity(self):
        grid = load_moving_ai_map(FIXTURE_MAP)
        sim = SimConfig(world_size=grid.world_size, dt=1.0, agent_radius=0.2, max_speed=2.0)
        shield = make_shield("pairwise", sim, max_iterations=3)
        positions = np.array([[1.5, 1.5]], dtype=np.float64)
        velocities = np.array([[1.0, 0.0]], dtype=np.float64)
        goals = np.array([[4.5, 1.5]], dtype=np.float64)
        radii = np.array([0.2], dtype=np.float64)
        safe, diagnostics = shield.apply(
            positions,
            velocities,
            goals,
            radii,
            obstacle_map=grid,
        )
        self.assertGreaterEqual(diagnostics.initial_obstacle_conflicts, 1)
        self.assertEqual(
            shield.predicted_obstacle_collisions(positions, safe, radii, obstacle_map=grid),
            (),
        )

    def test_phase3_cli_smoke_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "phase3"
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                "-m",
                "flow_shield.cli",
                "phase3",
                "--map",
                str(FIXTURE_MAP),
                "--output-dir",
                str(output_dir),
                "--num-agents",
                "3",
                "--train-scenarios",
                "2",
                "--eval-scenarios",
                "1",
                "--horizon",
                "5",
                "--max-steps",
                "8",
                "--max-neighbors",
                "2",
                "--min-start-goal-distance",
                "2",
                "--max-samples",
                "80",
                "--max-obstacle-tokens",
                "3",
                "--d-model",
                "16",
                "--num-heads",
                "2",
                "--num-layers",
                "1",
                "--batch-size",
                "64",
                "--variants",
                "none",
                "pairwise",
                "--max-iterations",
                "3",
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"pairwise"', completed.stdout)
            result_path = output_dir / "phase3_results.json"
            self.assertTrue(result_path.exists())
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["phase"], "phase3_obstacle_maps")
            self.assertEqual(result["map_metadata"]["width"], 7)
            self.assertIn("obstacle_collision_rate", result["metrics"]["pairwise"])


if __name__ == "__main__":
    unittest.main()
