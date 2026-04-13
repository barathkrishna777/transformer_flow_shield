import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flow_shield.config import DatasetConfig, ModelConfig, SimConfig
from flow_shield.benchmark import build_benchmark_plan, run_benchmark_plan
from flow_shield.dataset import build_dataset, load_dataset
from flow_shield.expert import (
    astar_grid_path,
    generate_scenarios,
    obstacle_map_velocity,
    prioritized_obstacle_map_velocity,
)
from flow_shield.experiment import evaluate_phase3_ablations, train_phase4_model
from flow_shield.maps import GridMap, load_moving_ai_map, load_moving_ai_scen
from flow_shield.shield import DEFAULT_SHIELD_VARIANTS, make_shield
from flow_shield.simulator import ContinuousWorld, Scenario, rollout


FIXTURE_MAP = Path(__file__).resolve().parent / "fixtures" / "tiny_obstacle.map"
FIXTURE_SCEN = Path(__file__).resolve().parent / "fixtures" / "tiny_obstacle.scen"


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

    def test_expert_baseline_reaches_simple_obstacle_task(self):
        grid = load_moving_ai_map(FIXTURE_MAP)
        sim = SimConfig(world_size=grid.world_size, agent_radius=0.18, max_speed=1.2, max_steps=120)
        scenario = Scenario(
            starts=np.array([[0.5, 0.5]], dtype=np.float64),
            goals=np.array([[6.5, 6.5]], dtype=np.float64),
            radii=np.array([sim.agent_radius], dtype=np.float64),
            world_size=grid.world_size,
            obstacle_map=grid,
        )

        def policy(positions, velocities, goals, radii):
            del velocities
            return obstacle_map_velocity(positions, goals, sim, grid, radii=radii)

        result = rollout(scenario, sim, policy, max_steps=sim.max_steps)
        self.assertTrue(result["success"])
        self.assertEqual(result["obstacle_collisions"], 0)

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

    def test_obstacle_waypoint_v2_observation_shape_and_metadata(self):
        sim = SimConfig(agent_radius=0.18, max_steps=10)
        data = DatasetConfig(
            num_scenarios=1,
            num_agents=2,
            horizon=3,
            max_neighbors=1,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=3,
            min_start_goal_distance=2.0,
            observation_version="obstacle_waypoint_v2",
            seed=42,
        )
        dataset = build_dataset(data, sim)
        self.assertEqual(dataset.observations.shape[1:], (5, 18))
        self.assertEqual(dataset.dataset_config["observation_version"], "obstacle_waypoint_v2")
        metadata = dataset.dataset_config["observation_metadata"]
        self.assertEqual(metadata["feature_dim"], 18)
        self.assertIn("next_waypoint_dir_x", metadata["feature_names"])
        self.assertTrue(np.any(np.abs(dataset.observations[:, 0, 10:12]) > 0.0))
        self.assertTrue(np.any(dataset.observations[:, 0, 15] > 0.0))

    def test_legacy_dataset_loading_defaults_observation_metadata(self):
        sim = SimConfig(agent_radius=0.18, max_steps=10)
        data = DatasetConfig(
            num_scenarios=1,
            num_agents=2,
            horizon=3,
            max_neighbors=1,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=3,
            min_start_goal_distance=2.0,
            seed=43,
        )
        dataset = build_dataset(data, sim)
        legacy_config = dict(dataset.dataset_config)
        legacy_config.pop("observation_version", None)
        legacy_config.pop("observation_metadata", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_like.npz"
            np.savez_compressed(
                path,
                observations=dataset.observations,
                masks=dataset.masks,
                targets=dataset.targets,
                dataset_config=json.dumps(legacy_config),
                sim_config=json.dumps(dataset.sim_config),
            )
            loaded = load_dataset(path)
        self.assertEqual(loaded.dataset_config["observation_version"], "legacy")
        self.assertEqual(loaded.dataset_config["observation_metadata"]["feature_dim"], 10)
        self.assertIsNone(loaded.auxiliary_targets)

    def test_phase5_auxiliary_correction_targets_are_optional_and_saved(self):
        sim = SimConfig(agent_radius=0.18, max_steps=10)
        data = DatasetConfig(
            num_scenarios=1,
            num_agents=2,
            horizon=3,
            max_neighbors=1,
            scenario_type="obstacle_map",
            map_path=str(FIXTURE_MAP),
            max_obstacle_tokens=2,
            min_start_goal_distance=2.0,
            include_auxiliary_targets=True,
            seed=44,
        )
        dataset = build_dataset(data, sim)
        self.assertIsNotNone(dataset.auxiliary_targets)
        assert dataset.auxiliary_targets is not None
        for key in (
            "correction_vector",
            "correction_norm",
            "correction_needed",
            "unsafe_command",
            "obstacle_intervention",
            "pairwise_intervention",
        ):
            self.assertIn(key, dataset.auxiliary_targets)
            self.assertEqual(dataset.auxiliary_targets[key].shape[0], dataset.num_samples)
        metadata = dataset.dataset_config["auxiliary_target_metadata"]
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["schema_version"], "phase5_auxiliary_targets_v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aux.npz"
            dataset.save(path)
            loaded = load_dataset(path)
        self.assertIsNotNone(loaded.auxiliary_targets)
        assert loaded.auxiliary_targets is not None
        self.assertEqual(
            loaded.auxiliary_targets["correction_vector"].shape,
            dataset.auxiliary_targets["correction_vector"].shape,
        )

    def test_moving_ai_scen_parser_and_generation(self):
        tasks = load_moving_ai_scen(FIXTURE_SCEN, limit=3)
        self.assertEqual(len(tasks), 3)
        self.assertTrue(np.allclose(tasks[0].start, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(tasks[0].goal, np.array([6.5, 6.5])))
        sim = SimConfig(agent_radius=0.18, safety_margin=0.02)
        data = DatasetConfig(
            num_scenarios=1,
            num_agents=2,
            horizon=3,
            scenario_type="obstacle_map",
            scenario_source="scen",
            map_path=str(FIXTURE_MAP),
            scen_path=str(FIXTURE_SCEN),
            max_obstacle_tokens=2,
            min_start_goal_distance=2.0,
        )
        scenarios = generate_scenarios(data, sim)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].starts.shape, (2, 2))

    def test_prioritized_astar_expert_smoke(self):
        grid = load_moving_ai_map(FIXTURE_MAP)
        sim = SimConfig(world_size=grid.world_size, agent_radius=0.18, max_speed=1.0)
        positions = np.array([[0.5, 0.5], [6.5, 6.5]], dtype=np.float64)
        goals = np.array([[6.5, 6.5], [0.5, 0.5]], dtype=np.float64)
        radii = np.full(2, sim.agent_radius, dtype=np.float64)
        velocity = prioritized_obstacle_map_velocity(positions, goals, sim, grid, radii=radii)
        self.assertEqual(velocity.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(velocity)))
        self.assertTrue(np.any(np.linalg.norm(velocity, axis=1) > 0.0))

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
        self.assertIn("expert_waypoint_baseline", result)
        self.assertIn("pairwise", result["expert_waypoint_baseline"]["metrics"])
        self.assertIn("learned_vs_expert", result)
        for variant in DEFAULT_SHIELD_VARIANTS:
            self.assertIn("obstacle_collision_rate", result["metrics"][variant])
            self.assertIn("correction_needed_rate", result["metrics"][variant])
            self.assertIn("obstacle_intervention_rate", result["metrics"][variant])
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
            self.assertEqual(result["observation_version"], "obstacle_waypoint_v2")
            self.assertEqual(result["observation_metadata"]["feature_dim"], 18)
            self.assertIn("expert_waypoint_baseline", result)
            self.assertIn("pairwise", result["expert_waypoint_baseline"]["metrics"])
            self.assertIn("learned_vs_expert", result)
            self.assertIn("observation_metadata", result)

    def test_phase3_cli_scen_smoke_reports_skip_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "phase3_scen"
            command = [
                sys.executable,
                "-m",
                "flow_shield.cli",
                "phase3",
                "--map",
                str(FIXTURE_MAP),
                "--scenario-source",
                "scen",
                "--scen",
                str(FIXTURE_SCEN),
                "--scen-limit",
                "4",
                "--output-dir",
                str(output_dir),
                "--num-agents",
                "1",
                "--train-scenarios",
                "1",
                "--eval-scenarios",
                "1",
                "--horizon",
                "4",
                "--max-steps",
                "20",
                "--max-neighbors",
                "1",
                "--min-start-goal-distance",
                "2",
                "--max-samples",
                "20",
                "--max-obstacle-tokens",
                "2",
                "--d-model",
                "12",
                "--num-heads",
                "1",
                "--num-layers",
                "1",
                "--batch-size",
                "16",
                "--variants",
                "none",
                "--max-iterations",
                "2",
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"none"', completed.stdout)
            result = json.loads((output_dir / "phase3_results.json").read_text(encoding="utf-8"))
            diagnostics = result["scenario_source_diagnostics"]
            self.assertEqual(diagnostics["scenario_source"], "scen")
            self.assertEqual(diagnostics["raw_tasks"], 4)
            self.assertGreaterEqual(diagnostics["skipped_tasks"], 1)
            self.assertIn("skip_counts", diagnostics)
            self.assertIn("start_collision", diagnostics["skip_counts"])
            self.assertIn("unreachable_astar", diagnostics["skip_counts"])

    def test_benchmark_plan_only_json_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            list_path = tmp / "maps.txt"
            list_path.write_text(f"{FIXTURE_MAP} {FIXTURE_SCEN}\n", encoding="utf-8")
            output_dir = tmp / "bench"
            plan = build_benchmark_plan(
                map_scen_list=list_path,
                output_dir=output_dir,
                agent_counts=(1, 2),
                seeds=(17,),
                train_scenarios=3,
                eval_scenarios=2,
                horizon=8,
                max_steps=20,
                max_neighbors=2,
                min_start_goal_distance=2.0,
                max_samples=50,
                max_obstacle_tokens=2,
                obstacle_context_range=4.0,
                observation_version="obstacle_waypoint_v2",
                expert_type="prioritized_astar",
                limit=1,
                smoke=True,
            )
            summary = run_benchmark_plan(
                plan,
                output_dir,
                plan_only=True,
                echo_progress=False,
            )
            self.assertFalse(summary["executed"])
            self.assertTrue((output_dir / "benchmark_plan.json").exists())
            self.assertTrue((output_dir / "benchmark_summary.json").exists())
            self.assertTrue((output_dir / "benchmark_progress.jsonl").exists())
            self.assertTrue((output_dir / "benchmark_status.json").exists())
            written = json.loads((output_dir / "benchmark_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(written["case_count"], 1)
            self.assertEqual(written["cases"][0]["expert_type"], "prioritized_astar")
            status = json.loads((output_dir / "benchmark_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "plan_only")


if __name__ == "__main__":
    unittest.main()
