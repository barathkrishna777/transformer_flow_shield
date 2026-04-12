import tempfile
import unittest

import numpy as np

from flow_shield.config import DatasetConfig, ModelConfig, SimConfig
from flow_shield.dataset import build_dataset
from flow_shield.expert import generate_scenarios, straight_line_velocity
from flow_shield.experiment import run_phase1_experiment
from flow_shield.geometry import collision_pairs
from flow_shield.model import NumpyAttentionPolicy
from flow_shield.shield import CollisionShield


class PhaseZeroOneTests(unittest.TestCase):
    def test_empty_scenario_sampling_has_no_initial_collisions(self):
        sim = SimConfig(world_size=(8.0, 8.0), agent_radius=0.2)
        data = DatasetConfig(num_scenarios=4, num_agents=5, seed=3)
        scenarios = generate_scenarios(data, sim)
        self.assertEqual(len(scenarios), 4)
        for scenario in scenarios:
            self.assertEqual(collision_pairs(scenario.starts, scenario.radii), ())

    def test_straight_line_expert_points_toward_goal(self):
        sim = SimConfig(max_speed=1.0, dt=0.1)
        positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        goals = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)
        velocities = straight_line_velocity(positions, goals, sim)
        self.assertTrue(np.allclose(velocities, np.array([[1.0, 0.0], [0.0, 1.0]])))

    def test_collision_shield_resolves_predicted_head_on_conflict(self):
        sim = SimConfig(world_size=(10.0, 10.0), dt=0.5, agent_radius=0.25, max_speed=2.0)
        shield = CollisionShield(sim, mode="pairwise", max_iterations=12)
        positions = np.array([[4.0, 5.0], [6.0, 5.0]], dtype=np.float64)
        velocities = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float64)
        goals = np.array([[8.0, 5.0], [2.0, 5.0]], dtype=np.float64)
        radii = np.array([0.25, 0.25], dtype=np.float64)
        safe, diagnostics = shield.apply(positions, velocities, goals, radii)
        self.assertGreaterEqual(diagnostics.initial_conflicts, 1)
        self.assertEqual(shield.predicted_collision_pairs(positions, safe, radii), ())

    def test_dataset_and_model_training_smoke(self):
        sim = SimConfig(world_size=(7.0, 7.0), max_steps=30)
        data = DatasetConfig(num_scenarios=4, num_agents=3, horizon=12, max_neighbors=2, seed=5)
        dataset = build_dataset(data, sim)
        model_config = ModelConfig(d_model=12, epochs=3, batch_size=16, learning_rate=3e-3, seed=4)
        model = NumpyAttentionPolicy.from_config(model_config, sim)
        before = model.loss(dataset.observations, dataset.masks, dataset.targets)
        history = model.fit(dataset.observations, dataset.masks, dataset.targets, model_config)
        after = model.loss(dataset.observations, dataset.masks, dataset.targets)
        self.assertEqual(dataset.observations.shape[2], model_config.feature_dim)
        self.assertEqual(len(history["train_loss"]), 3)
        self.assertTrue(np.isfinite(after))
        self.assertLess(after, before * 1.5)

    def test_phase1_pipeline_runs_end_to_end(self):
        sim = SimConfig(world_size=(7.0, 7.0), max_steps=25)
        data = DatasetConfig(num_scenarios=4, num_agents=3, horizon=10, max_neighbors=2, seed=9)
        model = ModelConfig(d_model=12, epochs=2, batch_size=16, learning_rate=3e-3, seed=13)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_phase1_experiment(
                tmpdir,
                sim_config=sim,
                dataset_config=data,
                model_config=model,
                eval_scenarios=2,
            )
        self.assertIn("learned_planner_only", result["metrics"])
        self.assertIn("learned_planner_plus_collision_shield", result["metrics"])
        self.assertGreater(result["num_training_samples"], 0)


if __name__ == "__main__":
    unittest.main()

