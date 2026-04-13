import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flow_shield.config import (
    DatasetConfig,
    ModelConfig,
    Phase4Config,
    SimConfig,
    phase4_component_configs,
)
from flow_shield.dataset import build_dataset
from flow_shield.experiment import evaluate_phase4_ablations, train_phase4_model
from flow_shield.model import NumpyTransformerPolicy, TorchTransformerPolicy, make_policy
from flow_shield.shield import DEFAULT_SHIELD_VARIANTS


class PhaseFourScalingTests(unittest.TestCase):
    def test_phase4_config_creation(self):
        config = Phase4Config(
            train_scenarios=3,
            eval_scenarios=2,
            num_agents=24,
            world_size=22.0,
            horizon=16,
            max_steps=32,
            max_neighbors=10,
            max_samples=500,
            seed=19,
            d_model=32,
            num_heads=4,
            num_layers=2,
        )
        sim, train_data, eval_data, model = phase4_component_configs(config)
        self.assertEqual(sim.world_size, (22.0, 22.0))
        self.assertEqual(sim.max_steps, 32)
        self.assertEqual(train_data.num_agents, 24)
        self.assertEqual(train_data.max_neighbors, 10)
        self.assertEqual(train_data.max_samples, 500)
        self.assertEqual(eval_data.num_scenarios, 2)
        self.assertEqual(model.policy_type, "numpy_transformer")
        self.assertEqual(model.num_heads, 4)

    def test_moderate_scale_dataset_generation(self):
        sim = SimConfig(world_size=(20.0, 20.0), max_steps=20)
        data = DatasetConfig(
            num_scenarios=2,
            num_agents=24,
            horizon=5,
            max_neighbors=6,
            min_start_goal_distance=4.0,
            seed=31,
            max_samples=180,
        )
        dataset = build_dataset(data, sim)
        self.assertGreater(dataset.num_samples, 0)
        self.assertLessEqual(dataset.num_samples, 180)
        self.assertEqual(dataset.observations.shape[1:], (7, 10))
        self.assertEqual(dataset.masks.shape[1], 7)
        self.assertEqual(dataset.targets.shape[1], 2)

    def test_phase4_model_train_eval_and_shield_compatibility(self):
        sim = SimConfig(world_size=(16.0, 16.0), max_steps=8)
        train_data = DatasetConfig(
            num_scenarios=2,
            num_agents=12,
            horizon=5,
            max_neighbors=5,
            min_start_goal_distance=3.0,
            seed=37,
            max_samples=120,
        )
        dataset = build_dataset(train_data, sim)
        model_config = ModelConfig(
            d_model=16,
            policy_type="numpy_transformer",
            num_heads=2,
            num_layers=1,
            batch_size=64,
            ridge_lambda=1e-3,
            seed=41,
        )
        model, history = train_phase4_model(dataset, sim, model_config)
        self.assertIsInstance(model, NumpyTransformerPolicy)
        self.assertIn("closed_form_ridge_output_head", history["fit_method"])

        eval_data = DatasetConfig(
            num_scenarios=1,
            num_agents=12,
            horizon=5,
            max_neighbors=5,
            min_start_goal_distance=3.0,
            seed=1037,
        )
        result = evaluate_phase4_ablations(
            model,
            sim,
            eval_data,
            variants=DEFAULT_SHIELD_VARIANTS,
            max_iterations=3,
        )
        self.assertEqual(tuple(result["variant_order"]), DEFAULT_SHIELD_VARIANTS)
        for variant in DEFAULT_SHIELD_VARIANTS:
            metrics = result["metrics"][variant]
            for key in (
                "collision_rate",
                "pair_collisions_per_run",
                "mean_min_separation_violation",
                "max_min_separation_violation",
                "mean_shield_correction_norm",
                "max_shield_correction_norm",
                "mean_smoothness",
                "mean_time_to_goal",
                "success_rate",
                "deadlock_rate",
                "steps_per_second",
                "agents_per_second",
            ):
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]))

    def test_torch_transformer_policy_type_is_optional(self):
        sim = SimConfig(world_size=(8.0, 8.0), max_steps=4)
        model_config = ModelConfig(
            feature_dim=10,
            d_model=8,
            policy_type="torch_transformer",
            num_heads=2,
            num_layers=1,
            epochs=1,
            batch_size=8,
            torch_device="cpu",
            seed=43,
        )
        if importlib.util.find_spec("torch") is None:
            with self.assertRaises(ImportError):
                make_policy(model_config, sim)
            return

        data = DatasetConfig(
            num_scenarios=1,
            num_agents=3,
            horizon=3,
            max_neighbors=2,
            min_start_goal_distance=2.0,
            seed=43,
        )
        dataset = build_dataset(data, sim)
        model_config = ModelConfig(
            feature_dim=int(dataset.observations.shape[2]),
            d_model=8,
            policy_type="torch_transformer",
            num_heads=2,
            num_layers=1,
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            torch_device="cpu",
            seed=43,
        )
        model, history = train_phase4_model(dataset, sim, model_config)
        self.assertIsInstance(model, TorchTransformerPolicy)
        self.assertEqual(history["fit_method"], "torch_transformer_adamw")
        predictions = model.predict_batch(dataset.observations[:2], dataset.masks[:2])
        self.assertEqual(predictions.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_phase4_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "phase4"
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                "-m",
                "flow_shield.cli",
                "phase4",
                "--output-dir",
                str(output_dir),
                "--num-agents",
                "8",
                "--world-size",
                "12",
                "--train-scenarios",
                "2",
                "--eval-scenarios",
                "1",
                "--horizon",
                "5",
                "--max-steps",
                "6",
                "--max-neighbors",
                "4",
                "--min-start-goal-distance",
                "3",
                "--max-samples",
                "80",
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
            self.assertIn('"none"', completed.stdout)
            result_path = output_dir / "phase4_results.json"
            model_path = output_dir / "phase4_policy.npz"
            dataset_path = output_dir / "phase4_scaled_dataset.npz"
            self.assertTrue(result_path.exists())
            self.assertTrue(model_path.exists())
            self.assertTrue(dataset_path.exists())
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIn("backend_diagnostics", result)
            self.assertIn("agents_per_second", result["metrics"]["none"])
            self.assertEqual(result["ablations"]["variant_order"], ["none", "pairwise"])


if __name__ == "__main__":
    unittest.main()
