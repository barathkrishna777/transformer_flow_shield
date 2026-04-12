import unittest

import numpy as np

from flow_shield.config import SimConfig
from flow_shield.experiment import evaluate_phase2_ablations
from flow_shield.expert import straight_line_velocity
from flow_shield.scenarios import phase2_adversarial_scenarios, two_agent_swap
from flow_shield.shield import DEFAULT_SHIELD_VARIANTS, make_shield


class PhaseTwoShieldTests(unittest.TestCase):
    def setUp(self):
        self.sim = SimConfig(
            world_size=(10.0, 10.0),
            dt=0.5,
            agent_radius=0.25,
            max_speed=2.0,
            safety_margin=0.03,
            max_steps=20,
        )
        self.safe_positions = np.array([[2.0, 2.0], [7.0, 7.0]], dtype=np.float64)
        self.safe_velocities = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)
        self.safe_goals = np.array([[4.0, 2.0], [7.0, 4.0]], dtype=np.float64)
        self.safe_radii = np.array([0.25, 0.25], dtype=np.float64)

    def test_safe_commands_are_minimally_changed_by_all_variants(self):
        for variant in DEFAULT_SHIELD_VARIANTS:
            with self.subTest(variant=variant):
                shield = make_shield(variant, self.sim, max_iterations=8)
                safe, diagnostics = shield.apply(
                    self.safe_positions,
                    self.safe_velocities,
                    self.safe_goals,
                    self.safe_radii,
                )
                self.assertTrue(np.allclose(safe, self.safe_velocities))
                self.assertEqual(diagnostics.initial_conflicts, 0)
                self.assertEqual(diagnostics.final_conflicts, 0)
                self.assertAlmostEqual(diagnostics.max_correction_norm, 0.0)

    def test_conflicting_commands_are_resolved_or_reported_by_each_variant(self):
        positions = np.array([[4.0, 5.0], [6.0, 5.0]], dtype=np.float64)
        velocities = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float64)
        goals = np.array([[8.0, 5.0], [2.0, 5.0]], dtype=np.float64)
        radii = np.array([0.25, 0.25], dtype=np.float64)
        resolving_variants = {"pairwise", "priority", "pibt"}

        for variant in DEFAULT_SHIELD_VARIANTS:
            with self.subTest(variant=variant):
                shield = make_shield(variant, self.sim, max_iterations=12)
                safe, diagnostics = shield.apply(positions, velocities, goals, radii)
                self.assertGreaterEqual(diagnostics.initial_conflicts, 1)
                if variant in resolving_variants:
                    self.assertEqual(
                        shield.predicted_collision_pairs(positions, safe, radii),
                        (),
                    )
                    self.assertTrue(diagnostics.resolved)
                else:
                    self.assertGreaterEqual(diagnostics.final_conflicts, 1)
                    self.assertTrue(diagnostics.limited)

    def test_phase2_evaluator_reports_required_metrics(self):
        scenarios = [two_agent_swap(self.sim)] + phase2_adversarial_scenarios(
            self.sim,
            circle_agents=4,
            bottleneck_agents=4,
        )[1:]

        def policy(positions, velocities, goals, radii):
            del velocities, radii
            return straight_line_velocity(positions, goals, self.sim)

        result = evaluate_phase2_ablations(
            policy,
            self.sim,
            scenarios=scenarios,
            variants=("none", "pairwise", "pibt"),
            max_iterations=6,
        )
        for variant in ("none", "pairwise", "pibt"):
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
            ):
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()
