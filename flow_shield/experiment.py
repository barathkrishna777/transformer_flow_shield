"""Training and evaluation pipelines for phase 1/2 experiments."""

from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .config import DatasetConfig, ModelConfig, SimConfig
from .dataset import TrajectoryDataset, build_dataset, scenario_to_jsonable
from .expert import (
    astar_cache_info,
    generate_scenarios,
    normalize_expert_type,
    obstacle_map_expert_velocity,
    obstacle_map_velocity,
    scen_task_diagnostics,
    straight_line_velocity,
)
from .maps import load_moving_ai_scen, load_obstacle_map, map_metadata
from .metrics import aggregate_rollouts
from .model import NumpyAttentionPolicy, load_policy, make_policy, policy_from_model
from .scenarios import NamedScenario, phase2_adversarial_scenarios
from .shield import CollisionShield, DEFAULT_SHIELD_VARIANTS, canonical_shield_variant, make_shield
from .simulator import Scenario, rollout, sim_config_for_scenario


def train_phase1_model(
    dataset: TrajectoryDataset,
    sim_config: SimConfig,
    model_config: ModelConfig,
    verbose: bool = False,
) -> tuple[NumpyAttentionPolicy, Dict[str, list]]:
    if model_config.feature_dim != int(dataset.observations.shape[2]):
        model_config = replace(model_config, feature_dim=int(dataset.observations.shape[2]))
    model = NumpyAttentionPolicy.from_config(model_config, sim_config)
    model.observation_version = str(dataset.dataset_config.get("observation_version", "legacy"))
    model.observation_metadata = dict(dataset.dataset_config.get("observation_metadata", {}))
    history = model.fit(
        dataset.observations,
        dataset.masks,
        dataset.targets,
        model_config,
        verbose=verbose,
    )
    return model, history


def backend_diagnostics(policy_type: str) -> Dict[str, object]:
    """Report optional accelerator availability without requiring PyTorch."""

    diagnostics: Dict[str, object] = {
        "selected_policy_type": policy_type,
        "selected_training_backend": (
            "pytorch" if policy_type.strip().lower().replace("-", "_") == "torch_transformer" else "numpy"
        ),
        "torch_available": False,
        "cuda_available": False,
    }
    if importlib.util.find_spec("torch") is None:
        diagnostics["fallback_reason"] = (
            "PyTorch is not installed; torch_transformer training is unavailable."
            if diagnostics["selected_training_backend"] == "pytorch"
            else "PyTorch is not installed; using the dependency-light NumPy phase 4 path."
        )
        return diagnostics

    diagnostics["torch_available"] = True
    try:
        import torch  # type: ignore

        diagnostics["cuda_available"] = bool(torch.cuda.is_available())
        diagnostics["cuda_device_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        diagnostics["torch_version"] = str(torch.__version__)
        diagnostics["fallback_reason"] = (
            "PyTorch transformer backend selected."
            if diagnostics["selected_training_backend"] == "pytorch"
            else "PyTorch is installed, but the selected policy uses the NumPy backend."
        )
    except Exception as exc:  # pragma: no cover - defensive optional dependency path.
        diagnostics["torch_import_error"] = repr(exc)
        diagnostics["fallback_reason"] = (
            "PyTorch was discoverable but could not be imported; using NumPy."
        )
    return diagnostics


def train_phase4_model(
    dataset: TrajectoryDataset,
    sim_config: SimConfig,
    model_config: ModelConfig,
    verbose: bool = False,
):
    """Train the configured phase 4 policy interface."""

    if model_config.feature_dim != int(dataset.observations.shape[2]):
        model_config = replace(model_config, feature_dim=int(dataset.observations.shape[2]))
    model = make_policy(model_config, sim_config)
    model.observation_version = str(dataset.dataset_config.get("observation_version", "legacy"))
    model.observation_metadata = dict(dataset.dataset_config.get("observation_metadata", {}))
    history = model.fit(
        dataset.observations,
        dataset.masks,
        dataset.targets,
        model_config,
        verbose=verbose,
    )
    return model, history


def evaluate_phase1_model(
    model: NumpyAttentionPolicy,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    eval_scenarios: int = 32,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    eval_config = DatasetConfig(
        num_scenarios=eval_scenarios,
        num_agents=dataset_config.num_agents,
        horizon=dataset_config.horizon,
        max_neighbors=dataset_config.max_neighbors,
        min_start_goal_distance=dataset_config.min_start_goal_distance,
        scenario_type=dataset_config.scenario_type,
        scenario_source=dataset_config.scenario_source,
        map_path=dataset_config.map_path,
        map_cell_size=dataset_config.map_cell_size,
        scen_path=dataset_config.scen_path,
        scen_limit=dataset_config.scen_limit,
        max_obstacle_tokens=dataset_config.max_obstacle_tokens,
        obstacle_context_range=dataset_config.obstacle_context_range,
        observation_version=dataset_config.observation_version,
        expert_type=dataset_config.expert_type,
        include_auxiliary_targets=dataset_config.include_auxiliary_targets,
        seed=dataset_config.seed + 1000 if seed is None else seed,
        include_reached_agents=dataset_config.include_reached_agents,
    )
    scenarios = generate_scenarios(eval_config, sim_config)
    obstacle_map = scenarios[0].obstacle_map if scenarios else None
    policy_sim_config = (
        sim_config_for_scenario(sim_config, scenarios[0]) if scenarios else sim_config
    )
    learned_policy = policy_from_model(
        model,
        dataset_config.max_neighbors,
        policy_sim_config,
        obstacle_map=obstacle_map,
        max_obstacle_tokens=dataset_config.max_obstacle_tokens,
        obstacle_context_range=dataset_config.obstacle_context_range,
        observation_version=dataset_config.observation_version,
    )
    shield = CollisionShield(sim_config, mode="priority")

    learned_only = []
    learned_shielded = []
    goals = []
    for scenario in scenarios:
        goals.append(scenario.goals)
        learned_only.append(
            rollout(
                scenario,
                sim_config,
                learned_policy,
                shield=None,
                max_steps=sim_config.max_steps,
            )
        )
        learned_shielded.append(
            rollout(
                scenario,
                sim_config,
                learned_policy,
                shield=shield,
                max_steps=sim_config.max_steps,
            )
        )

    return {
        "learned_planner_only": aggregate_rollouts(
            learned_only,
            goals,
            tolerance=sim_config.goal_tolerance,
        ),
        "learned_planner_plus_collision_shield": aggregate_rollouts(
            learned_shielded,
            goals,
            tolerance=sim_config.goal_tolerance,
        ),
    }


def _straight_line_policy(sim_config: SimConfig):
    def _policy(positions, velocities, goals, radii):
        del velocities, radii
        return straight_line_velocity(positions, goals, sim_config)

    return _policy


def _rollout_scalar_summary(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "success": bool(result["success"]),
        "termination_reason": str(result.get("termination_reason", "")),
        "failure_flags": dict(result.get("failure_flags", {})),
        "steps": int(result["steps"]),
        "time_to_goal": float(result["time_to_goal"]),
        "mean_final_distance_to_goal": float(result.get("mean_final_distance_to_goal", 0.0)),
        "max_final_distance_to_goal": float(result.get("max_final_distance_to_goal", 0.0)),
        "fraction_agents_within_goal_tolerance": float(
            result.get("fraction_agents_within_goal_tolerance", 0.0)
        ),
        "no_progress_deadlock": bool(result.get("no_progress_deadlock", False)),
        "mean_recent_progress": float(result.get("mean_recent_progress", 0.0)),
        "collision_steps": int(result["collision_steps"]),
        "pair_collisions": int(result["pair_collisions"]),
        "obstacle_collision_steps": int(result.get("obstacle_collision_steps", 0)),
        "obstacle_collisions": int(result.get("obstacle_collisions", 0)),
        "obstacle_motion_hits": int(result.get("obstacle_motion_hits", 0)),
        "mean_min_separation_violation": float(
            result.get("mean_min_separation_violation", 0.0)
        ),
        "max_min_separation_violation": float(
            result.get("max_min_separation_violation", 0.0)
        ),
        "mean_obstacle_separation_violation": float(
            result.get("mean_obstacle_separation_violation", 0.0)
        ),
        "max_obstacle_separation_violation": float(
            result.get("max_obstacle_separation_violation", 0.0)
        ),
        "mean_shield_correction_norm": float(result.get("mean_shield_correction_norm", 0.0)),
        "max_shield_correction_norm": float(result.get("max_shield_correction_norm", 0.0)),
        "correction_needed_rate": float(result.get("correction_needed_rate", 0.0)),
        "mean_correction_target_norm": float(result.get("mean_correction_target_norm", 0.0)),
        "max_correction_target_norm": float(result.get("max_correction_target_norm", 0.0)),
        "obstacle_intervention_rate": float(result.get("obstacle_intervention_rate", 0.0)),
        "pairwise_intervention_rate": float(result.get("pairwise_intervention_rate", 0.0)),
    }


def _evaluate_named_shield_ablations(
    policy: Callable,
    sim_config: SimConfig,
    named_scenarios: Iterable[NamedScenario],
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
    phase: str = "shield_ablations",
    notes: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Evaluate shield variants on named scenarios."""

    named_scenarios = list(named_scenarios)
    canonical_variants = [
        canonical_shield_variant(variant)
        for variant in (variants or DEFAULT_SHIELD_VARIANTS)
    ]

    results: Dict[str, Dict[str, object]] = {}
    metrics_by_variant: Dict[str, Dict[str, float]] = {}
    for variant in canonical_variants:
        shield = make_shield(
            variant,
            sim_config,
            max_iterations=max_iterations,
            damping=damping,
        )
        rollouts = []
        goals = []
        per_scenario: Dict[str, Dict[str, object]] = {}
        for named in named_scenarios:
            result = rollout(
                named.scenario,
                sim_config,
                policy,
                shield=shield,
                max_steps=sim_config.max_steps,
            )
            rollouts.append(result)
            goals.append(named.scenario.goals)
            per_scenario[named.name] = _rollout_scalar_summary(result)

        metrics = aggregate_rollouts(
            rollouts,
            goals,
            tolerance=sim_config.goal_tolerance,
        )
        metrics_by_variant[variant] = metrics
        results[variant] = {
            "metrics": metrics,
            "scenarios": per_scenario,
        }

    return {
        "phase": phase,
        "variant_order": canonical_variants,
        "scenario_names": [named.name for named in named_scenarios],
        "metrics": metrics_by_variant,
        "results": results,
        "notes": list(notes or ()),
        "sim_config": sim_config.to_dict(),
    }


def evaluate_phase2_ablations(
    policy: Callable,
    sim_config: SimConfig,
    scenarios: Optional[Iterable[NamedScenario]] = None,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
) -> Dict[str, object]:
    """Evaluate shield variants on adversarial phase 2 scenarios."""

    return _evaluate_named_shield_ablations(
        policy,
        sim_config,
        named_scenarios=scenarios or phase2_adversarial_scenarios(sim_config),
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
        phase="phase2_shield_ablations",
        notes=[
            "The velocity_clip variant is explicit and measurable, but the simulator also enforces max_speed at step time.",
            "The pibt variant is a one-step continuous priority inheritance/backtracking prototype; unresolved final conflicts remain visible in diagnostics and metrics.",
            "The corridor_bottleneck scenario emulates a narrow passage with starts/goals because static obstacles are not active in phase 2.",
        ],
    )


def run_phase2_ablations(
    output_path: Optional[str | Path] = None,
    sim_config: Optional[SimConfig] = None,
    model: Optional[NumpyAttentionPolicy] = None,
    max_neighbors: int = 8,
    variants: Optional[Iterable[str]] = None,
    circle_agents: int = 8,
    bottleneck_agents: int = 6,
    max_iterations: int = 12,
    damping: float = 1.0,
) -> Dict[str, object]:
    """Run phase 2 shield ablations and optionally write JSON results."""

    sim_config = sim_config or SimConfig()
    if model is None:
        policy = _straight_line_policy(sim_config)
        policy_name = "straight_line_intent"
    else:
        policy = policy_from_model(model, max_neighbors, sim_config)
        policy_name = "learned_attention_policy"

    result = evaluate_phase2_ablations(
        policy,
        sim_config,
        scenarios=phase2_adversarial_scenarios(
            sim_config,
            circle_agents=circle_agents,
            bottleneck_agents=bottleneck_agents,
        ),
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
    )
    result["policy"] = policy_name
    result["max_neighbors"] = int(max_neighbors)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output_path"] = str(path)
    return result


def _obstacle_map_from_config(dataset_config: DatasetConfig):
    return load_obstacle_map(
        dataset_config.map_path,
        cell_size=dataset_config.map_cell_size,
    )


def _scenario_source_diagnostics(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
) -> Dict[str, object]:
    if dataset_config.scenario_source != "scen" or dataset_config.scen_path is None:
        return {"scenario_source": dataset_config.scenario_source}
    obstacle_map = _obstacle_map_from_config(dataset_config)
    if obstacle_map is None:
        return {
            "scenario_source": dataset_config.scenario_source,
            "scen_path": dataset_config.scen_path,
            "raw_tasks": 0,
            "valid_tasks": 0,
            "skipped_tasks": 0,
            "skip_reason": "missing obstacle map",
        }
    scenario_config = replace(sim_config, world_size=obstacle_map.world_size)
    return scen_task_diagnostics(dataset_config, scenario_config, obstacle_map)


def _policy_for_dataset(model, dataset_config: DatasetConfig, sim_config: SimConfig):
    obstacle_map = _obstacle_map_from_config(dataset_config)
    policy_sim_config = (
        replace(sim_config, world_size=obstacle_map.world_size)
        if obstacle_map is not None
        else sim_config
    )
    return policy_from_model(
        model,
        dataset_config.max_neighbors,
        policy_sim_config,
        obstacle_map=obstacle_map,
        max_obstacle_tokens=dataset_config.max_obstacle_tokens,
        obstacle_context_range=dataset_config.obstacle_context_range,
        observation_version=dataset_config.observation_version,
    )


def _expert_waypoint_policy(
    sim_config: SimConfig,
    obstacle_map,
    expert_type: str = "independent_astar",
):
    def _policy(positions, velocities, goals, radii):
        del velocities
        if obstacle_map is None:
            return straight_line_velocity(positions, goals, sim_config)
        return obstacle_map_expert_velocity(
            positions,
            goals,
            sim_config,
            obstacle_map,
            radii=radii,
            expert_type=expert_type,
        )

    return _policy


def _expert_waypoint_baseline(
    sim_config: SimConfig,
    named_scenarios: Iterable[NamedScenario],
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
    expert_type: str = "independent_astar",
) -> Dict[str, object]:
    named_scenarios = list(named_scenarios)
    if not named_scenarios or named_scenarios[0].scenario.obstacle_map is None:
        return {}
    policy_sim = sim_config_for_scenario(sim_config, named_scenarios[0].scenario)
    return _evaluate_named_shield_ablations(
        _expert_waypoint_policy(
            policy_sim,
            named_scenarios[0].scenario.obstacle_map,
            expert_type=expert_type,
        ),
        sim_config,
        named_scenarios=named_scenarios,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
        phase="expert_waypoint_baseline",
        notes=[
            f"Expert baseline rolls out {expert_type} waypoint targets directly without training.",
        ],
    )


def _learned_vs_expert_comparison(
    learned_metrics: Dict[str, Dict[str, float]],
    expert_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    keys = (
        "success_rate",
        "deadlock_rate",
        "mean_time_to_goal",
        "collision_rate",
        "pair_collisions_per_run",
        "obstacle_collision_rate",
        "obstacle_collisions_per_run",
        "mean_obstacle_separation_violation",
        "max_obstacle_separation_violation",
    )
    comparison: Dict[str, Dict[str, float]] = {}
    for variant, learned in learned_metrics.items():
        expert = expert_metrics.get(variant)
        if expert is None:
            continue
        comparison[variant] = {
            f"{key}_delta_learned_minus_expert": float(
                learned.get(key, 0.0) - expert.get(key, 0.0)
            )
            for key in keys
        }
    return comparison


def _phase4_notes(dataset_config: Optional[DatasetConfig] = None) -> list[str]:
    notes = [
        "The pibt variant remains a one-step continuous priority inheritance/backtracking prototype; unresolved final conflicts remain visible in diagnostics and metrics.",
        "The NumPy transformer policy uses a fixed random attention encoder with a trained ridge-regression output head because PyTorch is not a repo dependency in this environment.",
    ]
    if dataset_config is not None and dataset_config.scenario_type == "obstacle_map":
        notes.insert(
            0,
            "Phase 4 obstacle scaling is enabled through the Phase 3 Moving AI map path; sampled maps and Moving AI .scen task sources are both available for one static map at a time.",
        )
    else:
        notes.insert(
            0,
            "Phase 4 defaults to scaled empty maps; obstacle-map scaling is available with scenario_type='obstacle_map' plus a Moving AI map_path.",
        )
    return notes


def _phase4_eval_dataset_config(
    dataset_config: DatasetConfig,
    eval_scenarios: int,
    seed: Optional[int],
) -> DatasetConfig:
    return DatasetConfig(
        num_scenarios=eval_scenarios,
        num_agents=dataset_config.num_agents,
        horizon=dataset_config.horizon,
        max_neighbors=dataset_config.max_neighbors,
        min_start_goal_distance=dataset_config.min_start_goal_distance,
        scenario_type=dataset_config.scenario_type,
        map_path=dataset_config.map_path,
        map_cell_size=dataset_config.map_cell_size,
        scenario_source=dataset_config.scenario_source,
        scen_path=dataset_config.scen_path,
        scen_limit=dataset_config.scen_limit,
        max_obstacle_tokens=dataset_config.max_obstacle_tokens,
        obstacle_context_range=dataset_config.obstacle_context_range,
        observation_version=dataset_config.observation_version,
        expert_type=dataset_config.expert_type,
        include_auxiliary_targets=dataset_config.include_auxiliary_targets,
        seed=dataset_config.seed + 10_000 if seed is None else seed,
        include_reached_agents=dataset_config.include_reached_agents,
        max_samples=None,
        agent_count_choices=dataset_config.agent_count_choices,
    )


def phase4_scaled_scenarios(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
) -> list[NamedScenario]:
    """Generate named scaled scenarios for phase 4 evaluation."""

    scenarios = generate_scenarios(dataset_config, sim_config)
    return [
        NamedScenario(
            (
                f"scaled_obstacle_map_{index:03d}_{scenario.num_agents}_agents"
                if scenario.obstacle_map is not None
                else f"scaled_empty_{index:03d}_{scenario.num_agents}_agents"
            ),
            scenario,
        )
        for index, scenario in enumerate(scenarios)
    ]


def evaluate_phase4_ablations(
    model,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
) -> Dict[str, object]:
    """Evaluate a trained phase 4 policy across scaled scenarios and shields."""

    policy = _policy_for_dataset(model, dataset_config, sim_config)
    named_scenarios = phase4_scaled_scenarios(dataset_config, sim_config)
    map_info = named_scenarios[0].scenario.map_metadata() if named_scenarios else None
    result = _evaluate_named_shield_ablations(
        policy,
        sim_config,
        named_scenarios=named_scenarios,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
        phase="phase4_scaled_shield_ablations",
        notes=_phase4_notes(dataset_config),
    )
    result["dataset_config"] = dataset_config.to_dict()
    result["expert_type"] = normalize_expert_type(dataset_config.expert_type)
    result["map_metadata"] = map_info
    result["scenario_source_diagnostics"] = _scenario_source_diagnostics(
        dataset_config,
        sim_config,
    )
    result["max_neighbors"] = int(dataset_config.max_neighbors)
    result["policy"] = model.__class__.__name__
    if map_info is not None:
        expert = _expert_waypoint_baseline(
            sim_config,
            named_scenarios,
            variants=variants,
            max_iterations=max_iterations,
            damping=damping,
            expert_type=dataset_config.expert_type,
        )
        result["expert_waypoint_baseline"] = expert
        result["learned_vs_expert"] = _learned_vs_expert_comparison(
            result["metrics"],
            expert.get("metrics", {}),
        )
    return result


def _phase3_notes() -> list[str]:
    return [
        "Phase 3 uses static Moving AI type=octile maps with blocked cells converted to continuous unit-cell AABBs.",
        "The obstacle expert is a per-agent octile A* planner with continuous waypoint following; it does not solve full multi-agent CBS/ECBS over obstacles.",
        "Obstacle-aware shields constrain one-step commands against static obstacles and keep reporting any unresolved obstacle or agent-agent conflicts in diagnostics and metrics.",
        "Moving AI .scen benchmark rows can be imported as obstacle-map start/goal tasks; invalid tasks for the current radius/clearance are skipped during scenario construction.",
    ]


def phase3_obstacle_scenarios(
    dataset_config: DatasetConfig,
    sim_config: SimConfig,
) -> list[NamedScenario]:
    """Generate named obstacle-map scenarios for phase 3 evaluation."""

    scenarios = generate_scenarios(dataset_config, sim_config)
    return [
        NamedScenario(
            f"phase3_{scenario.obstacle_map.name if scenario.obstacle_map else 'map'}_{index:03d}_{scenario.num_agents}_agents",
            scenario,
        )
        for index, scenario in enumerate(scenarios)
    ]


def evaluate_phase3_ablations(
    model,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
) -> Dict[str, object]:
    """Evaluate a trained phase 3 obstacle-map policy across shield variants."""

    if dataset_config.scenario_type != "obstacle_map" and dataset_config.map_path is None:
        raise ValueError("Phase 3 evaluation requires an obstacle_map dataset_config.")
    policy = _policy_for_dataset(model, dataset_config, sim_config)
    named_scenarios = phase3_obstacle_scenarios(dataset_config, sim_config)
    map_info = named_scenarios[0].scenario.map_metadata() if named_scenarios else None
    result = _evaluate_named_shield_ablations(
        policy,
        sim_config,
        named_scenarios=named_scenarios,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
        phase="phase3_obstacle_map_shield_ablations",
        notes=_phase3_notes(),
    )
    result["dataset_config"] = dataset_config.to_dict()
    result["expert_type"] = normalize_expert_type(dataset_config.expert_type)
    result["map_metadata"] = map_info
    result["scenario_source_diagnostics"] = _scenario_source_diagnostics(
        dataset_config,
        sim_config,
    )
    result["max_neighbors"] = int(dataset_config.max_neighbors)
    result["policy"] = model.__class__.__name__
    expert = _expert_waypoint_baseline(
        sim_config,
        named_scenarios,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
        expert_type=dataset_config.expert_type,
    )
    result["expert_waypoint_baseline"] = expert
    result["learned_vs_expert"] = _learned_vs_expert_comparison(
        result["metrics"],
        expert.get("metrics", {}),
    )
    return result


def run_phase3_experiment(
    output_dir: str | Path,
    map_path: str | Path,
    sim_config: Optional[SimConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    model_config: Optional[ModelConfig] = None,
    eval_scenarios: int = 8,
    eval_seed: Optional[int] = None,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
    verbose: bool = False,
) -> Dict[str, object]:
    """Run Phase 3 map data generation, training, evaluation, and ablations."""

    map_path = Path(map_path)
    sim_config = sim_config or SimConfig(max_steps=160)
    dataset_config = dataset_config or DatasetConfig(
        num_scenarios=24,
        num_agents=8,
        horizon=80,
        max_neighbors=8,
        min_start_goal_distance=3.0,
        scenario_type="obstacle_map",
        map_path=str(map_path),
        max_obstacle_tokens=8,
        obstacle_context_range=4.0,
        seed=307,
        max_samples=50_000,
    )
    dataset_config = replace(
        dataset_config,
        scenario_type="obstacle_map",
        map_path=str(map_path),
        max_obstacle_tokens=max(1, int(dataset_config.max_obstacle_tokens)),
        observation_version="obstacle_waypoint_v2",
        expert_type=normalize_expert_type(dataset_config.expert_type),
    )
    model_config = model_config or ModelConfig(
        d_model=32,
        policy_type="numpy_transformer",
        num_heads=4,
        num_layers=1,
        batch_size=512,
        ridge_lambda=1e-3,
        seed=dataset_config.seed + 17,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = backend_diagnostics(model_config.policy_type)
    dataset = build_dataset(dataset_config, sim_config)
    if model_config.feature_dim != int(dataset.observations.shape[2]):
        model_config = replace(model_config, feature_dim=int(dataset.observations.shape[2]))
    dataset_path = output_dir / "phase3_obstacle_dataset.npz"
    dataset.save(dataset_path)

    model, history = train_phase4_model(
        dataset,
        sim_config,
        model_config,
        verbose=verbose,
    )
    map_info = map_metadata(_obstacle_map_from_config(dataset_config))
    model_path = output_dir / "phase3_policy.npz"
    model.save(
        model_path,
        metadata={
            "sim_config": sim_config.to_dict(),
            "dataset_config": dataset_config.to_dict(),
            "model_config": model_config.to_dict(),
            "history": history,
            "backend_diagnostics": diagnostics,
            "map_metadata": map_info,
            "observation_version": dataset_config.observation_version,
            "observation_metadata": dataset.dataset_config.get("observation_metadata"),
            "auxiliary_target_metadata": dataset.dataset_config.get("auxiliary_target_metadata"),
            "expert_type": dataset_config.expert_type,
            "astar_cache_info": astar_cache_info(),
            "notes": _phase3_notes(),
        },
    )

    eval_config = replace(
        dataset_config,
        num_scenarios=eval_scenarios,
        seed=dataset_config.seed + 10_000 if eval_seed is None else eval_seed,
        max_samples=None,
    )
    ablations = evaluate_phase3_ablations(
        model,
        sim_config,
        eval_config,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
    )

    result_path = output_dir / "phase3_results.json"
    result = {
        "phase": "phase3_obstacle_maps",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "results_path": str(result_path),
        "num_training_samples": dataset.num_samples,
        "dataset_shape": {
            "observations": list(dataset.observations.shape),
            "masks": list(dataset.masks.shape),
            "targets": list(dataset.targets.shape),
        },
        "observation_version": dataset_config.observation_version,
        "observation_metadata": dataset.dataset_config.get("observation_metadata"),
        "auxiliary_target_metadata": dataset.dataset_config.get("auxiliary_target_metadata"),
        "expert_type": dataset_config.expert_type,
        "history": history,
        "metrics": ablations["metrics"],
        "ablations": ablations,
        "expert_waypoint_baseline": ablations.get("expert_waypoint_baseline", {}),
        "learned_vs_expert": ablations.get("learned_vs_expert", {}),
        "backend_diagnostics": diagnostics,
        "astar_cache_info": astar_cache_info(),
        "notes": _phase3_notes(),
        "sim_config": sim_config.to_dict(),
        "dataset_config": dataset_config.to_dict(),
        "eval_dataset_config": eval_config.to_dict(),
        "scenario_source_diagnostics": _scenario_source_diagnostics(
            dataset_config,
            sim_config,
        ),
        "model_config": model_config.to_dict(),
        "map_metadata": map_info,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_phase4_evaluation(
    model_path: str | Path,
    output_path: str | Path,
    sim_config: SimConfig,
    dataset_config: DatasetConfig,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
) -> Dict[str, object]:
    """Evaluate an existing phase 1/4 model on phase 4 scaled shield variants."""

    model = load_policy(model_path)
    result = evaluate_phase4_ablations(
        model,
        sim_config,
        dataset_config,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
    )
    result["model_path"] = str(model_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(path)
    return result


def run_phase4_experiment(
    output_dir: str | Path,
    sim_config: Optional[SimConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    model_config: Optional[ModelConfig] = None,
    eval_scenarios: int = 12,
    eval_seed: Optional[int] = None,
    variants: Optional[Iterable[str]] = None,
    max_iterations: int = 12,
    damping: float = 1.0,
    verbose: bool = False,
) -> Dict[str, object]:
    """Run phase 4 scaled data generation, training, and shield ablations."""

    sim_config = sim_config or SimConfig(world_size=(32.0, 32.0), max_steps=240)
    dataset_config = dataset_config or DatasetConfig(
        num_scenarios=48,
        num_agents=32,
        horizon=120,
        max_neighbors=16,
        min_start_goal_distance=6.0,
        seed=4007,
        max_samples=200_000,
    )
    model_config = model_config or ModelConfig(
        d_model=64,
        policy_type="numpy_transformer",
        num_heads=4,
        num_layers=2,
        batch_size=2048,
        ridge_lambda=1e-3,
        seed=dataset_config.seed + 17,
    )
    if dataset_config.scenario_type == "obstacle_map" and dataset_config.observation_version == "legacy":
        dataset_config = replace(dataset_config, observation_version="obstacle_waypoint_v2")
    dataset_config = replace(
        dataset_config,
        expert_type=normalize_expert_type(dataset_config.expert_type),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = backend_diagnostics(model_config.policy_type)
    dataset = build_dataset(dataset_config, sim_config)
    if model_config.feature_dim != int(dataset.observations.shape[2]):
        model_config = replace(model_config, feature_dim=int(dataset.observations.shape[2]))
    dataset_path = output_dir / "phase4_scaled_dataset.npz"
    dataset.save(dataset_path)

    model, history = train_phase4_model(
        dataset,
        sim_config,
        model_config,
        verbose=verbose,
    )
    model_path = output_dir / "phase4_policy.npz"
    model.save(
        model_path,
        metadata={
            "sim_config": sim_config.to_dict(),
            "dataset_config": dataset_config.to_dict(),
            "model_config": model_config.to_dict(),
            "history": history,
            "backend_diagnostics": diagnostics,
            "observation_version": dataset_config.observation_version,
            "observation_metadata": dataset.dataset_config.get("observation_metadata"),
            "auxiliary_target_metadata": dataset.dataset_config.get("auxiliary_target_metadata"),
            "expert_type": dataset_config.expert_type,
            "astar_cache_info": astar_cache_info(),
            "notes": _phase4_notes(dataset_config),
        },
    )

    eval_config = _phase4_eval_dataset_config(
        dataset_config,
        eval_scenarios=eval_scenarios,
        seed=eval_seed,
    )
    ablations = evaluate_phase4_ablations(
        model,
        sim_config,
        eval_config,
        variants=variants,
        max_iterations=max_iterations,
        damping=damping,
    )

    result_path = output_dir / "phase4_results.json"
    result = {
        "phase": "phase4_scaling_model_data_experiments",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "results_path": str(result_path),
        "num_training_samples": dataset.num_samples,
        "dataset_shape": {
            "observations": list(dataset.observations.shape),
            "masks": list(dataset.masks.shape),
            "targets": list(dataset.targets.shape),
        },
        "observation_version": dataset_config.observation_version,
        "observation_metadata": dataset.dataset_config.get("observation_metadata"),
        "auxiliary_target_metadata": dataset.dataset_config.get("auxiliary_target_metadata"),
        "expert_type": dataset_config.expert_type,
        "history": history,
        "metrics": ablations["metrics"],
        "ablations": ablations,
        "expert_waypoint_baseline": ablations.get("expert_waypoint_baseline", {}),
        "learned_vs_expert": ablations.get("learned_vs_expert", {}),
        "backend_diagnostics": diagnostics,
        "astar_cache_info": astar_cache_info(),
        "notes": _phase4_notes(dataset_config),
        "sim_config": sim_config.to_dict(),
        "dataset_config": dataset_config.to_dict(),
        "eval_dataset_config": eval_config.to_dict(),
        "map_metadata": ablations.get("map_metadata"),
        "scenario_source_diagnostics": _scenario_source_diagnostics(
            dataset_config,
            sim_config,
        ),
        "model_config": model_config.to_dict(),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_phase1_experiment(
    output_dir: str | Path,
    sim_config: Optional[SimConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    model_config: Optional[ModelConfig] = None,
    eval_scenarios: int = 32,
    verbose: bool = False,
) -> Dict[str, object]:
    """Run data generation, supervised training, and shielded evaluation."""

    sim_config = sim_config or SimConfig()
    dataset_config = dataset_config or DatasetConfig()
    model_config = model_config or ModelConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(dataset_config, sim_config)
    dataset_path = output_dir / "phase1_empty_dataset.npz"
    dataset.save(dataset_path)

    model, history = train_phase1_model(
        dataset,
        sim_config,
        model_config,
        verbose=verbose,
    )
    model_path = output_dir / "phase1_attention_policy.npz"
    model.save(
        model_path,
        metadata={
            "sim_config": sim_config.to_dict(),
            "dataset_config": dataset_config.to_dict(),
            "model_config": model_config.to_dict(),
            "history": history,
            "observation_version": dataset_config.observation_version,
            "observation_metadata": dataset.dataset_config.get("observation_metadata"),
        },
    )

    metrics = evaluate_phase1_model(
        model,
        sim_config,
        dataset_config,
        eval_scenarios=eval_scenarios,
        seed=dataset_config.seed + 2000,
    )

    result = {
        "phase": "phase1_minimal_empty_map",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "num_training_samples": dataset.num_samples,
        "history": history,
        "metrics": metrics,
        "sim_config": sim_config.to_dict(),
        "dataset_config": dataset_config.to_dict(),
        "model_config": model_config.to_dict(),
    }
    result_path = output_dir / "phase1_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
