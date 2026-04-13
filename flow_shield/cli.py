"""Command line entry points for continuous MAPF research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import build_benchmark_plan, run_benchmark_plan
from .config import DatasetConfig, ModelConfig, SimConfig
from .dataset import build_dataset, load_dataset
from .experiment import (
    evaluate_phase1_model,
    run_phase1_experiment,
    run_phase2_ablations,
    run_phase3_experiment,
    run_phase4_evaluation,
    run_phase4_experiment,
    train_phase1_model,
)
from .model import load_policy
from .shield import DEFAULT_SHIELD_VARIANTS


def _observation_version_for_args(args: argparse.Namespace, scenario_type: str) -> str:
    requested = getattr(args, "observation_version", "auto")
    if requested == "auto":
        return "obstacle_waypoint_v2" if scenario_type == "obstacle_map" else "legacy"
    return requested


def _optional_sample_cap(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _add_map_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenario-type",
        choices=("empty", "obstacle_map", "moving_ai"),
        default="empty",
        help="Scenario source. 'moving_ai' is accepted as an alias for obstacle_map.",
    )
    parser.add_argument("--map", dest="map_path", type=Path, default=None)
    parser.add_argument("--map-cell-size", type=float, default=1.0)
    parser.add_argument(
        "--scenario-source",
        choices=("sampled", "scen"),
        default="sampled",
    )
    parser.add_argument("--scen", dest="scen_path", type=Path, default=None)
    parser.add_argument("--scen-limit", type=int, default=None)
    parser.add_argument("--max-obstacle-tokens", type=int, default=0)
    parser.add_argument("--obstacle-context-range", type=float, default=4.0)
    parser.add_argument(
        "--observation-version",
        choices=("auto", "legacy", "obstacle_waypoint_v2"),
        default="auto",
    )
    parser.add_argument(
        "--expert-type",
        choices=("independent_astar", "prioritized_astar"),
        default="independent_astar",
    )
    parser.add_argument(
        "--include-auxiliary-targets",
        action="store_true",
        help="Store optional Phase 5 shield-correction supervision targets in generated datasets.",
    )


def _add_shared_sim_args(
    parser: argparse.ArgumentParser,
    num_agents: int = 8,
    world_size: float = 10.0,
    max_steps: int = 160,
    max_neighbors: int = 8,
) -> None:
    parser.add_argument("--num-agents", type=int, default=num_agents)
    parser.add_argument("--world-size", type=float, default=world_size)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-speed", type=float, default=1.2)
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--max-neighbors", type=int, default=max_neighbors)


def _sim_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        world_size=(args.world_size, args.world_size),
        dt=args.dt,
        agent_radius=args.agent_radius,
        max_speed=args.max_speed,
        max_steps=args.max_steps,
    )


def _dataset_config(args: argparse.Namespace) -> DatasetConfig:
    scenario_type = getattr(args, "scenario_type", "empty")
    if scenario_type == "moving_ai":
        scenario_type = "obstacle_map"
    max_obstacle_tokens = getattr(args, "max_obstacle_tokens", 0)
    if scenario_type == "obstacle_map" and max_obstacle_tokens <= 0:
        max_obstacle_tokens = 8
    return DatasetConfig(
        num_scenarios=args.num_scenarios,
        num_agents=args.num_agents,
        horizon=args.horizon,
        max_neighbors=args.max_neighbors,
        min_start_goal_distance=getattr(args, "min_start_goal_distance", 3.0),
        scenario_type=scenario_type,
        scenario_source=getattr(args, "scenario_source", "sampled"),
        map_path=(
            str(args.map_path)
            if getattr(args, "map_path", None) is not None
            else None
        ),
        map_cell_size=getattr(args, "map_cell_size", 1.0),
        scen_path=(
            str(args.scen_path)
            if getattr(args, "scen_path", None) is not None
            else None
        ),
        scen_limit=getattr(args, "scen_limit", None),
        max_obstacle_tokens=max_obstacle_tokens,
        obstacle_context_range=getattr(args, "obstacle_context_range", 4.0),
        observation_version=_observation_version_for_args(args, scenario_type),
        expert_type=getattr(args, "expert_type", "independent_astar"),
        include_auxiliary_targets=getattr(args, "include_auxiliary_targets", False),
        seed=args.seed,
        max_samples=getattr(args, "max_samples", None),
    )


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        d_model=args.d_model,
        policy_type=getattr(args, "policy_type", "numpy_attention"),
        num_heads=getattr(args, "num_heads", 1),
        num_layers=getattr(args, "num_layers", 1),
        epochs=getattr(args, "epochs", 40),
        batch_size=args.batch_size,
        learning_rate=getattr(args, "learning_rate", 2e-3),
        ridge_lambda=getattr(args, "ridge_lambda", 1e-3),
        seed=args.seed + 17,
    )


def _phase4_train_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    scenario_type = getattr(args, "scenario_type", "empty")
    if scenario_type == "moving_ai":
        scenario_type = "obstacle_map"
    max_obstacle_tokens = getattr(args, "max_obstacle_tokens", 0)
    if scenario_type == "obstacle_map" and max_obstacle_tokens <= 0:
        max_obstacle_tokens = 8
    return DatasetConfig(
        num_scenarios=args.train_scenarios,
        num_agents=args.num_agents,
        horizon=args.horizon,
        max_neighbors=args.max_neighbors,
        min_start_goal_distance=args.min_start_goal_distance,
        scenario_type=scenario_type,
        scenario_source=getattr(args, "scenario_source", "sampled"),
        map_path=(
            str(args.map_path)
            if getattr(args, "map_path", None) is not None
            else None
        ),
        map_cell_size=getattr(args, "map_cell_size", 1.0),
        scen_path=(
            str(args.scen_path)
            if getattr(args, "scen_path", None) is not None
            else None
        ),
        scen_limit=getattr(args, "scen_limit", None),
        max_obstacle_tokens=max_obstacle_tokens,
        obstacle_context_range=getattr(args, "obstacle_context_range", 4.0),
        observation_version=_observation_version_for_args(args, scenario_type),
        expert_type=getattr(args, "expert_type", "independent_astar"),
        include_auxiliary_targets=getattr(args, "include_auxiliary_targets", False),
        seed=args.seed,
        max_samples=args.max_samples,
    )


def _phase4_eval_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    scenario_type = getattr(args, "scenario_type", "empty")
    if scenario_type == "moving_ai":
        scenario_type = "obstacle_map"
    max_obstacle_tokens = getattr(args, "max_obstacle_tokens", 0)
    if scenario_type == "obstacle_map" and max_obstacle_tokens <= 0:
        max_obstacle_tokens = 8
    return DatasetConfig(
        num_scenarios=args.eval_scenarios,
        num_agents=args.num_agents,
        horizon=args.horizon,
        max_neighbors=args.max_neighbors,
        min_start_goal_distance=args.min_start_goal_distance,
        scenario_type=scenario_type,
        scenario_source=getattr(args, "scenario_source", "sampled"),
        map_path=(
            str(args.map_path)
            if getattr(args, "map_path", None) is not None
            else None
        ),
        map_cell_size=getattr(args, "map_cell_size", 1.0),
        scen_path=(
            str(args.scen_path)
            if getattr(args, "scen_path", None) is not None
            else None
        ),
        scen_limit=getattr(args, "scen_limit", None),
        max_obstacle_tokens=max_obstacle_tokens,
        obstacle_context_range=getattr(args, "obstacle_context_range", 4.0),
        observation_version=_observation_version_for_args(args, scenario_type),
        expert_type=getattr(args, "expert_type", "independent_astar"),
        include_auxiliary_targets=getattr(args, "include_auxiliary_targets", False),
        seed=args.seed,
    )


def _phase3_train_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        num_scenarios=args.train_scenarios,
        num_agents=args.num_agents,
        horizon=args.horizon,
        max_neighbors=args.max_neighbors,
        min_start_goal_distance=args.min_start_goal_distance,
        scenario_type="obstacle_map",
        scenario_source=getattr(args, "scenario_source", "sampled"),
        map_path=str(args.map_path),
        map_cell_size=args.map_cell_size,
        scen_path=(
            str(args.scen_path)
            if getattr(args, "scen_path", None) is not None
            else None
        ),
        scen_limit=getattr(args, "scen_limit", None),
        max_obstacle_tokens=args.max_obstacle_tokens,
        obstacle_context_range=args.obstacle_context_range,
        observation_version=_observation_version_for_args(args, "obstacle_map"),
        expert_type=getattr(args, "expert_type", "independent_astar"),
        include_auxiliary_targets=getattr(args, "include_auxiliary_targets", False),
        seed=args.seed,
        max_samples=args.max_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous MAPF learned planner + collision shield research CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="Generate phase 0 expert data.")
    _add_shared_sim_args(generate)
    generate.add_argument("--num-scenarios", type=int, default=128)
    generate.add_argument("--horizon", type=int, default=80)
    generate.add_argument("--min-start-goal-distance", type=float, default=3.0)
    generate.add_argument("--max-samples", type=_optional_sample_cap, default=None)
    generate.add_argument("--seed", type=int, default=7)
    _add_map_dataset_args(generate)
    generate.add_argument("--output", type=Path, default=Path("datasets/phase1_empty_train.npz"))

    train = subparsers.add_parser("train", help="Train the phase 1 attention policy.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model-out", type=Path, default=Path("artifacts/phase1_attention_policy.npz"))
    train.add_argument("--d-model", type=int, default=32)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--max-speed", type=float, default=1.2)
    train.add_argument("--verbose", action="store_true")

    evaluate = subparsers.add_parser("eval", help="Evaluate learned-only vs shielded policy.")
    _add_shared_sim_args(evaluate)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--num-scenarios", type=int, default=32)
    evaluate.add_argument("--horizon", type=int, default=80)
    evaluate.add_argument("--min-start-goal-distance", type=float, default=3.0)
    evaluate.add_argument("--seed", type=int, default=1007)
    _add_map_dataset_args(evaluate)
    evaluate.add_argument("--output", type=Path, default=Path("results/phase1_metrics.json"))

    phase1 = subparsers.add_parser("phase1", help="Run phase 0 data + phase 1 train/eval.")
    _add_shared_sim_args(phase1)
    phase1.add_argument("--num-scenarios", type=int, default=128)
    phase1.add_argument("--eval-scenarios", type=int, default=32)
    phase1.add_argument("--horizon", type=int, default=80)
    phase1.add_argument("--seed", type=int, default=7)
    phase1.add_argument("--d-model", type=int, default=32)
    phase1.add_argument("--epochs", type=int, default=40)
    phase1.add_argument("--batch-size", type=int, default=256)
    phase1.add_argument("--learning-rate", type=float, default=2e-3)
    phase1.add_argument("--output-dir", type=Path, default=Path("runs/phase1"))
    phase1.add_argument("--verbose", action="store_true")

    phase2 = subparsers.add_parser("phase2", help="Run phase 2 shield ablations.")
    _add_shared_sim_args(phase2)
    phase2.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional trained phase 1 model; defaults to straight-line intent policy.",
    )
    phase2.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_SHIELD_VARIANTS),
        help=f"Shield variants to evaluate. Defaults to: {', '.join(DEFAULT_SHIELD_VARIANTS)}.",
    )
    phase2.add_argument("--circle-agents", type=int, default=8)
    phase2.add_argument("--bottleneck-agents", type=int, default=6)
    phase2.add_argument("--max-iterations", type=int, default=12)
    phase2.add_argument("--damping", type=float, default=1.0)
    phase2.add_argument("--output", type=Path, default=Path("results/phase2_ablations.json"))

    phase3 = subparsers.add_parser(
        "phase3",
        help="Run phase 3 Moving AI obstacle-map data, training, and shield ablations.",
    )
    _add_shared_sim_args(
        phase3,
        num_agents=8,
        world_size=10.0,
        max_steps=160,
        max_neighbors=8,
    )
    phase3.add_argument("--map", dest="map_path", type=Path, required=True)
    phase3.add_argument("--map-cell-size", type=float, default=1.0)
    phase3.add_argument(
        "--scenario-source",
        choices=("sampled", "scen"),
        default="sampled",
    )
    phase3.add_argument("--scen", dest="scen_path", type=Path, default=None)
    phase3.add_argument("--scen-limit", type=int, default=None)
    phase3.add_argument("--train-scenarios", type=int, default=24)
    phase3.add_argument("--eval-scenarios", type=int, default=8)
    phase3.add_argument("--horizon", type=int, default=80)
    phase3.add_argument("--min-start-goal-distance", type=float, default=3.0)
    phase3.add_argument("--max-samples", type=_optional_sample_cap, default=50_000)
    phase3.add_argument("--max-obstacle-tokens", type=int, default=8)
    phase3.add_argument("--obstacle-context-range", type=float, default=4.0)
    phase3.add_argument(
        "--observation-version",
        choices=("auto", "legacy", "obstacle_waypoint_v2"),
        default="auto",
    )
    phase3.add_argument(
        "--expert-type",
        choices=("independent_astar", "prioritized_astar"),
        default="independent_astar",
    )
    phase3.add_argument(
        "--include-auxiliary-targets",
        action="store_true",
        help="Store optional Phase 5 shield-correction supervision targets in the Phase 3 dataset.",
    )
    phase3.add_argument("--seed", type=int, default=307)
    phase3.add_argument(
        "--policy-type",
        choices=("numpy_transformer", "numpy_attention"),
        default="numpy_transformer",
    )
    phase3.add_argument("--d-model", type=int, default=32)
    phase3.add_argument("--num-heads", type=int, default=4)
    phase3.add_argument("--num-layers", type=int, default=1)
    phase3.add_argument("--batch-size", type=int, default=512)
    phase3.add_argument("--ridge-lambda", type=float, default=1e-3)
    phase3.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_SHIELD_VARIANTS),
        help=f"Shield variants to evaluate. Defaults to: {', '.join(DEFAULT_SHIELD_VARIANTS)}.",
    )
    phase3.add_argument("--max-iterations", type=int, default=12)
    phase3.add_argument("--damping", type=float, default=1.0)
    phase3.add_argument("--output-dir", type=Path, default=Path("runs/phase3"))
    phase3.add_argument("--verbose", action="store_true")

    phase4 = subparsers.add_parser(
        "phase4",
        help="Run phase 4 scaled data generation, training, and shield ablations.",
    )
    _add_shared_sim_args(
        phase4,
        num_agents=32,
        world_size=32.0,
        max_steps=240,
        max_neighbors=16,
    )
    phase4.add_argument("--train-scenarios", type=int, default=48)
    phase4.add_argument("--eval-scenarios", type=int, default=12)
    phase4.add_argument("--horizon", type=int, default=120)
    phase4.add_argument("--min-start-goal-distance", type=float, default=6.0)
    phase4.add_argument("--max-samples", type=_optional_sample_cap, default=200_000)
    phase4.add_argument("--seed", type=int, default=4007)
    _add_map_dataset_args(phase4)
    phase4.add_argument(
        "--policy-type",
        choices=("numpy_transformer", "numpy_attention"),
        default="numpy_transformer",
    )
    phase4.add_argument("--d-model", type=int, default=64)
    phase4.add_argument("--num-heads", type=int, default=4)
    phase4.add_argument("--num-layers", type=int, default=2)
    phase4.add_argument("--batch-size", type=int, default=2048)
    phase4.add_argument("--ridge-lambda", type=float, default=1e-3)
    phase4.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_SHIELD_VARIANTS),
        help=f"Shield variants to evaluate. Defaults to: {', '.join(DEFAULT_SHIELD_VARIANTS)}.",
    )
    phase4.add_argument("--max-iterations", type=int, default=12)
    phase4.add_argument("--damping", type=float, default=1.0)
    phase4.add_argument("--output-dir", type=Path, default=Path("runs/phase4"))
    phase4.add_argument("--verbose", action="store_true")

    phase4_eval = subparsers.add_parser(
        "phase4-eval",
        help="Evaluate an existing model on phase 4 scaled shield ablations.",
    )
    _add_shared_sim_args(
        phase4_eval,
        num_agents=32,
        world_size=32.0,
        max_steps=240,
        max_neighbors=16,
    )
    phase4_eval.add_argument("--model", type=Path, required=True)
    phase4_eval.add_argument("--eval-scenarios", type=int, default=12)
    phase4_eval.add_argument("--horizon", type=int, default=120)
    phase4_eval.add_argument("--min-start-goal-distance", type=float, default=6.0)
    phase4_eval.add_argument("--seed", type=int, default=14007)
    _add_map_dataset_args(phase4_eval)
    phase4_eval.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_SHIELD_VARIANTS),
        help=f"Shield variants to evaluate. Defaults to: {', '.join(DEFAULT_SHIELD_VARIANTS)}.",
    )
    phase4_eval.add_argument("--max-iterations", type=int, default=12)
    phase4_eval.add_argument("--damping", type=float, default=1.0)
    phase4_eval.add_argument("--output", type=Path, default=Path("results/phase4_eval.json"))

    benchmark = subparsers.add_parser(
        "benchmark-obstacles",
        help="Build or run Phase 7 Moving AI obstacle benchmark plans.",
    )
    benchmark.add_argument("--map-scen-list", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/obstacles"))
    benchmark.add_argument("--agent-counts", type=_parse_int_list, default=(4,))
    benchmark.add_argument("--seeds", type=_parse_int_list, default=(7007,))
    benchmark.add_argument("--train-scenarios", type=int, default=24)
    benchmark.add_argument("--eval-scenarios", type=int, default=8)
    benchmark.add_argument("--horizon", type=int, default=80)
    benchmark.add_argument("--max-steps", type=int, default=160)
    benchmark.add_argument("--max-neighbors", type=int, default=8)
    benchmark.add_argument("--min-start-goal-distance", type=float, default=3.0)
    benchmark.add_argument("--max-samples", type=_optional_sample_cap, default=50_000)
    benchmark.add_argument("--max-obstacle-tokens", type=int, default=8)
    benchmark.add_argument("--obstacle-context-range", type=float, default=4.0)
    benchmark.add_argument(
        "--observation-version",
        choices=("legacy", "obstacle_waypoint_v2"),
        default="obstacle_waypoint_v2",
    )
    benchmark.add_argument(
        "--expert-type",
        choices=("independent_astar", "prioritized_astar"),
        default="independent_astar",
    )
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--smoke", action="store_true")
    benchmark.add_argument("--plan-only", "--dry-run", action="store_true", dest="plan_only")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-data":
        dataset = build_dataset(_dataset_config(args), _sim_config(args))
        dataset.save(args.output)
        print(f"wrote {dataset.num_samples} samples to {args.output}")
        return

    if args.command == "train":
        dataset = load_dataset(args.data)
        sim_config = SimConfig(max_speed=args.max_speed)
        model_config = _model_config(args)
        model, history = train_phase1_model(
            dataset,
            sim_config,
            model_config,
            verbose=args.verbose,
        )
        model.save(args.model_out, metadata={"history": history})
        print(json.dumps({"model": str(args.model_out), "history": history}, indent=2))
        return

    if args.command == "eval":
        model = load_policy(args.model)
        metrics = evaluate_phase1_model(
            model,
            _sim_config(args),
            _dataset_config(args),
            eval_scenarios=args.num_scenarios,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "phase1":
        result = run_phase1_experiment(
            output_dir=args.output_dir,
            sim_config=_sim_config(args),
            dataset_config=_dataset_config(args),
            model_config=_model_config(args),
            eval_scenarios=args.eval_scenarios,
            verbose=args.verbose,
        )
        print(json.dumps(result["metrics"], indent=2))
        return

    if args.command == "phase2":
        model = load_policy(args.model) if args.model is not None else None
        result = run_phase2_ablations(
            output_path=args.output,
            sim_config=_sim_config(args),
            model=model,
            max_neighbors=args.max_neighbors,
            variants=args.variants,
            circle_agents=args.circle_agents,
            bottleneck_agents=args.bottleneck_agents,
            max_iterations=args.max_iterations,
            damping=args.damping,
        )
        print(json.dumps(result["metrics"], indent=2))
        return

    if args.command == "phase3":
        result = run_phase3_experiment(
            output_dir=args.output_dir,
            map_path=args.map_path,
            sim_config=_sim_config(args),
            dataset_config=_phase3_train_dataset_config(args),
            model_config=_model_config(args),
            eval_scenarios=args.eval_scenarios,
            variants=args.variants,
            max_iterations=args.max_iterations,
            damping=args.damping,
            verbose=args.verbose,
        )
        print(json.dumps(result["metrics"], indent=2))
        return

    if args.command == "phase4":
        result = run_phase4_experiment(
            output_dir=args.output_dir,
            sim_config=_sim_config(args),
            dataset_config=_phase4_train_dataset_config(args),
            model_config=_model_config(args),
            eval_scenarios=args.eval_scenarios,
            variants=args.variants,
            max_iterations=args.max_iterations,
            damping=args.damping,
            verbose=args.verbose,
        )
        print(json.dumps(result["metrics"], indent=2))
        return

    if args.command == "phase4-eval":
        result = run_phase4_evaluation(
            model_path=args.model,
            output_path=args.output,
            sim_config=_sim_config(args),
            dataset_config=_phase4_eval_dataset_config(args),
            variants=args.variants,
            max_iterations=args.max_iterations,
            damping=args.damping,
        )
        print(json.dumps(result["metrics"], indent=2))
        return

    if args.command == "benchmark-obstacles":
        plan = build_benchmark_plan(
            map_scen_list=args.map_scen_list,
            output_dir=args.output_dir,
            agent_counts=args.agent_counts,
            seeds=args.seeds,
            train_scenarios=args.train_scenarios,
            eval_scenarios=args.eval_scenarios,
            horizon=args.horizon,
            max_steps=args.max_steps,
            max_neighbors=args.max_neighbors,
            min_start_goal_distance=args.min_start_goal_distance,
            max_samples=args.max_samples,
            max_obstacle_tokens=args.max_obstacle_tokens,
            obstacle_context_range=args.obstacle_context_range,
            observation_version=args.observation_version,
            expert_type=args.expert_type,
            limit=args.limit,
            smoke=args.smoke,
        )
        summary = run_benchmark_plan(
            plan,
            output_dir=args.output_dir,
            plan_only=args.plan_only,
        )
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()
