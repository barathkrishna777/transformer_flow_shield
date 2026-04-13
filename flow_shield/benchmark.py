"""Benchmark orchestration utilities for Moving AI obstacle-map experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DatasetConfig, ModelConfig, SimConfig
from .experiment import run_phase3_experiment


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    map_path: str
    scen_path: Optional[str]
    num_agents: int
    seed: int
    train_scenarios: int
    eval_scenarios: int
    horizon: int
    max_steps: int
    max_neighbors: int
    min_start_goal_distance: float
    max_samples: Optional[int]
    max_obstacle_tokens: int
    obstacle_context_range: float
    observation_version: str
    expert_type: str


def _read_map_scen_list(path: str | Path) -> List[Dict[str, Optional[str]]]:
    """Read JSON or line-based map/scen benchmark entries."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        payload = json.loads(text)
        entries = payload.get("cases", payload) if isinstance(payload, dict) else payload
        return [
            {
                "map_path": str(entry["map_path"] if "map_path" in entry else entry["map"]),
                "scen_path": (
                    str(entry["scen_path"] if "scen_path" in entry else entry["scen"])
                    if ("scen_path" in entry or "scen" in entry) and entry.get("scen_path", entry.get("scen")) is not None
                    else None
                ),
            }
            for entry in entries
        ]

    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace(",", " ").split()]
        entries.append(
            {
                "map_path": parts[0],
                "scen_path": parts[1] if len(parts) > 1 else None,
            }
        )
    return entries


def build_benchmark_plan(
    map_scen_list: str | Path,
    output_dir: str | Path,
    agent_counts: Iterable[int],
    seeds: Iterable[int],
    train_scenarios: int,
    eval_scenarios: int,
    horizon: int,
    max_steps: int,
    max_neighbors: int,
    min_start_goal_distance: float,
    max_samples: Optional[int],
    max_obstacle_tokens: int,
    obstacle_context_range: float,
    observation_version: str,
    expert_type: str,
    limit: Optional[int] = None,
    smoke: bool = False,
) -> Dict[str, object]:
    """Build a deterministic JSON-serializable run plan."""

    entries = _read_map_scen_list(map_scen_list)
    cases: List[BenchmarkCase] = []
    output_dir = Path(output_dir)
    for entry_index, entry in enumerate(entries):
        for num_agents in agent_counts:
            for seed in seeds:
                case = BenchmarkCase(
                    case_id=f"case_{len(cases):04d}",
                    map_path=str(entry["map_path"]),
                    scen_path=entry.get("scen_path"),
                    num_agents=int(num_agents),
                    seed=int(seed),
                    train_scenarios=1 if smoke else int(train_scenarios),
                    eval_scenarios=1 if smoke else int(eval_scenarios),
                    horizon=min(int(horizon), 4) if smoke else int(horizon),
                    max_steps=min(int(max_steps), 20) if smoke else int(max_steps),
                    max_neighbors=int(max_neighbors),
                    min_start_goal_distance=float(min_start_goal_distance),
                    max_samples=20 if smoke else max_samples,
                    max_obstacle_tokens=int(max_obstacle_tokens),
                    obstacle_context_range=float(obstacle_context_range),
                    observation_version=str(observation_version),
                    expert_type=str(expert_type),
                )
                cases.append(case)
                if limit is not None and len(cases) >= int(limit):
                    break
            if limit is not None and len(cases) >= int(limit):
                break
        if limit is not None and len(cases) >= int(limit):
            break
    return {
        "phase": "phase7_benchmark_orchestration",
        "plan_only_supported": True,
        "source_list": str(map_scen_list),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "notes": [
            "Plan generation is CPU-light and intended for local validation.",
            "Full training/evaluation should run on Lambda or another larger machine.",
        ],
    }


def write_benchmark_plan(plan: Dict[str, object], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "benchmark_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def run_benchmark_plan(
    plan: Dict[str, object],
    output_dir: str | Path,
    plan_only: bool = False,
) -> Dict[str, object]:
    """Execute a benchmark plan case-by-case or just write the plan."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = write_benchmark_plan(plan, output_dir)
    if plan_only:
        summary = {
            "phase": "phase7_benchmark_orchestration",
            "plan_path": str(plan_path),
            "executed": False,
            "case_count": int(plan.get("case_count", 0)),
            "results": [],
            "failed": [],
            "skipped": [],
        }
        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    results = []
    failed = []
    skipped = []
    for case in plan.get("cases", []):
        case_output = output_dir / str(case["case_id"])
        try:
            sim_config = SimConfig(
                max_steps=int(case["max_steps"]),
            )
            dataset_config = DatasetConfig(
                num_scenarios=int(case["train_scenarios"]),
                num_agents=int(case["num_agents"]),
                horizon=int(case["horizon"]),
                max_neighbors=int(case["max_neighbors"]),
                min_start_goal_distance=float(case["min_start_goal_distance"]),
                scenario_type="obstacle_map",
                scenario_source="scen" if case.get("scen_path") else "sampled",
                map_path=str(case["map_path"]),
                scen_path=case.get("scen_path"),
                max_obstacle_tokens=int(case["max_obstacle_tokens"]),
                obstacle_context_range=float(case["obstacle_context_range"]),
                observation_version=str(case["observation_version"]),
                expert_type=str(case["expert_type"]),
                seed=int(case["seed"]),
                max_samples=case.get("max_samples"),
            )
            model_config = ModelConfig(
                d_model=16,
                policy_type="numpy_transformer",
                num_heads=2,
                num_layers=1,
                batch_size=64,
                seed=int(case["seed"]) + 17,
            )
            result = run_phase3_experiment(
                output_dir=case_output,
                map_path=case["map_path"],
                sim_config=sim_config,
                dataset_config=dataset_config,
                model_config=model_config,
                eval_scenarios=int(case["eval_scenarios"]),
                variants=("none", "pairwise"),
                max_iterations=3,
            )
            per_map_path = case_output / "phase3_results.json"
            results.append(
                {
                    "case_id": case["case_id"],
                    "result_path": str(per_map_path),
                    "success": True,
                    "metrics": result.get("metrics", {}),
                }
            )
        except Exception as exc:  # pragma: no cover - failure reporting path.
            failed.append(
                {
                    "case_id": case.get("case_id", "unknown"),
                    "map_path": case.get("map_path"),
                    "scen_path": case.get("scen_path"),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    summary = {
        "phase": "phase7_benchmark_orchestration",
        "plan_path": str(plan_path),
        "executed": True,
        "case_count": int(plan.get("case_count", 0)),
        "completed_count": len(results),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "results": results,
        "failed": failed,
        "skipped": skipped,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
