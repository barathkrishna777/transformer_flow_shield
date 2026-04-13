"""Benchmark orchestration utilities for Moving AI obstacle-map experiments."""

from __future__ import annotations

import json
import signal
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
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


def _map_name(path: object) -> str:
    return Path(str(path)).stem if path else ""


def _compact_metric_row(
    case: Dict[str, object],
    variant: str,
    metrics: Dict[str, object],
    result_path: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    resumed: bool = False,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "case_id": case.get("case_id"),
        "map": _map_name(case.get("map_path")),
        "map_path": case.get("map_path"),
        "scen_path": case.get("scen_path"),
        "num_agents": case.get("num_agents"),
        "seed": case.get("seed"),
        "expert_type": case.get("expert_type"),
        "shield_variant": variant,
        "result_path": result_path,
        "elapsed_seconds": elapsed_seconds,
        "resumed": bool(resumed),
    }
    for key in (
        "success_rate",
        "deadlock_rate",
        "no_progress_rate",
        "collision_rate",
        "pair_collisions_per_run",
        "obstacle_collision_rate",
        "obstacle_collisions_per_run",
        "mean_time_to_goal",
        "mean_final_distance_to_goal",
        "max_final_distance_to_goal",
        "mean_fraction_agents_within_goal_tolerance",
        "correction_needed_rate",
        "obstacle_intervention_rate",
        "pairwise_intervention_rate",
        "steps_per_second",
        "agents_per_second",
    ):
        row[key] = metrics.get(key, 0.0)
    breakdown = metrics.get("failure_breakdown", {})
    if isinstance(breakdown, dict):
        for key, value in breakdown.items():
            row[f"failure_{key}"] = value
    return row


def compact_case_rows(
    case: Dict[str, object],
    metrics_by_variant: Dict[str, Dict[str, object]],
    result_path: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    resumed: bool = False,
) -> List[Dict[str, object]]:
    return [
        _compact_metric_row(
            case,
            variant,
            metrics,
            result_path=result_path,
            elapsed_seconds=elapsed_seconds,
            resumed=resumed,
        )
        for variant, metrics in sorted(metrics_by_variant.items())
    ]


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def grouped_compact_summary(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    numeric_keys = (
        "success_rate",
        "deadlock_rate",
        "no_progress_rate",
        "collision_rate",
        "obstacle_collision_rate",
        "mean_time_to_goal",
        "mean_final_distance_to_goal",
        "mean_fraction_agents_within_goal_tolerance",
        "correction_needed_rate",
        "obstacle_intervention_rate",
        "pairwise_intervention_rate",
    )
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row.get("map"), row.get("num_agents"), row.get("shield_variant"))
        groups.setdefault(key, []).append(row)
    grouped = []
    for (map_name, num_agents, variant), group_rows in sorted(groups.items()):
        item: Dict[str, object] = {
            "map": map_name,
            "num_agents": num_agents,
            "shield_variant": variant,
            "case_count": len(group_rows),
        }
        for key in numeric_keys:
            item[key] = _mean([float(row.get(key, 0.0)) for row in group_rows])
        grouped.append(item)
    return {"by_map_agent_count_shield": grouped}


def write_compact_outputs(rows: List[Dict[str, object]], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    compact_path = output_dir / "benchmark_compact_summary.json"
    csv_path = output_dir / "benchmark_compact_summary.csv"
    payload = {
        "rows": rows,
        "groups": grouped_compact_summary(rows),
    }
    compact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"compact_json": str(compact_path), "compact_csv": str(csv_path)}


def compare_benchmark_summaries(
    left_dir: str | Path,
    right_dir: str | Path,
    output_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Compare two compact benchmark summaries by map/agent/expert/shield keys."""

    def _load_rows(directory: str | Path) -> List[Dict[str, object]]:
        path = Path(directory) / "benchmark_compact_summary.json"
        if not path.exists():
            summary_path = Path(directory) / "benchmark_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing compact or aggregate summary in {directory}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows: List[Dict[str, object]] = []
            for result in summary.get("results", []):
                rows.extend(
                    compact_case_rows(
                        result,
                        result.get("metrics", {}),
                        result_path=result.get("result_path"),
                        elapsed_seconds=result.get("elapsed_seconds"),
                        resumed=bool(result.get("resumed", False)),
                    )
                )
            return rows
        return list(json.loads(path.read_text(encoding="utf-8")).get("rows", []))

    left_rows = _load_rows(left_dir)
    right_rows = _load_rows(right_dir)
    key_fields = ("map", "num_agents", "seed", "expert_type", "shield_variant")
    left_by_key = {tuple(row.get(field) for field in key_fields): row for row in left_rows}
    right_by_key = {tuple(row.get(field) for field in key_fields): row for row in right_rows}
    metric_keys = (
        "success_rate",
        "deadlock_rate",
        "no_progress_rate",
        "mean_time_to_goal",
        "mean_final_distance_to_goal",
        "obstacle_collision_rate",
        "pairwise_intervention_rate",
        "obstacle_intervention_rate",
    )
    comparisons = []
    for key in sorted(set(left_by_key) & set(right_by_key)):
        left = left_by_key[key]
        right = right_by_key[key]
        row = {field: value for field, value in zip(key_fields, key)}
        for metric in metric_keys:
            row[f"{metric}_left"] = left.get(metric, 0.0)
            row[f"{metric}_right"] = right.get(metric, 0.0)
            row[f"{metric}_delta_right_minus_left"] = float(right.get(metric, 0.0)) - float(
                left.get(metric, 0.0)
            )
        comparisons.append(row)
    result = {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "matched_rows": len(comparisons),
        "left_only_rows": len(set(left_by_key) - set(right_by_key)),
        "right_only_rows": len(set(right_by_key) - set(left_by_key)),
        "comparisons": comparisons,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output_path"] = str(output)
    return result


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


def _append_progress(path: Path, event: Dict[str, object], echo: bool) -> None:
    """Append one JSONL progress event and optionally echo it to stdout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wall_time": time(),
        **event,
    }
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if echo:
        print(line, flush=True)


def _write_status(path: Path, status: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


@contextmanager
def _case_timeout(seconds: Optional[float]):
    """Raise TimeoutError after seconds on Unix-like systems."""

    if seconds is None or float(seconds) <= 0.0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - signal timing path.
        del signum, frame
        raise TimeoutError(f"case exceeded timeout_seconds={float(seconds)}")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def run_benchmark_plan(
    plan: Dict[str, object],
    output_dir: str | Path,
    plan_only: bool = False,
    echo_progress: bool = True,
    case_timeout_seconds: Optional[float] = None,
    skip_completed: bool = False,
) -> Dict[str, object]:
    """Execute a benchmark plan case-by-case or just write the plan."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = write_benchmark_plan(plan, output_dir)
    progress_path = output_dir / "benchmark_progress.jsonl"
    status_path = output_dir / "benchmark_status.json"
    _append_progress(
        progress_path,
        {
            "event": "plan_written",
            "plan_path": str(plan_path),
            "case_count": int(plan.get("case_count", 0)),
            "plan_only": bool(plan_only),
        },
        echo=echo_progress,
    )
    if plan_only:
        summary = {
            "phase": "phase7_benchmark_orchestration",
            "plan_path": str(plan_path),
            "executed": False,
            "case_count": int(plan.get("case_count", 0)),
            "results": [],
            "failed": [],
            "skipped": [],
            "progress_path": str(progress_path),
            "status_path": str(status_path),
            "compact_rows": [],
        }
        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_status(
            status_path,
            {
                "state": "plan_only",
                "case_count": int(plan.get("case_count", 0)),
                "summary_path": str(summary_path),
            },
        )
        return summary

    results = []
    failed = []
    skipped = []
    compact_rows: List[Dict[str, object]] = []
    cases = list(plan.get("cases", []))
    total_start = perf_counter()
    for index, case in enumerate(cases):
        case_output = output_dir / str(case["case_id"])
        per_map_path = case_output / "phase3_results.json"
        case_start = perf_counter()
        case_status = {
            "state": "running",
            "case_index": index,
            "case_count": len(cases),
            "case_id": case["case_id"],
            "map_path": case.get("map_path"),
            "scen_path": case.get("scen_path"),
            "num_agents": case.get("num_agents"),
            "expert_type": case.get("expert_type"),
            "case_output": str(case_output),
            "completed_count": len(results),
            "failed_count": len(failed),
            "timeout_seconds": case_timeout_seconds,
            "skip_completed": bool(skip_completed),
        }
        _write_status(status_path, case_status)
        _append_progress(
            progress_path,
            {
                "event": "case_start",
                **case_status,
            },
            echo=echo_progress,
        )
        if skip_completed and per_map_path.exists():
            try:
                existing = json.loads(per_map_path.read_text(encoding="utf-8"))
                elapsed = float(perf_counter() - case_start)
                case_result = {
                    "case_id": case["case_id"],
                    "map_path": case.get("map_path"),
                    "scen_path": case.get("scen_path"),
                    "num_agents": case.get("num_agents"),
                    "seed": case.get("seed"),
                    "expert_type": case.get("expert_type"),
                    "result_path": str(per_map_path),
                    "success": True,
                    "resumed": True,
                    "elapsed_seconds": elapsed,
                    "metrics": existing.get("metrics", {}),
                    "case_summary": {
                        "map": _map_name(case.get("map_path")),
                        "num_agents": case.get("num_agents"),
                        "seed": case.get("seed"),
                        "expert_type": case.get("expert_type"),
                        "shield_variants": sorted(existing.get("metrics", {}).keys()),
                    },
                }
                results.append(case_result)
                compact_rows.extend(
                    compact_case_rows(
                        dict(case),
                        existing.get("metrics", {}),
                        result_path=str(per_map_path),
                        elapsed_seconds=elapsed,
                        resumed=True,
                    )
                )
                skip_record = {
                    "case_id": case.get("case_id"),
                    "map_path": case.get("map_path"),
                    "scen_path": case.get("scen_path"),
                    "reason": "completed_existing",
                    "result_path": str(per_map_path),
                }
                skipped.append(skip_record)
                _append_progress(
                    progress_path,
                    {
                        "event": "case_skipped_existing",
                        **skip_record,
                        "completed_count": len(results),
                        "failed_count": len(failed),
                        "skipped_count": len(skipped),
                    },
                    echo=echo_progress,
                )
                continue
            except Exception as exc:  # pragma: no cover - corrupt resume artifact path.
                _append_progress(
                    progress_path,
                    {
                        "event": "case_resume_read_failed",
                        "case_id": case.get("case_id"),
                        "result_path": str(per_map_path),
                        "reason": type(exc).__name__,
                        "message": str(exc),
                    },
                    echo=echo_progress,
                )
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
            with _case_timeout(case_timeout_seconds):
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
            case_result = {
                "case_id": case["case_id"],
                "map_path": case.get("map_path"),
                "scen_path": case.get("scen_path"),
                "num_agents": case.get("num_agents"),
                "seed": case.get("seed"),
                "expert_type": case.get("expert_type"),
                "result_path": str(per_map_path),
                "success": True,
                "resumed": False,
                "elapsed_seconds": float(perf_counter() - case_start),
                "metrics": result.get("metrics", {}),
                "case_summary": {
                    "map": _map_name(case.get("map_path")),
                    "num_agents": case.get("num_agents"),
                    "seed": case.get("seed"),
                    "expert_type": case.get("expert_type"),
                    "shield_variants": sorted(result.get("metrics", {}).keys()),
                },
            }
            results.append(case_result)
            compact_rows.extend(
                compact_case_rows(
                    dict(case),
                    result.get("metrics", {}),
                    result_path=str(per_map_path),
                    elapsed_seconds=case_result["elapsed_seconds"],
                    resumed=False,
                )
            )
            _append_progress(
                progress_path,
                {
                    "event": "case_complete",
                    "case_id": case["case_id"],
                    "elapsed_seconds": case_result["elapsed_seconds"],
                    "result_path": str(per_map_path),
                    "completed_count": len(results),
                    "failed_count": len(failed),
                },
                echo=echo_progress,
            )
        except Exception as exc:  # pragma: no cover - failure reporting path.
            failure = {
                "case_id": case.get("case_id", "unknown"),
                "map_path": case.get("map_path"),
                "scen_path": case.get("scen_path"),
                "reason": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": float(perf_counter() - case_start),
            }
            failed.append(failure)
            _append_progress(
                progress_path,
                {
                    "event": "case_failed",
                    **failure,
                    "completed_count": len(results),
                    "failed_count": len(failed),
                },
                echo=echo_progress,
            )
    summary = {
        "phase": "phase7_benchmark_orchestration",
        "plan_path": str(plan_path),
        "executed": True,
        "case_count": int(plan.get("case_count", 0)),
        "completed_count": len(results),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "elapsed_seconds": float(perf_counter() - total_start),
        "progress_path": str(progress_path),
        "status_path": str(status_path),
        "compact_rows": compact_rows,
        "grouped_summary": grouped_compact_summary(compact_rows),
        "results": results,
        "failed": failed,
        "skipped": skipped,
    }
    summary.update(write_compact_outputs(compact_rows, output_dir))
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_status(
        status_path,
        {
            "state": "complete",
            "case_count": len(cases),
            "completed_count": len(results),
            "failed_count": len(failed),
            "skipped_count": len(skipped),
            "elapsed_seconds": summary["elapsed_seconds"],
            "summary_path": str(summary_path),
        },
    )
    _append_progress(
        progress_path,
        {
            "event": "benchmark_complete",
            "summary_path": str(summary_path),
            "completed_count": len(results),
            "failed_count": len(failed),
            "skipped_count": len(skipped),
            "elapsed_seconds": summary["elapsed_seconds"],
        },
        echo=echo_progress,
    )
    return summary
