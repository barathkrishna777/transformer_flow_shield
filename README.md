# Continuous MAPF Flow Shield

This workspace implements phase 0 through phase 4 from `research_plan.md`, plus foundation work for phases 5 through 7:

- Phase 0: a continuous 2D circular-agent simulator and empty-map expert data generation.
- Phase 1: a small NumPy attention policy trained by supervised imitation, plus learned-only vs. learned-plus-shield evaluation on empty maps.
- Phase 2: shield ablations across no shield, velocity clipping, pairwise projection, priority yielding, and a PIBT-inspired priority inheritance/backtracking prototype on adversarial scenarios.
- Phase 3: static obstacle maps via Moving AI `type octile` `.map` parsing, optional `.scen` task import, continuous blocked-cell collision checks, obstacle-free scenario sampling, A* waypoint expert data, path-aware obstacle observations, map-aware shields, expert-baseline JSON, CLI artifacts, and tests.
- Phase 4: scaled empty-map dataset generation plus optional Phase 3 obstacle-map scaling, larger configurable agent/map/neighbor/horizon settings, a NumPy transformer-style policy interface, scaled shield ablations, throughput metrics, and JSON/model/dataset artifacts.
- Phase 5 foundations: optional shield-correction supervision targets in datasets and rollout/eval metrics for correction/intervention rates. This is not yet a full shield-in-the-loop co-design trainer.
- Phase 6 foundation: a dependency-free `prioritized_astar` reservation-table waypoint expert for obstacle maps, alongside the default `independent_astar`, with expansion/fallback diagnostics. This remains experimental and is not CBS/ECBS.
- Phase 7 foundation: a `benchmark-obstacles` planner/runner for multiple Moving AI `.map`/`.scen` entries with plan-only/dry-run mode, smoke limits, resume/skip-completed support, per-case JSON, compact JSON/CSV summaries, and failure reporting for Lambda execution.

The implementation intentionally depends only on NumPy. Phase 4 detects whether PyTorch/CUDA are available and records that diagnostic in results, but the current scalable path uses a NumPy transformer-style fixed attention encoder with a closed-form ridge-regression output head. New transformer runs also append raw self-token features to that output head so path-aware Phase 3 features are not lost behind the random encoder. This is a prototype scaling path, not a full PyTorch transformer trainer.

```bash
python3 -m pip install -r requirements.txt
```

## Run

Generate phase 0 expert data:

```bash
python3 -m flow_shield.cli generate-data --output datasets/phase1_empty_train.npz
```

Train the phase 1 policy:

```bash
python3 -m flow_shield.cli train --data datasets/phase1_empty_train.npz --model-out artifacts/phase1_attention_policy.npz
```

Evaluate learned-only vs. shielded planning:

```bash
python3 -m flow_shield.cli eval --model artifacts/phase1_attention_policy.npz --output results/phase1_metrics.json
```

Or run the whole phase 0/1 pipeline:

```bash
python3 -m flow_shield.cli phase1 --output-dir runs/phase1
```

Run phase 2 shield ablations with the default straight-line intent policy:

```bash
python3 -m flow_shield.cli phase2 --output results/phase2_ablations.json
```

Run the same ablations with a trained phase 1 model:

```bash
python3 -m flow_shield.cli phase2 --model artifacts/phase1_attention_policy.npz --output results/phase2_learned_ablations.json
```

The phase 2 scenarios are `two_agent_swap`, `multi_agent_crossing`, `dense_circle_swap`, and `corridor_bottleneck`. The bottleneck remains encoded with starts/goals so Phase 2 stays a pure agent-agent shield baseline; Phase 3 adds real static obstacle maps.

Run the phase 3 obstacle-map pipeline on a Moving AI `.map` file:

```bash
python3 -m flow_shield.cli phase3 \
  --map tests/fixtures/tiny_obstacle.map \
  --output-dir runs/phase3 \
  --num-agents 8 \
  --train-scenarios 24 \
  --eval-scenarios 8 \
  --max-obstacle-tokens 8 \
  --observation-version obstacle_waypoint_v2
```

Use the Phase 6 prioritized expert baseline when you want lightweight reservation-table coordination in the generated expert targets and expert JSON baseline:

```bash
python3 -m flow_shield.cli phase3 \
  --map tests/fixtures/tiny_obstacle.map \
  --output-dir runs/phase3_prioritized_smoke \
  --num-agents 3 \
  --train-scenarios 2 \
  --eval-scenarios 1 \
  --horizon 5 \
  --max-steps 8 \
  --expert-type prioritized_astar
```

Add Phase 5 auxiliary correction targets to a dataset when you want shield-correction supervision artifacts for later co-design work:

```bash
python3 -m flow_shield.cli generate-data \
  --scenario-type obstacle_map \
  --map tests/fixtures/tiny_obstacle.map \
  --num-scenarios 2 \
  --num-agents 3 \
  --horizon 5 \
  --include-auxiliary-targets \
  --output datasets/phase5_aux_smoke.npz
```

For data generation only:

```bash
python3 -m flow_shield.cli generate-data \
  --scenario-type obstacle_map \
  --map tests/fixtures/tiny_obstacle.map \
  --max-obstacle-tokens 8 \
  --output datasets/phase3_obstacle_train.npz
```

Phase 3 writes:

- `phase3_obstacle_dataset.npz`
- `phase3_policy.npz`
- `phase3_results.json`

Phase 3 obstacle runs default to `obstacle_waypoint_v2`, which keeps the legacy 10-feature token fields and adds fixed-shape self-token path context: next A* waypoint direction, second waypoint direction when available, normalized remaining path length, path availability, and direct goal direction. Legacy artifacts without observation metadata still load as `legacy`.

Moving AI map support expects:

- `type octile`
- `height <rows>`
- `width <cols>`
- `map`
- exactly `height` grid rows of length `width`

Passable tiles are `.`, `G`, and `S`; blocked tiles are `@`, `O`, `T`, and `W`. Each grid cell becomes a continuous unit square, so a `7 x 7` map has world bounds `(7.0, 7.0)` unless `--map-cell-size` is changed. Starts and goals are sampled only in free cells with agent-radius plus safety-margin clearance, a minimum start-goal distance, no initial agent-agent overlaps, and an A* reachability check.

Moving AI `.scen` files can be used as the task source:

```bash
python3 -m flow_shield.cli phase3 \
  --map tests/fixtures/tiny_obstacle.map \
  --scenario-source scen \
  --scen tests/fixtures/tiny_obstacle.scen \
  --scen-limit 32 \
  --output-dir runs/phase3_scen \
  --num-agents 1 \
  --train-scenarios 8 \
  --eval-scenarios 4
```

Standard scenario rows are parsed as `bucket map width height startX startY goalX goalY optimalLength`; starts/goals become continuous cell centers. Tasks that collide for the current radius/clearance, mismatch map dimensions, are too short, or lack an A* path are skipped and reported in `scenario_source_diagnostics.skip_counts`. Scenario grouping is seeded and greedy instead of fixed sequential chunks; grouping failures are also surfaced as diagnostics.

Create a Phase 7 benchmark plan without running training/eval:

```bash
printf '%s %s\n' tests/fixtures/tiny_obstacle.map tests/fixtures/tiny_obstacle.scen > /tmp/tiny_maps.txt
python3 -m flow_shield.cli benchmark-obstacles \
  --map-scen-list /tmp/tiny_maps.txt \
  --output-dir runs/benchmark_plan_smoke \
  --agent-counts 1,2 \
  --seeds 7007 \
  --expert-type prioritized_astar \
  --smoke \
  --limit 1 \
  --plan-only
```

Map lists may contain either one map path per line or `map scen` pairs. On Lambda, use the same command without `--plan-only` and with larger limits. Keep full benchmark runs off laptops: point `--map-scen-list` at a text or JSON list of Moving AI pairs, write to a mounted output directory, and collect `benchmark_plan.json`, `benchmark_summary.json`, `benchmark_compact_summary.json`, `benchmark_compact_summary.csv`, and each case's `phase3_results.json`.

Resume a partially completed benchmark directory:

```bash
python3 -m flow_shield.cli benchmark-obstacles \
  --map-scen-list /lambda/maps.txt \
  --output-dir /lambda/out/obstacle_medium \
  --agent-counts 4,8 \
  --seeds 7007,7008,7009 \
  --train-scenarios 16 \
  --eval-scenarios 8 \
  --horizon 80 \
  --max-steps 160 \
  --max-samples 10000 \
  --case-timeout-seconds 600 \
  --resume
```

Compare two benchmark output directories:

```bash
python3 -m flow_shield.cli benchmark-compare \
  --left /lambda/out/independent \
  --right /lambda/out/prioritized \
  --output /lambda/out/compare_independent_vs_prioritized.json
```

Run the phase 4 scaled data/model/evaluation pipeline:

```bash
python3 -m flow_shield.cli phase4 --output-dir runs/phase4
```

Useful small CPU smoke run:

```bash
python3 -m flow_shield.cli phase4 \
  --output-dir runs/phase4_smoke \
  --num-agents 20 \
  --world-size 20 \
  --train-scenarios 4 \
  --eval-scenarios 2 \
  --horizon 20 \
  --max-steps 40 \
  --max-neighbors 8 \
  --max-samples 2000 \
  --d-model 32 \
  --num-heads 4 \
  --num-layers 1
```

Evaluate an existing phase 1 or phase 4 model on scaled phase 4 scenarios:

```bash
python3 -m flow_shield.cli phase4-eval --model runs/phase4/phase4_policy.npz --output results/phase4_eval.json
```

The default phase 4 run writes:

- `phase4_scaled_dataset.npz`
- `phase4_policy.npz`
- `phase4_results.json`

For 100+ agents, increase the map size and keep CPU runs bounded with fewer scenarios or a sample cap, for example `--num-agents 100 --world-size 60 --max-neighbors 24 --max-samples 200000`. For obstacle-map scaling, use a sufficiently large Moving AI map and keep `--max-obstacle-tokens` nonzero.

Run phase 4 on a static obstacle map by enabling the Phase 3 map path:

```bash
python3 -m flow_shield.cli phase4 \
  --scenario-type obstacle_map \
  --map tests/fixtures/tiny_obstacle.map \
  --output-dir runs/phase4_obstacle_smoke \
  --num-agents 8 \
  --train-scenarios 4 \
  --eval-scenarios 2 \
  --horizon 20 \
  --max-steps 40 \
  --max-neighbors 8 \
  --max-obstacle-tokens 8 \
  --max-samples 2000
```

## Metrics

The evaluators report:

- `collision_rate`
- `pair_collisions_per_run`
- `mean_min_separation_violation`
- `max_min_separation_violation`
- `obstacle_collision_rate`
- `obstacle_collisions_per_run`
- `mean_obstacle_separation_violation`
- `max_obstacle_separation_violation`
- `mean_shield_correction_norm`
- `max_shield_correction_norm`
- `correction_needed_rate`
- `mean_correction_target_norm`
- `max_correction_target_norm`
- `obstacle_intervention_rate`
- `pairwise_intervention_rate`
- `success_rate`
- `mean_time_to_goal`
- `mean_smoothness`
- `deadlock_rate`
- `no_progress_rate`
- `mean_final_distance_to_goal`
- `max_final_distance_to_goal`
- `mean_fraction_agents_within_goal_tolerance`
- `failure_breakdown`
- `steps_per_second`
- `agents_per_second`
- `mean_agents_per_run`

Obstacle-map Phase 3 and Phase 4 result JSON also includes `expert_waypoint_baseline` and `learned_vs_expert` for the same shield variants. Compare `success_rate`, `deadlock_rate`, `no_progress_rate`, `mean_final_distance_to_goal`, `mean_time_to_goal`, and safety metrics there when diagnosing learned-policy liveness. Per-scenario summaries now also include `termination_reason`, `failure_flags`, final goal distances, and recent-progress diagnostics so low success can be attributed before larger sweeps.

## Lambda observations

Recent Lambda runs show that the cache work made medium independent obstacle sweeps practical, but the results remain quality diagnostics rather than final planner evidence:

- Cached independent 8-case obstacle run: 8/8 completed in about 116.6 seconds.
- Cached independent medium run: 24/24 completed in about 1561 seconds with no failures/skips using `--train-scenarios 16 --eval-scenarios 8 --horizon 80 --max-steps 160 --max-samples 10000 --limit 24 --case-timeout-seconds 600`.
- Small cached `prioritized_astar` probe: 4/4 completed in about 147 seconds, but it was slower than independent on that probe and remains experimental.
- Success rates in the medium run were mostly 0.0, with occasional 0.125 or 0.25 cases. Treat these outputs as infrastructure/throughput validation plus early quality diagnostics; use compact summaries and final-distance/failure breakdowns before scaling benchmarks further.

Prototype notes:

- `velocity_clip` is explicit and measurable, but the simulator also clips commands to `max_speed` during stepping.
- `pibt` is a one-step continuous priority inheritance/backtracking prototype. It is diagnostic, not a liveness proof; unresolved conflicts remain visible in `final_conflicts`, `collision_rate`, and separation-violation metrics.
- Phase 4's NumPy transformer policy trains only the output head over fixed random attention plus raw self-token features. Results include `backend_diagnostics` and notes documenting this fallback.
- Phase 3 obstacle experts are per-agent A* waypoint followers. They provide robust supervised velocities and an evaluation baseline for static maps, but they are not joint optimal MAPF solvers.
- `prioritized_astar` is a simple reservation-table A* baseline that plans agents in priority order with vertex/edge reservations. It has expansion limits and reports reservation failures/fallback counts in `astar_cache_info`, but it is incomplete, slower in current probes, and can fall back to independent A* if no reserved path is found.
- Phase 5 is foundation-only: auxiliary targets and metrics are written, but the current NumPy policies still train only velocity outputs.
- Phase 7 orchestration exists, but full benchmark evidence still needs Lambda-scale runs. Local usage should stay to `--plan-only`, `--smoke`, or tiny fixture cases.

## Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/flow_shield_pycache python3 -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/flow_shield_pycache python3 -m compileall flow_shield tests
```

The `PYTHONPYCACHEPREFIX` keeps bytecode writes inside a local writable cache on sandboxed macOS setups.
