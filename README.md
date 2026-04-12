# Continuous MAPF Flow Shield

This workspace implements phase 0 through phase 4 from `research_plan.md`:

- Phase 0: a continuous 2D circular-agent simulator and empty-map expert data generation.
- Phase 1: a small NumPy attention policy trained by supervised imitation, plus learned-only vs. learned-plus-shield evaluation on empty maps.
- Phase 2: shield ablations across no shield, velocity clipping, pairwise projection, priority yielding, and a PIBT-inspired priority inheritance/backtracking prototype on adversarial scenarios.
- Phase 3: static obstacle maps via Moving AI `type octile` `.map` parsing, continuous blocked-cell collision checks, obstacle-free scenario sampling, A* waypoint expert data, obstacle-context observation tokens, map-aware shields, CLI artifacts, and tests.
- Phase 4: scaled empty-map dataset generation plus optional Phase 3 obstacle-map scaling, larger configurable agent/map/neighbor/horizon settings, a NumPy transformer-style policy interface, scaled shield ablations, throughput metrics, and JSON/model/dataset artifacts.

The implementation intentionally depends only on NumPy. Phase 4 detects whether PyTorch/CUDA are available and records that diagnostic in results, but the current scalable path uses a NumPy transformer-style fixed attention encoder with a closed-form ridge-regression output head. This is a prototype scaling path, not a full PyTorch transformer trainer.

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
  --max-obstacle-tokens 8
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

Moving AI map support expects:

- `type octile`
- `height <rows>`
- `width <cols>`
- `map`
- exactly `height` grid rows of length `width`

Passable tiles are `.`, `G`, and `S`; blocked tiles are `@`, `O`, `T`, and `W`. Each grid cell becomes a continuous unit square, so a `7 x 7` map has world bounds `(7.0, 7.0)` unless `--map-cell-size` is changed. Starts and goals are sampled only in free cells with agent-radius plus safety-margin clearance, a minimum start-goal distance, no initial agent-agent overlaps, and an A* reachability check.

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
- `success_rate`
- `mean_time_to_goal`
- `mean_smoothness`
- `deadlock_rate`
- `steps_per_second`
- `agents_per_second`
- `mean_agents_per_run`

Prototype notes:

- `velocity_clip` is explicit and measurable, but the simulator also clips commands to `max_speed` during stepping.
- `pibt` is a one-step continuous priority inheritance/backtracking prototype. It is diagnostic, not a liveness proof; unresolved conflicts remain visible in `final_conflicts`, `collision_rate`, and separation-violation metrics.
- Phase 4's NumPy transformer policy trains only the output head over fixed random attention features. Results include `backend_diagnostics` and notes documenting this fallback.
- Phase 3 obstacle experts are per-agent A* waypoint followers. They provide robust supervised velocities for static maps, but they are not joint optimal MAPF solvers and do not import Moving AI `.scen` benchmark files yet.
- Phase 4 obstacle scaling now works through the Phase 3 static map path for one Moving AI map at a time. It is still the NumPy prototype path, not a full GPU transformer trainer or full benchmark-suite runner.

## Tests

```bash
python3 -m unittest discover -s tests
python3 -m compileall flow_shield tests
```
