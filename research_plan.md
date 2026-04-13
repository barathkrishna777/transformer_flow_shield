Continuous-Space MAPF with Learned Planning + Collision Shield

Agentic Research Execution Plan

1. Main Research Idea

Title

Decentralized Continuous-Space MAPF via Learned Flow-Based Planning with PIBT-Inspired Collision Shielding

Core Hypothesis

A learned decentralized planner (e.g., transformer trained via flow matching) can capture long-horizon coordination and liveness, but requires a continuous collision shield to guarantee safety and resolve local conflicts—analogous to how PIBT operates in grid MAPF.

Architecture Overview

Planner (Learned)

Input: local observations (neighbor states, goals, map context)

Output: continuous motion intent (velocity / short trajectory / flow field)

Trained via: imitation learning / flow matching / trajectory supervision

Collision Shield (Analytical / Rule-based / Optimization-based)

Input: proposed motion from planner

Output: safe, minimally modified motion

Key mechanisms:

Priority assignment + inheritance

Local collision avoidance

Backtracking / yielding

Minimal deviation from intent

Key Contributions

Continuous analogue of PIBT-style collision resolution

Decentralized, scalable continuous MAPF solver

Separation of:

Global reasoning (ML)

Local safety (shield)

2. Supporting / Related Work (Agent Instructions)

Goal

Build a retrievable research knowledge base for repeated use.

Search Topics

The agent should search and collect papers/code under:

MAPF (Core)

PIBT (Priority Inheritance Backtracking)

ECBS / EECBS / CBS variants

Decentralized MAPF

Lifelong MAPF

Liveness guarantees

Continuous Multi-Agent Planning

ORCA (Optimal Reciprocal Collision Avoidance)

RVO / HRVO

Velocity obstacles

Control barrier functions (CBFs) for multi-agent systems

Decentralized MPC for collision avoidance

Learning-Based Planning

Transformer-based planning

Decentralized policies for multi-agent navigation

Graph neural networks for MAPF

Diffusion / flow matching / rectified flow models

Trajectory prediction models (multi-agent)

Hybrid Systems (Closest to Our Idea)

Learning + safety filter

RL + CBF

Shielded RL

Safe motion planning with learned priors

Data Collection Instructions

Folder Structure

research/
  ├── mapf/
  ├── continuous_planning/
  ├── learning_based/
  ├── hybrid_methods/
  ├── notes/
  └── summaries/


For Each Paper

Store the following structure:

paper_name/
  ├── paper.pdf
  ├── summary.md
  ├── key_points.json
  └── citations.bib


summary.md Format

Problem setup

Key idea

Strengths

Weaknesses

Relevance to our idea

Reusable components

key_points.json Format

{
  "method_type": "",
  "centralized": true/false,
  "continuous": true/false,
  "uses_learning": true/false,
  "handles_liveness": true/false,
  "collision_method": "",
  "notes": ""
}


Retrieval Optimization

Build an embedding index over summaries.

Tag papers with:

"collision"

"liveness"

"decentralized"

"continuous"

"learning"

Enable queries like:

"continuous decentralized collision avoidance"

"learning-based MAPF with guarantees"

3. Experimentation Plan

Benchmark

Use Moving AI MAPF benchmark.

Start with:

Empty maps

Then obstacle maps

Then dense scenarios

Phase 0: Infrastructure Setup

Simulator: Continuous 2D environment, circular agents with radius.

Dataset Generation: Generate expert trajectories:

Straight-line optimal motion (empty map)

ORCA / simple planner for obstacles

Phase 1: Minimal Setup (No Obstacles)

Goal: Test if learned planner + shield works in simplest setting.

Setup:

Map: empty

Agents: small number (5–20)

Model: Small transformer

Input: neighbor states + goal

Output: velocity vector

Experiments:

Learned planner only

Learned planner + collision shield

Metrics: Collision rate, Time to goal, Smoothness, Deadlocks

Implementation status:

- Phase 0 implemented in `flow_shield`: continuous 2D simulator, circular agents, empty-map straight-line expert, scenario generation, observation encoding, and dataset save/load.
- Phase 1 implemented in `flow_shield`: NumPy attention policy, supervised training, collision shield comparison, metrics, CLI pipeline, and tests.
- Run end-to-end with `python3 -m flow_shield.cli phase1 --output-dir runs/phase1`.
- Phase 2 implemented in `flow_shield`: explicit shield variants/factory, no-shield and velocity-clip baselines, pairwise projection, priority yielding, PIBT-inspired priority inheritance/backtracking prototype, adversarial scenario suite, richer safety/correction metrics, CLI ablation runner, and tests.
- Run shield ablations with `python3 -m flow_shield.cli phase2 --output results/phase2_ablations.json`.

Phase 2: Introduce Collision Shield

Shield Design Variants:

Simple: Velocity clipping, Pairwise avoidance

Intermediate: Priority-based yielding

Advanced: Priority inheritance graph, Local backtracking

Compare: No shield vs. shield variants

Implementation status:

- Added variants: `none`, `velocity_clip`, `pairwise`, `priority`, and `pibt`.
- Added adversarial scenarios: two-agent swap, multi-agent crossing, dense circle/ring swap, and corridor-like bottleneck. The bottleneck is represented by starts/goals because static obstacles are not active yet.
- Added metrics: collision rate, pair collisions per run, mean/max min-separation violation, mean/max shield correction norm, smoothness, time to goal, success rate, and deadlock rate.
- Prototype limitation: the PIBT-inspired shield is a one-step local heuristic with priority inheritance and greedy backtracking. It exposes unresolved final conflicts in diagnostics/metrics rather than claiming global PIBT liveness guarantees.

Phase 3: Obstacle Maps

Add: Static obstacles from Moving AI maps.

New challenges: Narrow passages, Congestion.

Evaluate: Liveness (do agents get stuck?), Throughput, Success rate.

Implementation status:

- Phase 3 implemented in `flow_shield`: Moving AI `type octile` `.map` parsing, Moving AI `.scen` row parsing for one-map task sources, continuous blocked-cell obstacle representation, simulator obstacle projection/reporting, obstacle collision/separation metrics, obstacle-free start/goal sampling with radius clearance and A* reachability checks, per-agent grid A* waypoint expert velocities, path-aware `obstacle_waypoint_v2` observation tokens, map-aware shield diagnostics/projection, expert waypoint baseline evaluation, and focused tests.
- Added Phase 3 CLI/data workflow:
  - `python3 -m flow_shield.cli phase3 --map tests/fixtures/tiny_obstacle.map --output-dir runs/phase3`
  - `python3 -m flow_shield.cli generate-data --scenario-type obstacle_map --map tests/fixtures/tiny_obstacle.map --max-obstacle-tokens 8 --output datasets/phase3_obstacle_train.npz`
- Expected Phase 3 artifacts: `phase3_obstacle_dataset.npz`, `phase3_policy.npz`, and `phase3_results.json`.
- Current liveness hardening: Phase 3 obstacle runs default to `obstacle_waypoint_v2`, which adds next/second A* waypoint directions, normalized remaining path length, path availability, and direct goal direction while preserving legacy loading for older 10-feature artifacts. The NumPy transformer output head can consume raw self-token features directly, so the supervised ridge head can imitate waypoint velocities instead of relying only on random encoded features. JSON includes `expert_waypoint_baseline` plus learned-vs-expert deltas for success, deadlock, time-to-goal, final-distance, no-progress, and safety metrics.
- Current quality diagnostics: rollout summaries now expose `termination_reason`, `failure_flags`, final distance to goal, fraction of agents within goal tolerance, and no-progress/recent-progress diagnostics. Aggregates include `failure_breakdown`, `no_progress_rate`, final-distance metrics, and within-tolerance fractions so low success can be attributed before larger sweeps.
- Limitations: the expert is per-agent A* plus continuous waypoint following, not a joint CBS/ECBS expert; `.scen` support parses standard rows and reports skipped invalid tasks but does not yet orchestrate full benchmark suites; Phase 3 evaluates one static map at a time and surfaces unresolved obstacle/agent interactions in JSON metrics and shield diagnostics.

Phase 4: Scaling Model + Data

Increase: Number of agents (20 → 100+), Map size, Training data.

Improve model: Larger transformer, Add attention over neighbors, Possibly graph structure.

Implementation status:

- Phase 4 implemented in `flow_shield`: scaled empty-map scenario/data generation supports larger agent counts, map sizes, neighbor windows, horizons, reproducible seeds, and optional sample caps for CPU-bounded runs. Phase 4 can also run obstacle-map scaling through the Phase 3 Moving AI map path with `scenario_type="obstacle_map"`, `map_path`, nonzero `max_obstacle_tokens`, optional `.scen` task sources, and the same expert waypoint baseline JSON.
- Added a configurable phase 4 experiment path with a transformer-style policy interface. The CPU-compatible default uses a fixed random multi-head attention encoder and trains a ridge-regression output head. A real trainable PyTorch transformer backend is now available with `policy_type="torch_transformer"` / CLI `--policy-type torch_transformer`; it trains all weights with AdamW when PyTorch is installed. PyTorch is still not a required dependency for local smoke tests, and runs detect/report PyTorch and CUDA availability in `backend_diagnostics`.
- Added scaled evaluation over Phase 2 shield variants (`none`, `velocity_clip`, `pairwise`, `priority`, and `pibt`) with the existing rollout/evaluation APIs.
- Added throughput/scaling metrics: `steps_per_second`, `agents_per_second`, and `mean_agents_per_run`, while preserving collision rate, pair collisions, min-separation violation, shield correction norms, smoothness, time to goal, success rate, and deadlock rate.
- Added Phase 4 CLI commands:
  - `python3 -m flow_shield.cli phase4 --output-dir runs/phase4`
  - `python3 -m flow_shield.cli phase4-eval --model runs/phase4/phase4_policy.npz --output results/phase4_eval.json`
- Expected Phase 4 artifacts: `phase4_scaled_dataset.npz`, `phase4_policy.npz`, and `phase4_results.json`.
- Prototype limitation: Phase 4 obstacle scaling now uses the Phase 3 static-map sampler or `.scen` task source plus the per-agent A* expert on one Moving AI map at a time. It still does not run a full benchmark-suite manager or train a full PyTorch/GPU transformer backend.

Phase 5: Learned + Shield Co-Design

Ideas:

Train model with shield in loop.

Penalize motions that require heavy correction.

Predict uncertainty / confidence.

Implementation status:

- Phase 5 foundations implemented in `flow_shield`: dataset generation can optionally store auxiliary shield-correction targets with `include_auxiliary_targets=True` or CLI `--include-auxiliary-targets`.
- Auxiliary target schema is optional and backward-compatible with old datasets. Current targets include correction vector, correction norm, correction-needed flag, unsafe-command flag, obstacle-intervention flag, and pairwise-intervention flag.
- Rollout/eval JSON now reports `correction_needed_rate`, `mean_correction_target_norm`, `max_correction_target_norm`, `obstacle_intervention_rate`, and `pairwise_intervention_rate`.
- Limitation: the current NumPy policies still train only the velocity output. This is not yet a full learned+shield co-design trainer or auxiliary-head model.

Phase 6: Advanced Shield Improvements

Priority inheritance graph (PIBT-like)

Multi-step conflict resolution

Temporary goal reassignment

Backtracking policies

Implementation status:

- Phase 6 foundation implemented as a lightweight obstacle-map coordination expert: `expert_type="prioritized_astar"` uses a deterministic reservation-table A* baseline with vertex/edge reservations and wait actions.
- The original `expert_type="independent_astar"` behavior remains the default. Expert type is stored in dataset, model, and evaluation JSON.
- CLI support: pass `--expert-type prioritized_astar` to Phase 3/4 obstacle-map dataset runs and benchmark plans.
- Guardrails: reservation-table A* has bounded time-expanded search, reports reservation calls/failures/expansion-limit hits, and reports fallback-to-independent and total prioritized failure counts in `astar_cache_info`.
- Limitation: `prioritized_astar` is a dependency-free baseline, not CBS/ECBS and not a complete joint solver. It can fall back to independent A* when a reservation-constrained path is not found. It remains experimental until Lambda-scale quality evidence improves.

Phase 7: Stress Testing

Dense crowd scenarios

Narrow corridors

Adversarial setups

Implementation status:

- Phase 7 foundation implemented as `python3 -m flow_shield.cli benchmark-obstacles`.
- The runner accepts a text or JSON list of Moving AI `.map`/`.scen` pairs, including map-only lines, plus agent counts, seeds, observation version, expert type, train/eval limits, `--smoke`, `--limit`, and `--plan-only`/`--dry-run`.
- Plan-only mode writes `benchmark_plan.json` and `benchmark_summary.json` without running training/eval, which is the recommended local validation path.
- Execution mode writes per-case Phase 3 result JSON under the output directory, an aggregate summary with failed/skipped case reporting, compact JSON/CSV tables grouped by map, agent count, seed, expert, and shield variant, and progress/status JSONL files.
- Usability additions: `--resume`/`--skip-completed` reuses case directories with existing `phase3_results.json`, and `benchmark-compare` compares two benchmark directories using dependency-free JSON output.
- Lambda guidance: generate/validate a plan locally with tiny fixture data, then run the same command on Lambda without `--plan-only` and with real Moving AI lists, larger train/eval limits, and mounted output storage.
- Limitation: this is orchestration around the current NumPy Phase 3/4 pipeline, not proof of large-scale performance. Full benchmark claims require Lambda-scale runs.

Recent Lambda observations:

- Cached independent 8-case obstacle run: 8/8 complete, about 116.6 seconds.
- Cached independent medium run: 24/24 complete, about 1561 seconds using `--train-scenarios 16 --eval-scenarios 8 --horizon 80 --max-steps 160 --max-samples 10000 --limit 24 --case-timeout-seconds 600`.
- Small cached `prioritized_astar` probe: 4/4 complete, about 147 seconds, still slower than independent on that probe and still experimental.
- Success rates remain low, mostly 0.0 with occasional 0.125 or 0.25 cases in the medium run. Current results should be treated as infrastructure/throughput validation plus early quality diagnostics, not final planner performance.

4. Additional Components

A. Code Structure

project/
  ├── models/
  ├── shield/
  ├── simulator/
  ├── training/
  ├── evaluation/
  └── configs/


B. Evaluation Metrics

Safety: Collision rate, Minimum distance violations

Efficiency: Time to goal, Path optimality

Liveness: % agents reaching goal, Deadlock frequency

Stability: Oscillation / jitter

C. Ablations

No shield vs. shield

Different shield strengths

Model size

Training data size

With vs. without obstacles

D. Visualization

Trajectory plots

Collision heatmaps

Priority propagation visualization

E. Key Risks

Shield too conservative → kills performance

Shield too weak → collisions

Learned policy ignores shield behavior

Deadlocks in dense environments

F. Stretch Goals

Real robot deployment (ROS2 + multi-robot)

Extend to SE(2) / dynamics

Integrate with MPC

Theoretical guarantees (safety / liveness bounds)

5. Final Instruction to Agent

You are responsible for:

Building a structured research knowledge base.

Implementing a minimal working pipeline quickly.

Iteratively scaling complexity.

Continuously comparing: Learned-only vs. Learned+Shield.

Logging everything for reproducibility.

Core Directive: Focus on fast iteration first, sophistication later.
