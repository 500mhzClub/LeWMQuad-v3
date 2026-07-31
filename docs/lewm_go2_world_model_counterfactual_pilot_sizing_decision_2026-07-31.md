# World-model counterfactual pilot: source-only sizing and decision contract

Date: 2026-07-31

Status: **source-only design; not an authority, preregistration, runtime result,
or permission to inspect generated inputs.** No generated payload, checkpoint,
RGB leaf, simulator, renderer, or GPU was opened for this sizing exercise.

## 1. Decision

Do **not** authorize a counterfactual render or training job from the current
source. The immediate blocker is not an unknown number of render-hours. The
repository has no path that both produces and consumes a matched,
physics-executed, scene-disjoint H6-contract counterfactual bundle, and it has no
WM-A.4 action-ranking/regret evaluator.

The next counterfactual operation should be a calibration-only run, but only
after the missing source is implemented, synthetically tested, committed, and
independently reviewed. Calibration measures throughput and numerical
repeatability; it is not a scientific pilot and does not authorize training.

This narrows the previous handoff: counterfactual **data semantics and tooling**,
not bulk generation itself, are the present world-model critical path. Formal
repository promotion still runs through G2-G8; this diagnostic path cannot
replace it.

## 2. Why the existing path is not physics-valid counterfactual evidence

The available pieces do not compose into the required experiment:

1. `build_jepa_counterfactual_render_plans.py` integrates candidate endpoints
   kinematically and writes `physics_validated: false`. Rendering the resulting
   camera pose produces an image, not a locomotion outcome.
2. `replay_jepa_physics_calibration.py` does step Genesis through the locomotion
   policy, but begins each candidate through `_set_pose`. That reset zeroes base
   velocity, installs standing joints with zero joint velocity, resets PPO
   action-latency history, and zeros the previous executed command. Both PPO
   inference and command clipping depend on the discarded histories. The result
   is a standardized reset-pose rollout, not another action from the matched H6
   state. The script also emits no RGB target.
3. Frozen H6 rows hold RGB identities and primitive IDs, not a restorable
   simulator/controller snapshot. The old counterfactual-row builder adds base
   pose but still omits base twist, joint state, PPO previous action, previous
   executed command, engine state, and deterministic replay prefix.
4. `build_jepa_spatial_future_dataset.py` can propagate a physics-validity flag,
   but it retains kinematic consequence labels. Merely flipping the flag would
   produce semantically inconsistent rows.
5. `dev_probe_counterfactual_action_fidelity.py` hard-codes the two legacy
   phase2b JSONLs and the old canonical RGB root. Its decoder rejects a new
   development pilot root, and its CLI has no expected input hash.
6. The fidelity and capacity probes measure latent energy only. They load a
   progress label but do not use it, so they do not implement the plan's
   privileged action ranking/regret test or a current-state task control.

Consequently, the historical 94-group result remains a protocol error record.
There are currently **zero eligible claim-bearing groups** known from source;
no old generated file was opened to reach that conclusion.

## 3. Required source contract before calibration

The preferred implementation is an online, synchronized branch collector. It
should co-generate new development-only H6-contract states instead of trying to
reconstruct hidden physical state from old H6 rows.

For each state:

1. Build parallel Genesis environments for one scene with identical spawn,
   physics seed, PPO state, previous executed command, and common two-block
   prefix.
2. Capture three context endpoints separated by the canonical five-command-tick
   block cadence. Record the two requested primitive IDs and their exact
   executed `(5,3)` command tapes.
3. Before branching, record base pose/twist, joint position/velocity, PPO
   previous-action state, previous executed command, simulator time/RNG, and a
   cross-environment equality receipt. Hidden engine state is avoided by
   co-running branches rather than approximated from a pose reset.
4. Execute all nine one-block primitive candidates, each for five command ticks
   through the same PPO and safety limiter. Record requested primitive, exact
   executed tape, fall/swept-safety/recoverability state, physical target
   progress, and endpoint physical state.
5. Render the actual endpoint with the batched camera, validate native RGB and
   transient depth, store immutable 224×224 RGB, and bind every leaf by bytes
   and SHA-256. Do not persist depth unless separately justified.
6. Include repeated same-action branches during calibration. Do not set
   `physics_validated` unless common-prefix state equality, executed-action
   identity, camera validity, and repeatability are all auditable.

The consumers must then change as a unit:

- replace hard-coded JSONL/RGB paths with one immutable pilot manifest plus
  expected byte count and SHA-256;
- confine every new input/output to the exact authority-bound development root;
- require mutually disjoint calibration, train, and evaluation scene lists;
- reject requested actions that collapse to duplicate executed command tapes;
- keep train capacity and withheld-scene generalization as different claims;
- add paired, scene-clustered comparisons against action-blind, action-shuffled,
  and current-state controls;
- define a train-only utility/readout contract that converts predicted future
  state to physical progress/safety, then report privileged-oracle action regret
  on evaluation scenes; and
- add a receipt-only checker that opens no undeclared RGB or checkpoint.

Until those entrypoints exist, an exact counterfactual execution authority is
impossible: there is no executable command to bind.

## 4. Source-derived unit costs

These are identities from ordinary source, not measured throughput:

| quantity | exact value |
|---|---:|
| primitive candidates | 9 |
| command ticks per action block | 5 |
| simulated seconds per action block | 0.5 s |
| policy steps per branch | 25 |
| physics steps per branch | 250 |
| native render | 640×480 RGB + transient depth |
| stored target | 224×224 RGB |
| raw stored bytes per RGB frame | 150,528 B |
| raw native float32 depth if mistakenly persisted | 1,228,800 B/frame |

The old two-block nine-by-nine factorial design would require 81 candidates and
two targets per state: 162 frames rather than nine. It is **18×** the one-step
target workload and answers a different question, so it is excluded.

## 5. Calibration tranche

Calibration is deliberately small and uses scenes that can never enter the
pilot roles:

| allocation | exact cap |
|---|---:|
| scene families | 8 |
| calibration-only scenes | 8 (one/family) |
| states per scene | 2 |
| distinct action branches per state | 9 |
| ordinary branches | 144 |
| repeat-control branches | 16 (one repeat/state) |
| total branch cap | **160** |
| branch simulation | **80 simulated seconds** |
| context frames | 48 |
| target frames including repeats | 160 |
| raw stored RGB ceiling | **31,309,824 B (29.86 MiB)** |

Use a deterministic repeat allocation fixed in the future source contract: one
HOLD repeat for the first state and one `forward_medium` repeat for the second
state of every family. Calibration performs no model training and creates no
WM-A result.

Its terminal receipt must report external end-to-end wall time and separate
scene-build, common-prefix, branch-step, render, resize/encode/write, and hashing
times; peak GPU memory; actual bytes; all-nine-action group yield; executed-tape
distinctness; pre-branch state equality; repeated-action state/RGB deltas;
camera-invalid/fall/incomplete counts; and every source/input/output binding.

The calibration decision is:

- **STOP_SOURCE_REDESIGN** if the implementation cannot expose a complete
  pre-branch equality receipt, if intended action IDs collapse after clipping,
  or if repeat noise prevents a tolerance/noise model from being frozen before
  pilot execution.
- **FREEZE_PILOT_CONTRACT** only after the calibration scenes are excluded from
  both pilot roles and exact repeatability tolerances, timing/byte ceilings, and
  validity rules are committed in a new pilot preregistration.

No refill, retry, second calibration, or automatic pilot launch follows either
decision.

## 6. Pilot ladder and hard cap

Spend the small budget on scene diversity before correlated states within one
scene. All counts below cover every one of the eight families, use all nine
actions per state, and keep train/evaluation scenes disjoint from one another
and from calibration.

| tranche | scenes/family | states/scene | states | branches/targets | branch simulation | raw target RGB |
|---|---:|---:|---:|---:|---:|---:|
| low | 2 train + 2 eval | 4 | 128 | 1,152 | 576 s | 165.38 MiB |
| recommended | 2 train + 2 eval | 8 | 256 | 2,304 | 1,152 s | 330.75 MiB |
| hard cap | 2 train + 2 eval | 12 | 384 | 3,456 | 1,728 s | 496.13 MiB |

For the recommended tranche:

- train: 16 scenes, 128 states/groups, 1,152 physical targets;
- evaluation: 16 different scenes, 128 states/groups, 1,152 physical targets;
- per role: 128 examples/action and 16 examples/family/action;
- total: 11,520 command ticks, 57,600 policy steps, and 576,000 physics
  steps;
- co-generated context: 768 additional frames and 110.25 MiB raw RGB; and
- accidental persisted float32 depth would add about 2.64 GiB.

Two evaluation scenes per family do not support family-specific generalization
claims. This is an exploratory overall, scene-clustered pilot. The hard cap is a
ceiling, not a default and not permission to refill rejected groups.

Choose among `STOP`, `LOW`, or `RECOMMENDED` by applying committed calibration
measurements to fixed resource ceilings. `HARD_CAP` is not an automatic choice;
it requires a separate justification and authority. A future pilot
preregistration must state the ceilings explicitly rather than treating the
table as permission.

## 7. Wall-clock and GPU sizing formulas

Do not infer throughput by dividing simulated seconds by parallel environments.
Scene construction, batched simulation, rendering, encoding, and hashing scale
differently.

```text
W_total =
  sum(scene_build_wall_s)
  + sum(state_common_prefix_wall_s)
  + sum(branch_step_wall_s)
  + sum(native_rgbd_render_wall_s)
  + sum(resize_encode_write_wall_s)
  + audit_and_hash_wall_s
```

The calibration must measure both external process elapsed time and the stage
terms above. Existing renderer `wall_seconds` values begin after scene
construction and are not end-to-end cost.

For the recommended 256-group diagnostic workload, source permits these exact
unit counts but not seconds:

- the current capacity implementation would cache `(3 context + 9 target)`
  normalized tensors per group: about **441 MiB** of input tensors alone,
  excluding model, optimizer, activations, and allocator overhead;
- 60 epochs imply 7,680 train-group updates;
- evaluation at every epoch plus the repeated terminal snapshot implies 47,616
  group evaluations across both roles and three modes; and
- the standalone four-arm fidelity probe implies 1,024 group scores.

Those defaults are not endorsed. The source must expose a bounded evaluation
cadence, seed, mask panel, and measured-memory preflight before pilot authority.

## 8. Pilot interpretation and stop rules

The pilot must separate four outcomes:

1. **train capacity absent** — the intervention cannot fit even train groups;
2. **capacity without generalization** — train fidelity rises, withheld-scene
   fidelity/regret does not;
3. **latent fidelity without task value** — withheld-scene latent matching
   improves, action regret/current-state controls do not; and
4. **scene-disjoint action utility** — paired scene-clustered latent and regret
   advantages survive the frozen controls.

Only outcome 4 can justify one-step WM-I integration. None authorizes a
composable head, multistep rollout, G2-G8 execution, navigation, promotion, or
deployment. Outcomes 1-3 stop scale-up and localize the next source/science
question.

## 9. Source identities reviewed

The key source files inspected for this sizing decision were:

| path | SHA-256 |
|---|---|
| `scripts/build_jepa_counterfactual_render_plans.py` | `74f104651344d235561a85aaade489948663fc20737e7fdbfba55c2e5ec4d9c2` |
| `scripts/render_jepa_counterfactual_plan_root.py` | `42ed0103a768a1363603367ea4a3b5d717f0f182120573cfc0dfe10f87e05351` |
| `scripts/replay_jepa_physics_calibration.py` | `6a21545d21214c641d40d7db189c284b37d0d502f63d381772eac836b7b4e7ec` |
| `scripts/benchmark_lewm_closed_loop_mpc.py` | `8dfd314335825ba04c5feefdb7bb416b378ecef7d1e4b56ea31b66d3d109b776` |
| `scripts/build_jepa_spatial_future_dataset.py` | `aef4dd0e0d238ec5b994ba3ef8730e661d2074f7d275c546239eb1826e209660` |
| `scripts/build_go2_corpus_counterfactual_rows.py` | `f1708b962f162a4ec10da52432490fca9ad1b9ea541b536891700ffdcf788f96` |
| `scripts/dev_probe_counterfactual_action_fidelity.py` | `4def692e3d3582b55e5c5cc08ab4a2a2d835fba9281a2d9698ad970bd25a1c00` |
| `scripts/dev_probe_counterfactual_overfit_capacity.py` | `ae82a8936c0b580c0d04745da9c54e771c6f0e8020ffe712ef67fa13cd3ac72c` |
| `lewm_genesis/lewm_genesis/rollout.py` | `a26b9640a4fed85f5297d61aa656175e9d37dc4b48c9ac81f5370b37ace1d8fc` |
| `lewm_genesis/lewm_genesis/scene_builder.py` | `1eec27ee97de1e853890e34f6c008ff748ce6034f9467d20fea8d212c78f3158` |

These bindings document the source-only audit. They authorize no runtime read or
execution.
