# Go2 scene-diversity recurrent replication: qualified CPU-flat V3 science

Date frozen: 2026-08-05

Status before scientific execution: `PREREGISTERED_SCIENCE_NOT_RUN`

Scope: the single fresh development-only 64-scene scientific invocation
released by the exact CPU-flat V3 qualification PASS. This is the experiment
preregistered before the backend qualifications: increased scene diversity
with the frozen recurrent-DINO visual arm, its matched exact-zero-vision arm,
and the live task/action control. No model, seed, update, gate, threshold,
branch mechanism, or data allocation changes.

## Qualification release and evidence boundary

The only admissible qualification release artifact is:

- `.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_qualification/attempt_v1/qualification_decision.json`;
- SHA-256 `6eac9ff6458092d6284011934c865b45519cbd2ff14c0b2da3a515bbf4a6a299`;
- 1,789 bytes; and
- status `PASS_GENESIS_CPU_FLAT_DEVELOPMENT_V3_QUALIFICATION`.

The independent terminal review is:

- `docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_qualification_terminal_review_2026-08-05.json`;
- SHA-256 `bc13e880fc348515384b906ea7dce32bc7587df089f7aa7f6b630d71e87ce31d`;
- 13,591 bytes; and
- status `PASS_INDEPENDENT_QUALIFICATION_TERMINAL_REVIEW`.

The minimal decision records that scene gates, exact lane equality, VRAM and
release, kernel, and timing gates passed; backend is CPU; the branch mechanism
is `parallel_lockstep_envs_no_restore`; probes ran in order `[12, 0]`; the
scientific root was absent; and scientific-plan release is permitted. The
independent review deep-rehashed the qualification closure and confirms that
only the decision plus that review may gate a separately preregistered and
independently reviewed scientific plan.

Qualification evidence establishes only that the fixed CPU mechanism can
execute exact matched branches, render, satisfy integrity gates, and project
within the wall cap. It makes no representation, recurrent-model, planning,
navigation, or generalization claim.

The scientific builder and runner may open only the minimal decision and the
independent review from the qualification root. They must not open, copy,
parse, join, or reuse the qualification result, terminal, reservation, scene
results, state receipts, render receipts, RGB, meshes, cache, timings,
diagnostics, or any other qualification runtime payload. Their bindings may be
checked as metadata already carried by the decision/review, without opening
the bound targets.

## Exact scientific plan release

The frozen qualification plan is:

- `docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_qualification_exact_plan_2026-08-05.json`;
- SHA-256 `6a055839ab9bb6fe45b9cb5864e8f3c87e75f468dd7e9c26e8c950e4a6fedb78`;
- 355,206 bytes.

The scientific plan differs only in:

1. `attempt_id`, changed to the already preregistered deferred V3 science ID;
2. `output_root`, changed to its fresh science collection root; and
3. `successor_contract`, changed from qualification metadata to the exact
   decision/review release binding and frozen scientific protocol.

Normalizing those three fields must reproduce the exact qualification plan.
All scene, role, state, history, action, horizon, texture, asset, runtime,
render, input-binding, and count fields remain identical. The plan therefore
contains exactly 64 scenes, 256 states, 2,304 matched branches, 768 context
frames, 2,304 targets, nine candidate actions, 32 scenes and 128 states per
role, and four states per scene.

Science uses the fresh attempt root:

`.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1`

It must be absent and not a symlink before the one allowed invocation. No
qualification root or predecessor scientific runtime artifact is reusable.

## Fixed collection runtime

The interpreter remains:

`/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_render_vulkan/bin/python`

Each child receives exactly the qualified eleven-key environment:

```text
EGL_DEVICE_ID=1
GS_BACKEND=cpu
GS_PARA_LEVEL=0
HOME=/home/andrewknowles
MESA_VK_DEVICE_SELECT=1002:7551!
PATH=/usr/bin:/bin
PYOPENGL_PLATFORM=egl
PYTHONDONTWRITEBYTECODE=1
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
```

Physics is exact `gs.cpu` with serialized `GS_PARA_LEVEL=0`; policy is CPU.
Rendering and preflight remain on the bound Vulkan/EGL AMD Radeon AI PRO R9700
route. There is one fresh process group per scene, beginning independently at
scene 0. Every scene executes four states, both histories, nine matched
successors per state, the full horizon, exact nine-lane common-history
equality, physics invariants, and all render/depth/count/nonfinite/mesh/receipt
checks under the unchanged 36-lane no-restore mechanism.

The 300-second per-scene watchdog remains a hard safety ceiling. Selected-
device VRAM is continuously sampled against `16,977,405,952` bytes. Every
worker must exit without a leaked process group and pass the release barrier
against its own prelaunch R9700 baseline before its result is opened or the
next scene starts. The bracketed journal gate must find no new AMDGPU timeout,
reset, wedge, or HSA/KFD exception. Total collection, join, feature, training,
checkpoint, and evaluation wall time remains capped at 7,200 seconds.

There is no adaptive batching, snapshot/restore substitution, state-local
fallback, numerical tolerance relaxation, scene refill, worker retry, or
partial reuse.

## Frozen three-arm protocol

The benchmark remains byte-for-scientific-behavior inherited from
`go2_scene_diversity_recurrent_replication_v1` and
`go2_task_coupled_recurrent_dynamics_v1`:

- learned arms, in fixed order: `no_vision_recurrent_direct`, then
  `visual_recurrent_direct`;
- live analytic control: `task_action_only`, fit again on this train role;
- frozen DINOv2 ViT-S/14 context patches only;
- three context frames pooled from `16 x 16` to `4 x 4` cells;
- train-only channel PCA from 384 to 16 and train-only standardization;
- measured body-frame odometry and executed historical command tapes;
- candidate command tape queried after recurrent context formation;
- exact-zero visual input for the matched no-vision arm;
- matched model seeds `2026080411`, `2026080412`, `2026080413`;
- shared sampler seed `2026080414`;
- 800 updates, batch size eight;
- AdamW, learning rate `3e-4`, weight decay `1e-4`, betas `[0.9, 0.999]`,
  epsilon `1e-8`, gradient clipping at `1.0`;
- standardized four-output residual MSE plus `0.25` strict-pair rank loss;
- task ridge lambda `1e-3`;
- durable checkpoint before evaluation access; and
- two exact evaluations with identical results.

The recurrent model never receives the goal. There is no successor RGB,
successor feature, semantic or depth target, reward, planner, rollout, encoder
update, JEPA target, EMA, or exposed-role tuning route. The goal is available
only to the scorer and live control as already frozen.

## Access order and custody

After the fresh 64-scene collection and combined result are validated, the
model runner may initially open only collection metadata and the train
receipt/context closure. It must:

1. rehash the 128 train state receipts and 384 train context PNGs;
2. fit all train-only projection/statistical state and six learned members;
3. durably write and reopen the exact checkpoint;
4. only then rehash the 128 evaluation receipts and 384 evaluation context
   PNGs;
5. open zero train or evaluation successor PNGs; and
6. evaluate the durable checkpoint twice and require exact equality.

Train and evaluation scene/state identities must be disjoint. Access, source,
input, process, render, role, family, count, checkpoint, and result receipts
must fail closed on any mismatch.

No sealed, `sealed_*`, held-out, production, or other protected material may
be opened, searched, indexed, copied, or used. This is development evidence
only and cannot be promoted to final benchmark or deployment evidence.

## Fixed reports and decision gates

The scorer, uniform-random expectation, privileged physical oracle, and
10,000-resample paired family/scene bootstrap remain unchanged. The five gates
are fixed:

1. integrity, role disjointness, context-only custody, repeatability, and the
   privileged physical oracle pass;
2. ensemble visual normalized rank regret is at most `0.13`;
3. visual minus live task/action regret is at most `-0.02`, with paired 95%
   upper bound below zero;
4. visual minus matched no-vision regret is at most `-0.01`, with paired 95%
   upper bound below zero; and
5. visual beats uniform-random expectation.

No threshold is changed because the role is harder, the result is close, or a
point estimate improves. The result reports arm/member/ensemble regrets,
oracle-equivalent selection, target progress, physical MSE, per-family values,
paired comparisons, training traces, collection cost, access counts, and exact
source/checkpoint/result bindings.

A complete gate pass supports only the preregistered incremental visual-
information hypothesis on this development role. It does not by itself prove
that the model is useful for planning. Any gate failure stops this frozen
recurrent-DINO recipe; it does not authorize additional scenes, seeds,
updates, threshold changes, architecture changes, or another implementation
tweak.

## Independent source review and one-shot authority

Before the root is reserved, an independent scientific source review must
bind the exact preregistration, scientific plan builder, plan, runner, focused
tests, minimal PASS decision, and qualification terminal review. Its fixed
schema/status are:

- schema `lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_source_review_v1`;
- status `PASS_INDEPENDENT_SCIENTIFIC_SOURCE_REVIEW`.

It must explicitly confirm exact scientific payload identity, the eleven-key
environment, 64 fresh per-scene processes, unchanged three-arm protocol,
access order, checkpoint/evaluation boundary, five gates, minimal decision and
review as the only qualification reads, zero qualification payload reuse,
fresh root, and no protected access. It may clear exactly one scientific
invocation under the user's standing instruction; it creates no retry,
resume, refill, overwrite, repair, promotion, deployment, or scientific-claim
authority by itself.

The plan builder emits metadata only and marks scientific execution false
until that review passes. Any collection, infrastructure, integrity, training,
checkpoint, evaluation, or result failure consumes the one science attempt.
