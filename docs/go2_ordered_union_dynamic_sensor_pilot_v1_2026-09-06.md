# Frozen eight-run ordered-union moving sensor pilot

Scientific scope: test mounted native sensor repeatability during actual Go2
motion, including the junction context of the retained depth silhouette failure
and a previously uncollected near-wall, lower-friction context. This is not model
training, independent-layout evaluation, navigation qualification or a retry of
the stopped 120-episode collection. All previous results remain unchanged.

## Exact population and implementation

Use the frozen inventory without changing any trial specification, role, seed,
spawn, wall collision box, friction, gait, sensor timing or action schedule.
Run these four conditions once in the listed order, then repeat the same order
in four newly built scenes (eight builds/episodes total):

1. `l00_junction_recent_forward_nominal_a0`
2. `l00_junction_recent_forward_nominal_a3`
3. `l00_near_wall_recent_forward_lower_friction_a0`
4. `l00_near_wall_recent_forward_lower_friction_a1`

Each output directory is `repeat_0_` or `repeat_1_` plus its exact inventory trial
ID. The stored specification and result retain the original trial ID. Repeats
are diagnostics and do not supply independent layout samples. All cases remain
train-role; no sample from this pilot is granted training eligibility.

The distinct constructor uses the tested union visual surfaces and installs
fixed floor-first ordering in this scene instance, while leaving physical boxes,
Go2 and gait unchanged. Native RGB/depth capture is inherited from the frozen
5mm-near mounted capture. After each capture, check actual JIT draw order and
native vertex identities. Record raster subpixel bits, depth-target depth bits,
RGB multisample-target sample count and positions, restoring the depth framebuffer
and releasing the GL context even when a precision query fails. These readbacks
are implementation properties, not a proven metre-domain error bound.

Commands are the exact fixed inventory pulse/history schedule, not functions of
tracker success or privileged native pose. The existing shadow tracker is logged
but does not gate command selection. Preserve 750 native settling samples,
external contact/speed/domain stops, float64 requested commands, actual applied
commands, sensor clocks and event censoring. Maximum per episode: 2,400 native
samples, 34 RGB-D frames and 33 command ticks. No physics drain after a stop.

## Prospective scoring and failure behavior

For each completed artifact commit, reconstruct raw physics, contacts, body/gyro,
RGB/depth packets, setup, commands and stop semantics with the unchanged raw
auditor. Persist the exact precheck. Compute the original strict sampled 1mm
visibility score on every recorded frame. Retain all failed frames and report
them even if they lie at a silhouette. Separately compute the fixed evaluator-only
pixel-footprint coverage diagnostic, including stable interior, ambiguous boundary
and near-occlusion results. It grants neither boundary-pixel certification nor
training eligibility and is not an inference input or a pixel repair.

Strict visibility failures are diagnostic outcomes in this distinct assay; they
do not redefine the failed predecessor gate, select cases or authorize training.
Continue the fixed eight cases after physical stops and strict visibility results.
Any acquisition/integrity/implementation/storage failure is terminal: preserve
partial artifacts, last attempted run, prior commits and prechecks; no retry,
resume, discarded failure or replacement case. A physical stop inside a case
never authorizes extra commanded motion in that case.

After collection is terminal, run its exclusive terminal auditor once. Include
all eight prescribed runs, incomplete/partial/unattempted outcomes, exact artifact
coverage and saved precheck equality. Reconstruct four across-action prefix
comparisons (two contexts times two repeats) and four same-condition full-stream
comparisons. Prefixes include all native/contact history and all nine real sensor
packets before action divergence. Full streams include every native/contact array,
all actual packets and all native-depth arrays, with dtype, shape and length.
Exclude only non-sensor wall-clock/resource logging from full-stream comparison.

Report exact-complete, exact-partial, unequal and unavailable comparisons
separately. An exact partial stream is not complete repeatability. Missing
prefixes are not matches. Retain motion/contact/future-image target counts, but
do not export a training dataset or fit a model. All four complete-repeat checks
and all four available full-prefix checks are required to claim the prescribed
dynamic repeatability population succeeded. Strict depth failures remain separate
and prevent calling this a fully visibility-qualified sensor dataset.

## Custody, storage and next decision

Exclusive external navigation output:
`go2_ordered_union_dynamic_sensor_pilot_v1_attempt_001`.
Freeze the collector, constructor, session, helpers, terminal auditor, tests and
this protocol with the inherited source closure before execution. Bind the exact
successful ordered static bench launch/result/artifacts and native source,
including the separately checked native scene sorting implementation. No sealed
access, whole-tree export, GPU training or robot hardware control is included.

CPU/software EGL only. Total collection budget 1GiB, per-episode allowance
128MiB, free-space reserve 40GiB. Count serialized launch metadata before creating
output; reserve at most 1/16 of the total budget for it. Before each episode
require space for its full allowance. Commit only explicit known artifacts;
missing expected artifacts fail closed. The terminal audit writes metadata only
and may not alter original commits, observations or outcomes.

Next decision must use terminal native evidence. Successful static or dynamic
repeatability alone does not establish deployment-valid sensors, calibrated
collision risk, RGB/history/action utility, JEPA advantage, reliable execution,
memory/backtracking benefit or unseen-maze completion. Those remain required
subsequent independent experiments under the full scientific objective.
