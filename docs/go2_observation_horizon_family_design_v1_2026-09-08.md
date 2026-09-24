# Observation-horizon matched world-model design V1

The augmented five-command probe entered a previously observed nominal
clearance boundary despite a clear predicted chord. Replanning its unchanged
500-ms forecasts every 100 ms did not solve direct's clearance failure and
left JEPA holding during view acquisition for 226 of 240 selections. These
completed failures motivate a new, explicitly different temporal contract.

Derive eight targets at 100, 200, ..., 800 ms from the already collected,
raw-audited original family and moving-prefix/suffix data. Use exactly the
same 912-slot population and 408 training/420 transfer available contexts;
preserve all 84 old unavailable slots, their reasons and existing role and
cluster assignments. No new native data are needed for this derivation.
Label only the known prefix of the original actual command tape. Inactive
plan slots remain unknown, and contact is cumulative with motion and future
images censored at contact or missing acquisition. Native poses are target-side
labels only. The overlapping 500-ms target must reproduce the prior motion
and contact label exactly wherever in plan.

Model inputs remain four past RGB/body/control packets. Replace the action
interface by eight one-command blocks, each 100 ms, with explicit validity.
Unknown padding is zero and cannot become a braking command. Use the existing
causal observation encoder, latent width 32, action-conditioned latent
transition, direct head, rollout decoder, EMA target encoder and cumulative
contact semantics. New action-token widths and actual offset validation are
explicit new source; old snapshots are incompatible. Do not interpolate old
500-ms predictions into short-horizon labels or forecasts.

Prepare eighteen fresh matched models: the same fixed seeds 2026091001,
2026091401 and 2026091402 crossed with full/no-predictor-RGB inputs and
direct/supervised-rollout/JEPA objectives. Keep AdamW 0.001, EMA 0.99, gradient
clip 1, 1,200 updates and batch size six, with the exact previously prescribed
mixed-context schedules. Keep the existing 0.06-m position loss scale and
loss coefficients, now evaluated at the eight actual short horizons. Pair
initialization across the six arms of each seed; do not claim equality to
the old architecture's initialization. No warm start, checkpoint choice,
outcome-conditioned sampling or hyperparameter search.

First authenticate and derive the complete new target population, then check
causal tensor access and short-plan clocks/masks. Freeze training, evaluation,
admission and focused tests before a new short benchmark. Assess current
hardware, compare serial/four-worker throughput and require exact paired
benchmark updates and model identities before choosing science concurrency.
Benchmark weights are never scientific weights. Persist all 21,600 scientific
updates and every final snapshot, and save both roles' forecasts before
scoring. Full eighteen-model admission precedes any native use. Report both
100-ms and 500-ms predictive accuracy, all denominators, optimization-seed
variation and the existing source/cluster/role strata.

The fixed first-seed full-JEPA native candidate and same-seed full-direct
comparator remain assigned before these future results. Native execution
will require a separate frozen source/protocol using the actual 100-ms
first horizon, with unchanged physical, sensing and arrival gates. No native
run, timing qualification, independent-maze success or deployment follows
from this source/data/model work. The active broader goal remains open.

The separate three-member ensemble selector source and its two passing
synthetic tests are unlaunched development code. They are not part of this
temporal intervention, have admitted no runtime outputs, and grant no new
model, execution or qualification claim.
