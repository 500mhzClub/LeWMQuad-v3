# Prospective matched independent-layout prediction comparison V1

This is a bounded development experiment, not navigation, sensor deployment
qualification, checkpoint promotion or a final benchmark. The prior one-room
JEPA models still lose the empirical action/time motion baseline; room-return
remains 0/3. A negative JEPA result is scientifically valid. Do not erase those
results or require a positive result before improving the actual navigation
system.

## Entry and data gates

The executable is `scripts/run_go2_independent_pulse_matched_study_v1.py`.
It requires two explicit identities: the reviewed canonical source/config
definition SHA-256 and the terminal `result.json` SHA-256 from the original
fixed remaining-stage supervisor. Its supervisor launch must remain
`e47240aeacd1b3af711d2c9d8ba1698b62ad1020982bcd099b47f8ef8c873a26`.
The source definition inherits that supervisor's 771 bound source paths and adds
only the exact recursively discovered experiment/helper/test/protocol imports.
It makes no clean source copy/export and grants no predecessor or sealed access.

Preflight requires the exact interpreter and repository directory, CPU/thread
environment, deterministic PyTorch operation, unchanged source/native/input
bindings, the original successful twelve-batch supervisor result, no supervisor
failure, all twelve exact terminal receipts and successful complete-study loading.
A live partial collection is not a study. No output or fitting is started when
these preconditions fail. No alternate cohort, batch, checkpoint or role is
discovered. Complete-study loading authenticates recorded raw artifacts and
terminal audits but is not itself another physics or raw-audit replay.

The study output is the exclusive owned RecoveryStorage child
`go2_independent_pulse_matched_study_v1_attempt_001`. Do not retry, resume, overwrite,
skip a failed arm or replace this attempt. Its maximum artifact allowance is
8 GiB, with a 40 GiB remaining-space reserve. Preflight reserves room for the
whole allowance; every later write enforces the remaining budget and reserve.

After entry, persist the source/config definition, dataset receipt report and
coverage report before fitting. Require at least one eligible example in every
one of the six action-duration cells in every planned train, selection and
development-evaluation layout. All planned six-action groups must have complete,
exactly matching original native/body/control/RGB prefix witnesses. Missing
windows, censored labels and excluded groups remain explicit; they do not shrink
planned denominators. Missing/unmatched original prefixes or a missing
layout/action cell stop the attempt after its coverage report, with no fit.

Report contact, motion and future-image availability separately for every
episode/layout/role, including positive-contact horizons with and without future
images. At the common 2-second horizon, report all planned six-action groups,
missing actions, censored contacts and positive-versus-negative contrasts.
No positive contact contrast means hazard discrimination is untested, not
perfect. This does not suppress the separate motion-prediction experiment.

## Fixed factorial and budget

Run all 36 fresh fits in seed, input-treatment, objective order:

- Paired model/schedule seeds: 2026091101, 2026091102, 2026091103.
- Treatments: `full`, `no_rgb`, `latest_packet_only`, `no_candidate_command`.
- Objectives: `direct`, `supervised_rollout`, `jepa`.
- Each fit: 1,200 optimizer updates, batch size 6, latent width 32, AdamW learning
  rate 0.001 and zero weight decay, gradient norm limit 1, EMA momentum 0.99.
- Same cumulative integrated-softplus-hazard semantics, actual partial target
  times, 6 cm planar loss units and existing fixed regularization coefficients.
- Inference batch size 6 on CPU; final snapshot only, with no intermediate
  checkpoint scoring or stopping based on results.

Each seed publishes one train-only layout/action-balanced schedule. Reuse its
exact indices, order and draw multiplicity for every objective and treatment.
7,200 draws per fit provide ten complete population passes if all 720 train
windows are eligible and the fixed groups each contain twenty windows. Otherwise
report the actual coverage and draw multiplicities; do not call them ten epochs.
The total planned budget is 43,200 updates and 259,200 sample draws across fits.
This prospective budget expands the earlier short one-room pilot; it is not a
guarantee of optimization convergence and is not selected from evaluation loss.

All twelve fits within a seed must start from the identical full model tensor
hash. Objectives share model width and data exposure, but not FLOPs or identical
active-parameter sets: the direct objective does not train the recursive branch.
The supervised-rollout versus JEPA comparison isolates the added latent
prediction objective; a JEPA/direct difference alone does not isolate that
objective from the recursive architecture and auxiliary loss.

The input treatments have the exact meanings in
`lewm/independent_pulse_input_ablation_development.py`. No-RGB removes past and
future encoder/teacher RGB; latest-packet-only retains the last packet's internal
sensor/control histories; no-candidate-command retains time/duration and past
applied controls. These are retrained information-removal controls. They do not
establish sensor realism, eliminate all memory, or remove all action information.
Original prospective plans are verified before removal at both training and
inference. Labels, masks, row identities and horizons are unchanged.

## Persistence and common scoring

Write an exclusive request before each fit, binding experiment, dataset and
schedule hashes, treatment, seeded initial model identity and planned budget.
Persist every optimizer update as one flushed/fsynced JSON-lines record containing
its exact sample indices, treatment, schedule, losses, gradient norm, model hash
and elapsed wall time. After training, require exact contiguous update accounting,
ledger byte count and digest. A ledger-write failure after an optimizer step
retains that actual step count and partial ledger; it does not permit a retry.

Save one final bounded snapshot per fit and use its verified evaluation-only
reload for every prediction. Explicitly require the same treatment and weight
identity in the fit, snapshot and prediction outputs. Save exact prediction row
indices and raw head arrays as bounded NPZ, then reload with `allow_pickle=False`
and check exact array identity. Preserve primary and auxiliary heads without
selecting the better one: direct outcomes are primary for `direct`; recursive
outcomes are primary for `supervised_rollout` and `jepa`.

For each seed, fit the empirical action/time baseline on the exact repeated
train draw indices. Include a zero-motion/zero-yaw baseline that retains the
same empirical contact prediction; it is named `zero_motion_empirical_contact`,
not a claim that zero motion means zero collision. Missing empirical cells remain
unavailable without an evaluation-label fallback or interpolation.

Score train, selection and development-evaluation roles separately on identical
ordered rows. Train scoring is resubstitution; selection is reported but is not
used to choose weights or tune this attempt. Store per-seed layout-first motion,
yaw and contact scores, exact-time/context/history/support strata, and paired
method differences. Store the separate tie-aware common-horizon matched-contact
ranking analysis. Hazard-only ranking does not demonstrate progress to a goal.
An incomplete prediction head receives no favorable-subset comparison. Report
all seeds and all arms; no best-seed result, frame-level confidence interval,
probability-calibration claim or final-generalization claim is allowed. Three
development-evaluation layouts are only three independent topology units.

Recheck source and receipt identities before and after each fit. At the end,
reload all twelve batches, reauthenticate original data, require the same dataset
and coverage reports, verify all output hashes and require all 36 fits complete.
Only then write the terminal development-comparison result. Any infrastructure,
data, source, persistence or numerical failure retains evidence and terminates
the sequence without retries or continuing to other arms. Partial comparisons
are not relabeled a completed factorial.

## Relationship to the full goal

This study does not execute a navigation policy, use an online planner, exercise
long-horizon memory, validate realistic sensor errors/latency, meet real-time
deadlines or operate hardware. It tests local prediction and information utility
on development layouts only. After its bounded result, use the observed errors
to choose the next local-execution/planning change; do not indefinitely substitute
prediction tuning for physical progress. Reliable turning/slip response, matched
online-rollout and memory-on/off interventions, actual branch backtracking,
whole beacon-search/return missions, untouched final-layout confirmation and
bounded real-platform evidence remain required.
