# Independent-layout input ablations: implemented, not a scientific result

The goal remains effective RGB-plus-deployment-valid-sensor Go2 navigation in
unfamiliar mazes, with online memory and defensible comparisons of JEPA training
and online rollout. This change prepares an information-utility experiment;
passing its tests does not establish a learned navigation policy or maze success.

The preceding status turn was a verified wait: live supervisor 25963 was polled
and parent PID 2063013 / child PID 2063119 were confirmed running. This turn makes
implementation progress while that same collection continues without restart.

## Implemented comparisons

`lewm/independent_pulse_input_ablation_development.py` defines four treatments:

| Treatment | Information removed | Information deliberately retained |
| --- | --- | --- |
| `full` | None | All original policy inputs and training targets |
| `no_rgb` | Past RGB and future RGB consumed by both online and teacher encoders | Body/control observations, physical labels and target availability |
| `latest_packet_only` | Differences between the four past observation packets; repeat the last packet four times | Its internal 20-sample body and 15-sample control histories, original future targets, proposed action |
| `no_candidate_command` | Proposed command values, replaced by zeros | Validity/duration and target times, past applied control history, original future targets |

These are separately retrained missing-information controls, not test-time-only
corruptions. They retain the same model width, tensor shapes, raw sample indices,
draw multiplicity, native physical labels, exact partial horizons and censoring
masks. `latest_packet_only` is not a fully memoryless sensor, and
`no_candidate_command` is not an action/time-free baseline. Duration can reveal
the short/long pulse choice even when command values are removed. No-RGB encoder
biases remain learnable constants, but no past or future image content reaches
the objective. Unknown future RGB may become zero-valued padding under no-RGB;
its availability mask stays false and it does not become latent supervision.

The transform does not mutate its source tensors; unchanged tensors are shared
read-only. It validates original available observation finiteness and exact
shapes before removal. Invalid available images or commands cannot be hidden by
zeroing, and invalid old body packets cannot be hidden by repeating the last one.

`lewm/independent_pulse_study_runner_development.py` applies the same treatment
after raw training materialization and after policy-only inference materialization.
Both paths now verify the original prescribed prospective command plan before
any removal. Every update record, fit result and prediction result names the
input variant. The default remains `full` for existing numerical callers.

The outer scientific experiment still must require explicit matching treatment
identities across its manifest, training records, snapshot binding and inference
call. This numerical helper alone is not an experiment gate and does not infer
how arbitrary supplied weights were trained. No resume, checkpoint selection,
calibration, navigation action or recorded-data fit is performed by this change.

## Verification

- Focused run 4274 completes exit 0: 135 tests pass in 14.37 s across ablations,
  streamed runner, snapshots and cumulative-event models.
- After adding positive controls, 9456 completes exit 0: all 48 ablation tests
  pass in 9.78 s. Nine objective/treatment combinations have identical losses
  and parameter gradients when only the removed information changes; without
  removal, the same perturbations change the loss. All twelve combinations of
  four treatments and three objectives complete synthetic training and inference.
  Tests also cover original-plan corruption, exact row pairing, unchanged masks,
  finite active gradients, absent teacher gradients and failure latching.
- Recorded read-only check 72436 completes exit 0. It authenticates the completed
  l00 launch/audit and bound source/raw products through the existing terminal
  loader. All four treatments have matching training/inference inputs and
  unchanged labels/clocks/masks on six open-passage actions and the first
  near-wall contact example. The latter has five positive contact horizons but
  zero valid future-image targets. No recorded model was fitted or scored.
- Full explicit 238-file regression 12349 completes exit 0: **3,198 passed in
  246.79 s**, including all 48 ablation tests and their positive controls.
- Source check a4cd0c verifies all 771 live supervisor bindings unchanged and
  confirms the changed/new ablation and runner paths are outside that closure.
  Post-edit check 1ba2e1 again verifies all 771 bindings unchanged and validates
  the checkpoint JSON and three current source hashes. Final e46423 confirms
  those three source hashes still match the tested versions.

Source SHA-256 values at the full-regression launch:

- ablation: `903bb9c88836a515971b473ee5829dc08272b4ab0be2ecc93e421a796055c619`
- runner: `d932a7b54a4cc1164837cb633c279c6a8d2e937e11b5fd2678faa2546c822ef0`
- ablation tests: `c760c5e81f0d477e361095336ec71ca902c742d8ef2ef39312238f9075c618f8`

## Live acquisition and the next scientific step

The original supervisor 25963 remains live with l01 child PID 2063119; its latest
observed progress reaches 64/120 raw-prechecked l01 cases (6f9085). Parent and
child are confirmed live at 1988/1937 s elapsed (e46423). l00 is completed and
audited, not twelve completed layouts. No competing collection is launched and
no frozen collection source is edited. Earlier partial measurement aggregates
are not extrapolated to later cases.

Next implement and freeze the actual matched experiment executable: require all
twelve authenticated terminal receipts, keep all planned-layout denominators,
report missing physical outcomes and joint contact/motion/future-image coverage,
and use fixed paired seeds, exposure schedules and update budgets for all arms
and input treatments. Persist exact update accounting and bounded snapshots;
score their verified reloads on identical rows against the train-only empirical
action/time and zero-motion baselines. Report all paired seeds and per-layout
effects, not a selected best run or a frame-level pseudo-sample size. Three
development-evaluation layouts are three topology units, not final confirmation.

The scientific question is whether scene/history/action information and latent
prediction help; a negative answer must remain an acceptable result. The prior
JEPA models still lose the empirical motion baseline, and room-return remains
0/3. Avoid replacing physical progress with repeated prediction-only tuning.
After this bounded comparison, use its measured failures to choose the local
execution/planning change, test rollout against matched no-rollout selection,
and exercise actual branch memory/backtracking and full beacon-search/return
missions. Reliable turning/slip response, real-time deadlines, realistic sensing
and bounded hardware evidence remain unfinished requirements of the full goal.
