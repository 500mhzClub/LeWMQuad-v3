# Complete transition-prediction inputs validated; fitting remains next

The new transition-prediction bootstrap input stage completed with every336
training window and348 geometry-transfer window materialized. The actual
model-input fingerprints match the frozen causal derivation. All training
target contracts and both prediction heads accept the real timed plans.
No optimizer, checkpoint loading/saving, physics or native commands occurred.

This is a new explicit transition-prediction scope. The old four-second
navigation-design gate and its learning-readiness flag both remain false.
Their failure documents and artifacts are unchanged. The old policy-stream
checker requiring that flag was not called, and its output root remains absent.
No task discrimination, calibration, learned benefit or navigation claim is
inferred from this input check. The overall navigation goal remains active.

| Role | Actual windows | Active prediction slots | Motion/future training targets | Known contact training targets |
| --- | --- | --- | --- | --- |
| train | 336 | 1608 | 1352 | 1608, including256 positive |
| geometry_transfer | 348 | 1642 | none read | none read |

Training contains48 episodes, four layouts and two parameter clusters; transfer
contains48 episodes, four layouts and two other clusters. Mirrored siblings and
overlapping windows remain dependent. All768 planned contexts and84 missing
contexts remain represented by the original view. Missing post-contact motion
and images remain unavailable. Future images are read only as training targets;
transfer inference reads only actual past packets. Zero physical labels or
future observations enter either role's predictor inputs.

The fixed schedule uses seed2026091001,1200 planned updates and batch6. Each
of the48 training episodes appears exactly150 times. Sampling balances layout,
action, episode and available offsets without outcome-conditioned weighting.
The schedule file is a prospective artifact; actual optimizer steps are0.
The untrained model's state and absent gradients were verified unchanged after
every population was processed.

Eighteen focused scope/view/stream tests passed before launch. The real-data
stage took46.792s after launch and reached2,091,528,192 bytes peak process RSS.
Before launch, hardware had82.20GB available RAM,96.55GB artifact storage,
0.3% CPU activity, idle GPUs and no substantial competing Python task. It ran
in one CPU process with one Torch/OpenCV/BLAS thread. Resource snapshots were
recorded throughout materialization. All source/input hashes were reverified
after completion. The process exited0 and no input-check job remains running.

Root under the established navigation artifact base:
`go2_family_transition_bootstrap_inputs_v1_attempt_001`,849 bound sources.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | c30ae29e8f87d6fafe35ee6143c89a7ad51e1f3aa275eb218e711727d874d70f |
| result.json | 92b5fa6066715bad218b9d2cecdd853a10ff2448e93ab886d5af1214de09a01e |
| training_schedule.json | 6d242e257b2151e00b042b3a005988ca0efb279e0abf1f33c0e2eb06fe1236e9 |
| resource_monitor.jsonl | ca38488b5373881d9a0a80f5d8fc6a06b4cee252c594f4b376b97e8f697e2c3a |

The schedule's internal canonical-content identity, distinct from file bytes,
is `ac9dc60044c1ffda42e1551ff44ed98f3f4f6978409a5b0160cd1a63ba87859c`.
The unchanged untrained model-state identity is
`43d39e276646908ca8b632b1661a3a57a7c8447aeafb72e992253795a4a37895`.

The immutable collection/causal roots are
`go2_geometry_progress_family_v1_attempt_001` and
`go2_geometry_progress_family_causal_v1_attempt_001`, with result hashes
`376b2eeacfd5e741ba6b7a0e1b5e04399f782d16943d66ffdff92eada93ef0fb` and
`37bd88d43fd0ebdcee80282d38695f82a41b83313d08158d076f5ad4ce7145a7`.
The input launch binds their complete artifact rosters. New scope admission is
implemented in `lewm/family_transition_bootstrap_scope_development.py` and the
completed check in `scripts/check_go2_family_transition_bootstrap_inputs_v1.py`.

## Ordered continuation toward actual goal-reaching

1. Preserve all frozen inputs, sources, failures and completed receipts. The
   corner observer passes complete270-frame replay; use
   `docs/go2_corner_support_observer_replay_result_2026-09-08.md` and its exact
   receipt `97517b61fc33d2cdc484d96f0f47aeef0e730defb866444b08cc0ac41cb97375`.
   The spatial-SIFT quota candidate remains failed and ineligible.
2. Implement and freeze the six-fit transition bootstrap already specified in
   `docs/go2_family_transition_bootstrap_inputs_v1_2026-09-08.md`: one fixed
   initialization seed2026091001, full/no_rgb inputs crossed with direct,
   supervised_rollout and jepa;1200 updates, batch6, latent32, AdamW0.001,
   EMA0.99 and the unchanged loss. Use the validated common schedule above.
   Preserve every run, with no transfer-based checkpoint or seed selection.
3. Assess current hardware and benchmark representative actual-batch fitting
   serially versus useful process concurrency before the six scientific fits.
   Freeze benchmark seeds, update counts, source/input identities, equality
   checks and the concurrency decision rule first. Benchmark models must remain
   separate from the six fresh scientific fits. Prior four-worker collection
   throughput is not a fitting benchmark. Never resume benchmark weights.
4. Reuse `CumulativePulseTrainer`, input-ablation transforms and receipt-bound
   evaluation-only snapshots. A new typed schedule/prediction runner is needed:
   `independent_pulse_study_runner_development.py` requires the old
   `IndependentPulseEvaluation` and short pulse/brake plans. Do not fake that
   type or pass these longer family plans through its old validator. Authenticate
   the exact `FamilyWindowView` schedule and `remaining_candidate` plans.
   `FamilyPolicyStream` already has verified training and past-only inference
   interfaces. Score every actual role window with segregated labels; retain
   raw predictions and missingness, and keep transfer out of optimization.
5. After complete fit and raw-score verification, the predetermined first native
   bootstrap candidate is full-JEPA at seed2026091001, regardless of transfer
   ranking. Create new controller/loader/audit/mission bindings to combine that
   exact snapshot with the corner observer. Preserve the failed original
   bootstrap sources. Test actual downstream goal arrival in fresh native
   episodes; do not substitute retrospective scoring for control. Known panels
   are integration probes, not previously unseen maze evidence.
6. Continue to independent mazes, matched predictive/reactive/non-predictive
   control and RGB/JEPA/planning/memory ablations only with actual goal-reaching
   evidence. A no_rgb predictor with shared RGB-D localization is a predictor
   input ablation, not removal of all visual sensing. Moving suffix training
   does not cover every action switch at a moving state. The old turn-only
   behavior, full-loop timing deficit, exploration and physical backtracking
   all remain unresolved until new evidence demonstrates them.

No fit or benchmark runner has been launched at this checkpoint. All three
replay/input processes from this continuation are terminal. The committed
worktree remains unchanged; new implementation and result files remain local
and uncommitted alongside the original handoff.
