# Go2 V5 joint-training execution gap audit

Date: 2026-07-13

Status: **read-only first-principles audit; navigation readiness blocked**

## Finding

The repository has a reviewed single-encoder V5 model, complete joint-loss
arithmetic, exact migration from a V4 fit model, and a reviewed staged G2/G3
execution lifecycle. It does not yet have an executable, reviewed path that
produces the shared V5 checkpoint which that lifecycle is meant to evaluate.

A passing standalone V4 fit checkpoint is therefore necessary but not
sufficient for navigation readiness or one-shot G2.

## What exists

- `ObservableCameraRayEvidenceV4Model` trains the observable camera-ray
  encoder/head and publishes a development-only fit checkpoint.
- `SharedObservableCameraRayJepaV5.migrate_from_fit_model` exactly copies that
  fit encoder and every compatible evidence-head state, then hard-syncs the
  EMA target. `V4HeadMigrationReceiptV5` binds the migrated states.
- V5 has one online `VisionEncoder`, an action-conditioned JEPA branch, the
  encoder-free V4 evidence head, and mandatory current/next four-equal V4 loss
  arithmetic. Its model/output/loss bytes passed independent review.
- The staged one-shot runner/finalizer/publisher source passed independent
  review and remains correctly fail-closed with all production identities
  unset.
- The corrected paired navigation dataset has train, checkpoint-selection,
  probability-calibration, and untouched-G2 roles plus current/next RGB,
  actions, and transition provenance.
- The immutable V4 fit dataset contains exact raw camera-ray supervision for
  the current and next endpoints of 160 train-role transitions. The frozen fit
  panel retains each transition's primitive and relative SE(2), so a strict
  join is possible without regenerating the N=320 train labels.

## What is missing

1. No non-test call site invokes `migrate_from_fit_model`,
   `forward_training_pair`, or `combine_joint_losses`.
2. No reviewed trainer jointly optimizes the JEPA package and both frames'
   mandatory four-equal V4 objectives, updates the EMA target exactly once per
   optimizer step, or publishes a V5 training record.
3. No reviewed data boundary performs the available exact join between the
   160 paired fit-panel transitions and their 320 raw V4 endpoint records. The
   current raw artifact also has no registered checkpoint-selection or
   probability-calibration role supervision. The paired navigation labels
   alone are derived physical rasters and cannot replace the required raw V4
   objectives.
4. No matched no-JEPA ablation runner shares initialization, ordering,
   architecture, supervised losses, and budget with the promoted arm.
5. No V5 checkpoint-selection/JEPA-health implementation, raw-evidence
   calibration/threshold finalizer, or immutable training-record publisher is
   wired to the staged one-shot checkpoint authority.
6. The staged G2 runner can evaluate only an already existing, bound shared
   checkpoint. It cannot and must not manufacture the missing checkpoint.

Repository-wide non-test call-site searches confirm these absences. No G2,
held-out, sealed, checkpoint output, GPU, or navigation input was opened for
this audit.

## Required closure order

1. Freeze and independently review a development-role dataset successor that
   produces exact paired raw-V4 supervision for train, checkpoint-selection,
   and probability-calibration roles while keeping the untouched G2 role
   unopened.
2. Freeze seeds, optimizer, schedule, selection cadence, JEPA-health rules,
   escalation limit, calibration algorithm, matched-ablation contract, and
   immutable output namespace before learned output.
3. Implement and independently review the V4-to-V5 migration plus joint
   trainer. Every update must use the established JEPA package and both
   complete current/next V4 loss packages; hierarchical-raster-only training
   remains ineligible.
4. Run the promoted arm and matched no-JEPA ablation on train only, select the
   promoted update on checkpoint-selection with mandatory JEPA-health
   eligibility, and evaluate the ablation at that exact update.
5. Fit calibration and thresholds once on probability-calibration, publish an
   immutable V5 training record/checkpoint, and independently reconstruct its
   state, inputs, selection, calibration, and access ledger.
6. Only then bind the staged G2 runner authority and consume the untouched G2
   role once.

## Readiness consequence

The current navigation-readiness milestone must contain this V5 checkpoint
production gate between the V4 fit ladder and G2. Treating the V4 fit
checkpoint as the G2 checkpoint would bypass the JEPA objective, the shared
runtime model, the matched ablation, and the V5 checkpoint provenance contract;
that shortcut is forbidden.
