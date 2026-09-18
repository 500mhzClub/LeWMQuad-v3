# Hold-reorientation native comparison preparation

This source preparation is for a future prospective development experiment on
the existing maze 02, using the exact expanded full-JEPA model state
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
It launches no simulation and consumes no new layout. The original six-model
adapter batch and the already queued reached-frontier experiment remain intact.
The CPU raw replay waiter owns admission and execution of the prior raw replay.

New source files:

- `scripts/hold_reorientation_maze02_episode_development.py`
- `scripts/hold_reorientation_maze02_audit_development.py`
- `scripts/hold_reorientation_native_prefix_development.py`

The first two are narrow derivatives of the exact original residual-anchored
collector and auditor. Original SHA-256 identities, also verified against the
live raw waiter's inherited source map before derivation:

- Collector: `4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949`
- Auditor: `6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3`

Only the controller import/construction, collector status/log label, and explicit
hold-reorientation receipt flag/check change. Whole-module AST normalization
requires all other collection, persistence, sensor reconstruction, visibility,
command audit, navigation evaluation, contact and timing behavior to be exact.
The collector and auditor import the same HoldReorientationController. Existing
files were not modified or exported; there was no whole-tree materialization.

The new prefix helper admits only the completed raw replay of the fixed saved
boundary at observation 405. It reconstructs all saved comparisons, original
commands and public packet fingerprints; hashes alone do not replace those
checks. At a future native comparison it requires:

- Exactly matching first 21,000 physics samples, through pre-command sample
  20,999 at observation 405.
- Matching public packets at all 406 observations, including the boundary.
- Identical completed commands 0–404, and complete original hold/new left-turn
  commands at 405, both with endpoints 20,999 to 21,049.
- Every complete new native decision through 405 exactly equal to its raw
  prospective replay. Forecasts, utilities and geometry receipts remain exact.

Physics after the changed request is allowed to differ. It must be measured
and audited independently; the original later trajectory is not the new turn's
outcome. Completing a turn is not escape, goal-reaching or backtracking.

Tests completed in tool session 69926: 25 passed in 7.67 seconds. They cover
whole-module source differences, matching controller classes, reconstructed raw
comparisons, partial/mutated inputs, command endpoints, public/native prefix
equality, and permission for post-intervention physical outcomes to differ.
These are synthetic tests, not a completed physical or raw-controller replay.

Before launch, the required full raw replay must finish and be independently
verified. The original six-case batch and the queued frontier experiment must
finish in their existing order. A separately frozen single-case native launcher
still needs to admit those completed identities, the same assigned model and
original first-case artifacts, require an idle native slot, and enforce the
existing resource envelope. Keep original failures and all later outcomes. Do
not apply this candidate to the independent comparison layouts for tuning.
