# Matched study executable preparation

The previous goal turn made progress by implementing four matched input
treatments, checking seven authenticated recorded examples and completing the
3,198-test regression. This turn connects those treatments to an actual bounded
experiment executable. No new recorded-data fit, navigation action, hardware
operation or competing collection is launched.

## Implemented end-to-end path

`scripts/run_go2_independent_pulse_matched_study_v1.py` connects completed supervisor
receipts, the full twelve-layout loader, explicit coverage gates, the authenticated
policy-only stream, identical per-seed schedules, all objective/treatment fits,
durable update ledgers, bounded final snapshots, evaluation-only reloads, exact
prediction-array persistence and common per-seed/per-layout scoring. All failures
stop the sequence with partial evidence retained and no implicit retry.

The prospective protocol is
`docs/go2_independent_pulse_matched_study_v1_2026-09-07.md`. It specifies three
paired seeds, four input treatments, three objectives, 1,200 updates per fit and
batch size 6: 36 fresh fits / 43,200 updates / 259,200 sample draws. The last
snapshot is fixed in advance; selection-role scores do not choose weights.
Direct and rollout objectives are not FLOP-matched; the specific latent-training
comparison is JEPA versus supervised rollout on the same input treatment.

The zero-motion baseline retains empirical contact predictions and is named
accordingly. Missing empirical cells remain unavailable. Motion prediction and
common-horizon contact ranking stay separate; choosing a low-contact action does
not establish progress, task success or a learned navigation policy. Complete
prefix matching and layout/action availability are gates; native contact,
collision-free motion and future-image supervision have separate coverage counts.

## Evidence and exact identities

- Initial focused 15826 completes exit 0: 28 tests pass in 16.48 s.
- After adding explicit treatment/weight/source-mismatch faults and a ledger
  failure after an actual optimizer update, focused 29201 completes exit 0:
  **145 passed in 30.10 s**, including 32 new experiment tests, 48 ablation tests,
  26 numerical-runner tests and 39 snapshot tests.
- The synthetic end-to-end fixture executes all twelve objective/treatment
  combinations for one seed at a reduced one-update budget. It checks identical
  initial weights/schedules, all update ledgers, actual snapshot reloads, saved
  prediction indices/weights/treatments, common scores, missing baseline cells
  and final artifact hashes. These are synthetic fits, not scientific results.
- Synthetic failure cases cover data coverage, materialization, durability after
  a completed step, snapshot saving, inference, final data verification, changed
  source identity and mismatched snapshot/prediction treatments or weights.
  Incomplete or unbound terminal cohorts fail before loading or output creation.
- Read-only 72998 completes exit 0 (66e9a7): the prospective definition validates
  **786 source bindings**, preserving all **771 live supervisor sources**.
  The supervisor result is absent and the scientific-study output is absent.
  No scientific preflight/fit or runtime output was launched by this check.
- Full explicit 239-file regression 6754 completes exit 0: **3,230 passed in
  262.73 s**, including all 32 new experiment tests.
- Final read-only 35028 completes exit 0 (3e22f8): the exact 786-source definition
  remains unchanged, all 771 live bindings verify, the scientific-study output
  remains absent and the updated autonomous checkpoint is valid JSON.

Exact prospective canonical definition SHA-256:
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.

Bound source identities:

- executable: `f66f078cd04c2225add4f517f6b6242f31e1f8dac7da04f0e8665e3bf2346452`
- experiment tests: `bce0096b1c9482e7e14e2b0f5bf1a69f9dde009e5b48bd81397148a136878bdb`
- prospective protocol: `54a59c5fad17ad8efbe1cddffd27ed267bb532280ed2cf6e2582277a1999993d`

This preparation document and the mutable autonomous checkpoint are outside the
prospective source definition. Do not edit the launched 771-source collection
closure. Preserve the 786-source experiment definition for the planned fit;
any necessary prelaunch change requires fresh review/testing and an explicitly
updated definition identity, not silently using the old digest.

## Running work and next action

The original collection supervisor 25963 remains live, with l01 child PID 2063119.
Its latest observed progress reaches 104/120 l01 raw prechecks (5a94ca); both
parent/child are independently confirmed live at 2995/2943 s elapsed (f1cca0). l00 remains
completed and audited; the full cohort is not yet complete. No new collection
or audit process is started by this work.

Regression 6754 is terminal successful, not a live job. Continue observing
the exact supervisor and its own children. Once it is authoritatively terminal
and successful, inspect its exact result path, bind that result SHA-256, verify
the unchanged definition above and invoke the study's two-hash CLI in its exact
CPU environment. Do not invent a receipt while it is still running, launch on a
partial cohort, or replace a failed sequence. The ordinary CLI then performs
full preflight and coverage gates before fitting. A source/coverage/data failure
requires its actual evidence to determine the next action, not an automatic retry.

Meanwhile, meaningful noncompeting work can address the existing local-execution
failures and preparation of online memory/rollout tests outside frozen sources.
Read-only review this turn confirms that the latest paired inner-arrival result,
not the earlier room program, is the relevant execution baseline: both nominal
routes lose tracking during TURN_BACK, and low friction fails the first leg.
The [completed inner-arrival audit](go2_inner_arrival_collection_result_2026-09-06.md)
localizes each nominal failure to five-versus-six spatial support cells, despite
passing captured inlier-fraction/translation checks. This does not establish that
those rejected estimates were reliable. A prior balanced frontend recovered one
old failure interval but lost 514 frames overall, so it is not a validated fix.
The next noncompeting perception task is whole-stream conditioning/error/latency
diagnosis under unchanged gates, not another same-room tolerance change or
retrospective per-trial frontend selection. Native truth must stay evaluator-only.
The full goal remains unfinished: the latest actual room return is 0/3, prior
JEPA predictors lose the empirical baseline, and unfamiliar-maze navigation,
real-time/deployment-valid sensors and bounded hardware evidence remain unproved.
