# Recorded-evidence checker prepared and scheduled

Completed update, 2026-09-10 15:02 UTC: the original replay, independent
checker and its waiter have completed successfully. Verification SHA-256 is
`03eb57a49fb357a6939e1c11a0bab30b29bb185a8e28a47ca51ceb2b6e8b9cc5`.
See [the completed result](go2_shared_surface_anchored_prefix_result_2026-09-10.md).
The preparation and waiting observations below are retained history.

The independent verifier for the original receipt-sharing replay is implemented
in `scripts/verify_go2_shared_surface_anchored_prefix_v1.py`. It accepts the
explicit completed result identity at the fixed launch, reconstructs all 405
saved original and expected candidate decision hashes, checks every command
endpoint and original stream/tape binding, compares public-input fingerprints
with both completed references, recomputes timing windows, and checks the four
reported state hashes against the completed reference. It does not rerun neural
inference, reload sensor packets or independently reconstruct hidden state.
The original replay owns those executions.

Validation completed before scheduling:

- 33 focused tests passed in 2.45 seconds, session 25001, exit 0. They reject
  partial/reordered populations, changed receipts, candidate/reference/public
  hashes, commands/endpoints, timing/order corruption, altered model/state or
  intervention scope, false completion/qualification claims and missing final
  results. They also verify that checker sources are bound before and after
  result checking, with no result written after a binding change.
- Source-only preflight passed 1,994 bindings, session 92056, exit 0. It created
  no verification result and executed no model or native scene.

An observation-only waiter, session 8557, PID 2707248, creation epoch
1789051304.83, is now waiting for the exact original replay owner
PID 2705097 / creation 1789049815.58. It pins replay launch
`3b2c3d3650ebefc2e314dab4de107877cc2dc805bed43bf5b190422bc6f4d97d` and the checked verifier source bytes.
After that owner exits, it requires a complete result and no failure, rechecks
sources, then invokes the verifier once with the actual final result hash.
Missing completion, changed owner/source identity, a two-hour wait expiry or
checker failure terminates without retry. No native job is launched or changed.

The scheduling record is `go2_shared_surface_anchored_verification_scheduling_2026-09-10.json`, SHA-256
`58414cd0dabaf94432056eb1b1a154ccf2ec8d8a0a19e5e7185ca0db71b42730`. The expected final verification
file is `go2_shared_surface_anchored_prefix_verification_2026-09-10.json`.
It does not exist yet; preparation and waiting are not a completed verification.
The replay had checked 322 observations at 14:42:18 UTC, with no failure and
no final result. All 405 observations and final original input admission remain
required before the checker can run.
