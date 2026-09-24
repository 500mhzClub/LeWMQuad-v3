# Staged population driver prepared

Implemented `scripts/independent_round_trip_staged_population_development.py`.
The driver connects the fixed schedule to fresh collection and audit spawn
workers, explicit start barriers, parent-owned process registration and
completion, complete collection handoff, separate raw audit, and failure
draining. Its native-slot check excludes only the exact registered owned audit
process after a source-bound overlap verifier has returned successfully.
Unrelated native runners and unregistered spawn workers remain competitors.

**104 tests passed in 56.09 seconds**, session 35227, exit 0. Tests include all
32 cases through 64 distinct synthetic stage identities, controlled worker
failures and cancelled starts, real lightweight spawn barriers, an actual
owned live audit process in the native-slot exception, synthetic collection
with real handoff hashing, and source-bound verifier rejection. No native
scene, original raw audit or trained-model execution occurred in these tests.

Preparation verified **2,033 source bindings**, session 46984, exit 0:
`docs/go2_independent_round_trip_staged_population_preparation_2026-09-11.json`,
SHA-256 `04934a58658e69ba16cc816900c6f24fd183e3f20e33d139488cc25d51482025`.
The first preparation observation, session 6757, exited before writing its
record because the original scoped replay ended during observation. That
completion was authenticated before preparation was recorded again.

The driver has no CLI or final launcher. Actual CPU-only audit qualification,
its source-bound overlap verifier, final study policy review and a source-bound
joined input/extended-queue verifier remain necessary before execution.
Overlap timing and memory have not been measured. Existing native jobs and
their queue were unchanged; no independent-layout data were collected.
