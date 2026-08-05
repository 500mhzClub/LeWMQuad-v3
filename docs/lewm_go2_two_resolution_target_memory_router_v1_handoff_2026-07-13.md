# Two-resolution target memory and router V1 handoff

Date: 2026-07-13

Status: **hash-frozen candidate awaiting independent source review**

## Scope

This additive candidate closes the previously missing G5 posterior and target
route boundary without changing the frozen legacy G5 memory or the reviewed G3
V2 and G5 evidence sources.

The implementation consumes exact, single-use
`TwoResolutionPositiveTargetEvidenceV1` and
`TwoResolutionNegativeTargetEvidenceV1` objects. It stores sparse target mass
on the `0.10 m` configuration lattice while retaining the source evidence's
`0.05 m` physical-frame, calibration, runner, checkpoint, revision, and
conversion bindings.

## Frozen identities

- posterior source, `lewm/planning/two_resolution_reversible_target_belief_v1.py`:
  `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3`;
- posterior tests, `lewm/tests/test_two_resolution_reversible_target_belief_v1.py`:
  `6c20fd5b237673aed10a2d03759ed87af057cc55da461a45f046444367540cee`;
- router source, `lewm/planning/two_resolution_target_router_v1.py`:
  `fbef970bc8637c2c87159edaeffa779b3da12b7f6b9bd4ae67af4f14dd3df252`;
- router tests, `lewm/tests/test_two_resolution_target_router_v1.py`:
  `506858cdae10bd2ff8b9644a839c2034fd56b1487e2489bdd5ada9c92f52a3b6`;
- upstream frozen two-grid evidence source:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2`;
- upstream frozen two-grid evidence tests:
  `e33dbf595fe27c18c2fddf89cc8f22a005574f67348c2d8746b8ee1ca039de26`.

## Posterior behavior

- one exact evidence issuer and one non-copyable memory instance;
- sorted registered target identities and one normalized sparse posterior plus
  unlocalized mass per target;
- bounded positive transfer from unlocalized mass without decaying existing
  modes;
- negative transfer only over issuer-certified visible configuration cells;
- a strictly positive posterior floor, so negative evidence cannot delete a
  mode and later positive evidence can recover it;
- separated four-connected modes remain separate deterministic hypotheses;
- context, evidence, and raw-outcome replay rejection;
- immutable two-frame, two-shape, projection, support, runner, checkpoint,
  calibration, and taint bindings;
- exact-instance posterior snapshots that become stale on any later memory
  revision;
- explicit `production_promotion_authorized=false` and
  `hardware_execution_authorized=false`.

## Router behavior

The router consumes one exact current G3 V2 snapshot/component and one exact
current posterior snapshot. It:

1. orders posterior modes deterministically by mass, peak mass, and cell;
2. enumerates only confirmed-FREE component cells within the canonical
   `0.10 m` to `1.20 m` target-view range;
3. excludes every hypothesis cell and runner-excluded target cell from both
   the terminal set and the complete retained path;
4. retains and revalidates an exact live G3 V2 A* path;
5. computes the target-facing terminal yaw and records the canonical
   `0.25 rad` bearing contract;
6. binds candidate-set, posterior, evidence-chain, both map revisions/frames,
   morphology supports, start pose, exact path, route cost, and score in a
   non-copyable single-use receipt;
7. explicitly denies production promotion and hardware execution.

The candidate deliberately prefers conservative rejection to admitting a path
through a target hypothesis. The frozen G3 V2 A* API has no temporary-forbidden
cell argument; consequently a candidate is rejected when the exact retained
shortest path crosses a hypothesis even if a longer alternative may exist.
This can create a false negative, never an unsafe positive, and must be
reported by the future navigation rollout.

## Verification completed by author

- posterior plus frozen G5 evidence focused suite passed;
- router focused suite: `4/4` passed in `41.65 s` under CPU-only one-thread
  caps;
- Python compilation passed;
- `pyflakes` passed;
- `git diff --check` passed;
- no GPU, G2, held-out, sealed benchmark, or navigation rollout was opened.

## Independent review requirements

The reviewer must reproduce the frozen hashes and test:

- positive, negative, and later-positive recovery with exact normalization;
- repeated negative evidence never reaches zero;
- separated modes and high-index `0.05 m -> 0.10 m` conversion;
- copied/replayed evidence and copied/stale posterior rejection;
- wrong frame, origin, shape, revision, support, checkpoint, calibration, and
  candidate-domain rejection;
- complete-path exclusion of hypothesis/target/UNKNOWN/OCCUPIED cells;
- deterministic selection, terminal yaw, `0.10 m` path cost, and exact G3 path
  issuance;
- copied/replayed/stale route rejection and both authority denials;
- adjacent frozen G3 V2, G4 V2, G5 evidence, and legacy G5 regressions.

