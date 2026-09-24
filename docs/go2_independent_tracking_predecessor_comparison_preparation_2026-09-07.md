# Actual predecessor geometry, starts and value-only sensor prefixes

Implemented bound predecessor-witness extraction and post-evaluation comparison
for the planned eight-trial tracking challenge. This does not collect a new
scene, rerun an observer, adopt a tracker or establish maze generalization.

## New evidence from existing recordings

Read-only18111 terminates exit0 after verifying the exact launch, collection
result and terminal raw audit of both old room-return cohorts, then the25
explicitly accessed artifacts for each of their three trials. It extracts nine
RGB-D/body/gyro observations and the first1,150 native pose samples per trial.
Bindings are checked again afterward. Read-only50485 additionally measures the
actual RGB pixel differences at the unequal frames, with before/after bindings.

Corresponding `inner` and `intent` predecessors have identical four-wall native
geometry, initial poses and native pose prefixes. All nine depth, body, fast-gyro
and control value witnesses also match. Nevertheless:

| Paired condition | Unequal RGB frames (zero-based) | Changed pixels per unequal frame |
| --- | --- | ---: |
| Nominal left |6,7,8 |1 of307,200 |
| Nominal right |3,4 |1 of307,200 |
| Lower-friction left |None |0 |

The largest channel difference is21 on left frame7. The cause of those isolated
RGB differences has not been established here. This is concrete evidence that
**different RGB hashes alone cannot establish independent observations**. All
three paired predecessors correctly fail the combined new-observation check.
Their original room-return failures remain unchanged; nothing is rescored as
navigation success.

Complete selected bindings, native geometry/start witnesses, value-only hashes,
comparisons and pixel counts are saved in
[the actual predecessor evidence](go2_independent_tracking_actual_predecessor_witnesses_2026-09-07.json).
The file binds the comparison helper and accessed artifact bytes, not a recursive
source closure, new benchmark role or execution authorization.

## What the comparison checks

The existing whole-packet prefix identity includes metadata and commands. The
new helper keeps those integrity checks separate from observational content:

- RGB identity is based on the decoded pixel array, not PNG bytes or filenames.
- Depth identity includes metric values and valid-ray masks.
- Body and fast-gyro identities use measured values and validity masks.
- Controls are reported separately and cannot supply observational novelty.
- The sensor reader still validates calibration, timestamps and episode contracts;
  ignoring metadata for a value comparison does not authorize invalid packets.

Native geometry uses world-space box corners, ignoring names, materials, object
ordering, quaternion sign and equivalent box-axis permutations. Bipartite
matching uses a10µm numerical-comparison tolerance, not an estimated physical
error bound. A materially separated actual initial start requires at least5cm
planar separation or0.05rad orientation separation. The combined check also
requires both the first RGB view and first depth values to differ: later motion,
changed commands or label differences alone cannot pass.

Missing geometry, fewer than nine sensor frames or fewer than1,150 native
samples produce an unavailable comparison, not a novel observation. Each new
trial is compared against all six declared predecessors, retaining all48 pairs.

These are deliberately **nonidentity checks**, not a statistical-independence,
global rigid-congruence, perceptual-diversity, topology-isomorphism or unseen-maze
generalization test. Changing the partition of a physical wall into boxes can
change an inventory without changing its union surface. Globally transformed
duplicates also require stronger equivalence analysis. The planned two new
scene constructions and eight treatments must not be reported as eight
independent mazes. Qualification and full-challenge flags stay false.

## Native-access ordering

`compare_new_population` first authenticates and reconstructs all eight base
streams and88 stress streams. It then requires the bound completed raw-audit/
base-plus-stress scoring result and all eight raw-audit outputs. Only afterward
can it parse new native poses or geometry for comparison. All new sensor/result
bindings are checked again after comparison. Predecessor input is limited to
the exact six already exposed development recordings; there is no recursive
search, sealed access, source export or predecessor experiment rerun.

The function returns its report; it does **not** overwrite the completed score
result or create a comparison artifact on its own. The future native launcher
must persist and bind that report at an explicit reviewed path and include its
metadata allowance in the total resource contract. The existing stress-component
envelope remains35.0703125GiB within36GiB; additional launcher outputs are not
silently included in its214-path roster.

## Tests and next actions

Focused71825 terminates exit0:32 tests pass in2.04s. They cover metadata/control
counterexamples, equivalent native boxes, numeric perturbations, missing prefixes,
clock/RGB/native association faults, bounded predecessor rosters and failed
authority/phase ordering. An additional mocked population test covers all48
returned pairs and repeated final phase authentication. It is integration wiring
evidence, not an acquired new scene or native reconstruction qualification.
Final adjacent regression is recorded below after completion.

Next, finish the actual native eight-trial launcher, explicit comparison-result
persistence and the frozen source/runtime/resource review. Validate deferred
native storage and memory bounds and actual predecessor/nonidentity reports.
Preserve the original live collector and pending36-fit matched learning study;
do not start a competing native job or displace that study. Only a completed,
independently verified challenge can support a fresh closed-loop turn/return
adoption assay. Latest actual room return remains0/3. Useful JEPA prediction
beyond baseline, online rollout benefit, memory/backtracking, novel-maze success
and deployment-valid hardware evidence remain unachieved.

## Final preparation verification

Invocation4563 terminates exit0: **330 tests passed in280.36s** across thirteen
explicit files, including33 new predecessor-comparison tests and297 adjacent
tests. JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_predecessor_adjacent_v1.xml`.
No runtime or test source changed during this invocation. The result is not a
new full-repository regression, native challenge or hardware qualification.

Read-only74876 terminates exit0: the unchanged786-source matched-study definition
and701 completed-replay source hashes verify; all four new source/test/report/
witness paths are outside both frozen sets. Check8fb926 verifies the saved
six-prior/150-selected-artifact witness roster, comparison-helper identity and
five one-pixel differences. Original experiment outcomes are preserved.

The original supervisor remains live on l05's terminal audit. All120 collection
prechecks are recorded, but the final l05 audit and supervisor verified receipt
are not yet present at55c286. Only five completed layouts count toward the
verified600-trial dataset. No competing native job, new model fit or controller
adoption was launched.
