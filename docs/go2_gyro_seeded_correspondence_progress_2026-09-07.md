# Gyro-seeded correspondence: complete negative result

Replay74693 now completes exit0 (f5abe8), and PID2076274 is absent (4fba70).
The candidate fails early on all six trajectories:471/7,433 available frames
versus7,400 for the original,6,929 lost and zero recovered. Do not adopt it.
See the [verified result and next initialization experiment](go2_gyro_seeded_correspondence_result_2026-09-07.md).
Result SHA256: `afe3baba0c13f542d4ea481b1814dd8a30183156bfb24754dc5baa31b08af697`.
Verification69278 passes691source/predecessor/26output bindings before a strict
numeric cross-check assertion fails.77301 investigates the4.53e-12rad maximum
quaternion-versus-near-orthogonal-matrix difference;74423 then completes all
row/clock/availability/first-failure and independent position/orientation summary
checks, with no experiment or acceptance-gate changes.65123 also completes the
12-case synthetic displacement diagnostic: original12/12, candidate11/12 with
failure at0.3m lateral displacement. Full regression89270 is terminal3,302pass.
The preparation and early-live observations below are historical, not live jobs.

The previous goal turn completed and verified the nine-stream registration
diagnosis. This turn implements the selected distinct correspondence hypothesis
and a complete paired replay. It does not weaken the frozen rigid-pose gates,
change the running collection or claim a new physical return.

## Implemented

`lewm/gyro_seeded_rgbd_correspondence_development.py` tracks deduplicated reference
features from a gyro/depth projection seed without requiring descriptor detection
in the current image. It filters failed/nonfinite/out-of-image forward endpoints
before reverse flow, keeps the 0.5-pixel bidirectional check, deduplicates current
locations and uses existing depth lifting/discontinuity filters. The fixed
candidate class inherits reference selection, promotion and failure latching,
and calls the unchanged rigid registration and increment checks. Per-attempt
counts and qualification/rejection evidence are recorded; latched no-op frames
cannot repeat stale candidate evidence.

This changes the correspondence checks: mutual descriptor association and
one-pixel descriptor-seed proximity are not retained. The gyro projection is an
initial guess, not a measured point match. One-pixel rigid reprojection and the
six-cell/inlier-fraction/displacement gates remain unchanged. There is no
commanded-displacement prior, native-pose input, alternate-method search or
physical-control integration.

`scripts/replay_go2_gyro_seeded_correspondence_v1.py` compares original and new
observers on all six completed inner/intent trajectories:7,433 recorded frames,
14,866 frame/arm observations. It authenticates the complete predecessor replay
and raw/source/native inputs, checks every original pose/selection/failure
against that replay and persists all sensor estimates before native evaluation.
The evaluator adds orientation errors and common-frame paired position/orientation
distributions, alongside all available errors, first failures and the full
both/original-only/candidate-only/neither population. This avoids comparing an
early-failed candidate's truncated accuracy with an original complete trajectory.

All timing remains offline observer-only, excluding packet loading and control,
with sequential original-first calls and possible overlapping CPU work. It is
not real-time, hardware or independent-validation evidence.

## Tests and limitations retained

- Initial matcher-test run3917 reports
  **25 passed / one failed in2.28s** (26 cases at that stage). The failed test
  wrongly expected a10mm common depth bias to pass existing gates. It was
  rejected. Preserve that as an expected rejection, and separately test2mm bias,
  which passes with wrong metric translation. No estimator gate was changed.
- Focused27401 completes exit0:60 tests passed in3.56s after that correction.
- The next combined apply_patch request has a malformed added-line marker and
  is rejected atomically; it does not run tests or change files. Corrected
  separate patches are applied afterward.
- Focused32390 completes exit0: **76 passed in11.70s**, including31 new matcher/
  runtime tests and12 new paired-replay/evaluator tests, plus existing rigid/
  correspondence tests. Known static-plane transforms include camera lever arms,
  gyro rotation and actual depth changes, not just identical-frame success.
- Repeated periodic imagery passes conditional registration despite being
  consistent with a different true displacement. This counterexample and small
  common-depth-bias acceptance remain explicit. Passing these tests means the
  limitations are represented honestly, not that ambiguity or bias is solved.
- Negative cases cover unknown/discontinuous depth, invalid gyro/image/masks,
  out-of-image/nonfinite flow, converging duplicate tracks, blank/occluded/
  unrelated imagery, wrong associations, gyro perturbation, stale/privileged/
  mismatched packets, failure latching, exact original replay mismatches,
  omitted/extra frames, corrupted sensor estimates, incomplete sensor phase,
  native clock/access failures, storage and source failures.

## Preflight and current work

Read-only10751 completes exit0 (ef9301/b56fd4). It verifies **691 source bindings**,
the predecessor result/output/raw/native witnesses and a24,052,854-byte launch
definition under the existing32MiB metadata limit. The new output is absent at
preflight. It independently verifies that the planned matched study's786-source
definition remains
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.

Reviewed source identities, unchanged in716689:

- Matcher/runtime: `2bb11822810a00b8b6abfc80fd2faca1567eccc970f384483d264f93c49e0050`
- Replay: `addfefaf07608e7e75aa3865327b88ccb13c54da5f233a31b7565144853281a7`
- Matcher tests: `f4fce12eebfeb4c26c63e16918219693af71e021c9767fef53ef800b0c543095`
- Replay tests: `0dab5fbf484d43741b0ccee0efc242352044861efe693d22f3d5c9f7426fd6bb`
- Protocol: `a5ba3b5c209bf9e866386c69f308bab3e8e53f25e6d14d06e8a572aa70307e06`

Full explicit243-file regression89270 completes exit0 (697455): **3,302 passed
in278.42s**, including all43 new tests. It is not a live job.
The original collector25963 remains live:
PID2063013 andl02 child2071161 are confirmed in716689, with latest62d004 stdout
at64/120l02 raw prechecks. No other collector/auditor is launched. l00/l01 remain
the completed audited layouts, not the complete12-layout study.

The single fixed paired replay is launched as74693, PID2076274, independently
confirmed live in6ae270. Read-only10168 completes exit0 (894d4b), verifies all691
launch source bindings and confirms no terminal result/failure yet. Launch SHA:
`e4bfb91e49acf8ce29358966c8671bddfed16206fd49f1b74daff7a7cfa359c4`.
Actual output6b2676 reaches inner-left frame750: original751 available poses,
new observer only58, with failure at frame58. This is an early real-data
regression, not an improvement. Preserve the full fixed replay rather than
changing the matcher or truncating the comparison. No native evaluation has
yet been observed. The collector reaches76/120l02 prechecks in d3f7c6.
Keep all691 launched source bindings unchanged and retain
all outcomes. Even improved availability will require additional independent
scene/sensor/error and end-to-end latency challenges before physical adoption.
The low-friction dynamics failure, matched JEPA study, online rollout/memory,
unfamiliar-maze missions and deployment/hardware evidence remain unfinished.
