# Rotation-only flow initialization fails the complete development comparison

Do not adopt this estimator. The fixed gyro-seeded correspondence candidate
provides poses on **471/7,433 frames**, compared with **7,400/7,433** for the
original descriptor-seeded observer. It loses6,929 previously available frames,
recovers none, and fails early in every recorded trajectory. This is a completed
negative scientific result, not an infrastructure failure or a physical return.
The latest simulated room return remains0/3 and no JEPA benefit is established.

## Completed evidence

- Full243-file regression89270:3,302 passed in278.42s, exit0 (697455).
- Actual691-source preflight10751: exit0,24,052,854-byte launch definition,
  original inputs/native/replay witnesses verified, output absent before launch,
  planned786-source matched-study definition unchanged.
- Replay74693, PID2076274: terminal exit0 (f5abe8); PID absent in4fba70.
  All six paired sensor streams finish before native evaluation. Every original
  pose, reference selection and terminal failure reproduces the predecessor.
- Verification69278 authenticates691 sources, predecessor raw/native/output
  bindings and26 new output bindings before an overly tight independent numeric
  comparison fails. Investigation77301 recomputes all streams and measures
  position disagreement at most1.20e-16m and orientation disagreement at most
  4.53e-12rad. The independent quaternion conversion projects accumulated
  near-orthogonal gyro matrices; their maximum orthogonality defect is2.11e-11.
  The original gyro implementation multiplies incremental matrices and checks
  orthogonality at1e-8. No recorded estimate or experiment gate changes.
- Completed verification74423 exits0 (147885). It checks all7,433 ordered
  sensor/evaluation row pairs, exact native sample indices/timestamps, first
  failures, availability categories, all available/paired position and
  orientation summary distributions, and final artifact hashes. Independent
  numerical tolerances are1e-12m and1e-10rad; these compare calculations only,
  not physical acceptance. It retains the earlier checker failure explicitly.

Exact output root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_gyro_seeded_correspondence_replay_v1_attempt_001`.

Launch SHA-256:
`e4bfb91e49acf8ce29358966c8671bddfed16206fd49f1b74daff7a7cfa359c4`.
Complete sensor-phase SHA-256:
`ed02513aacfa22ef790b039403189296587e9a33403d5630fb6d12a861a2be53`.
Result SHA-256:
`afe3baba0c13f542d4ea481b1814dd8a30183156bfb24754dc5baa31b08af697`.

## Outcomes without truncation bias

| Recorded stream | All frames | Original available | Candidate available / first failure |
| --- | ---: | ---: | ---: |
| Inner left |1138|1127|58 /58|
| Inner right |1014|1003|98 /98|
| Inner low friction |576|576|78 /78|
| Intent left |2413|2413|58 /58|
| Intent right |1820|1809|98 /98|
| Intent low friction |472|472|81 /81|

There are471 both-available frames,6,929 original-only frames, zero candidate-only
frames and33 neither-available frames. All14,866 frame/arm observations, including
latched terminal no-ops, remain in the analysis. These are correlated development
observations, not14,866 independent trials or independent-maze validation.

On the shared early frames, original/candidate mean position errors in mm are
0.456/0.426,0.894/0.697,0.432/0.233,0.456/0.426,0.894/0.698 and0.439/0.428 in the
table order. Those small early-frame differences do not compensate for the
availability collapse. Candidate maximum shared-frame error is worse on intent
low friction (1.359mm versus0.966mm). Both methods use the same measured gyro;
paired orientation errors agree to numerical precision, which is not evidence
of a new orientation capability.

Candidate available-frame observer medians are approximately40.0–40.5ms versus
38.3–42.6ms for the original's available frames. These timing populations are
different and exclude packet loading and the full controller. Candidate all-frame
medians near0.04ms mainly measure terminal no-ops and must not be presented as
a speedup. There is no isolated real-time or hardware result.

## What was learned and what remains uncertain

The method deliberately removed mutual descriptor association and descriptor-
endpoint proximity, retaining bidirectional flow, depth consistency and all
rigid/grid/fraction/increment gates. It used a zero body-translation guess when
projecting reference points with current gyro rotation. More surviving tracks
did not ensure adequate consistent support.

Read-only5130 checks unchanged closed inner-left/right sensor files. On left
frame57,266 unique reference locations yield210 depth-valid tracks and133
consensus points (fraction0.633), spanning8/8 grid cells. Frame58 retains210
tracks but fails the composite rigid consensus/grid/displacement gate. On right
frame97,185 tracks yield116 inliers (fraction0.627), also8/8 cells. At frame98
the primary retains181 tracks but fails, and its older reference also fails.
Rejected consensus details were not captured by this candidate's compact trace;
do not claim the exact final failing gate from the composite error alone.

Read-only65123 completes a fixed synthetic diagnostic after the method is frozen:
body-forward and body-lateral translations of0,.02,.05,.1,.2,.3m against the
existing textured2m plane fixture. The original succeeds in all12 cases. The
candidate succeeds in11, but fails the0.3m lateral case with51 matched tracks;
the original has314 inliers there and0.0363mm translation error. At0.2m lateral,
the candidate's consensus fraction is0.834 versus1.0 for the original. These
results support an initialization/capture-range limitation, but do not prove
that it is the sole cause of the room failures. The initial passing geometric
tests covered small motions; they missed this larger reference-displacement
challenge and near-field/multiple-depth/occlusion combinations.

Periodic-texture ambiguity and accepted small common-depth bias remain separate
counterexamples. An initializer cannot make unobservable motion observable or
turn residual/grid checks into calibrated uncertainty. Do not lower the inlier
fraction, support count or reprojection thresholds to rescue this result.

## Next bounded experiment

Test **initialization from the previous measured pose**, not another zero-guess
frontend or an executed-command prior. For reference rotation/position R_ref,p_ref
and the most recent accepted position p_last, initialize reference-frame
translation as R_ref.T*(p_last-p_ref). Project observed reference points using
that translation and current measured relative gyro rotation. The previous pose
is only an optimizer initialization; do not output it as the current pose or
accept flow merely because it agrees with the seed. Keep the current-time RGB-D
measurement, consensus, reprojection, spatial support, increment and reference
checks unchanged. Use no native pose, future image or command-derived translation.

Implement this as a distinct prospective method in new files; preserve the691
bound sources and terminal result here. Before another full recorded comparison:

1. Use multi-frame synthetic image/depth sequences so the seed comes from a
   previous estimated observation, not ground-truth displacement. Cover reference
   displacement through the existing keyframe operating range, both translation
   directions, varied depths, rotations, occlusion and multi-surface scenes.
2. Include deliberately wrong/stale prior-pose hypotheses and repeated textures.
   Retain false acceptances and missingness; do not claim that a warm start solves
   ambiguity or bias. Sensor/clock/identity corruption remains terminal.
3. Compare original descriptor association, frozen rotation-only flow and the
   measured-pose initializer on fixed sensor pairs/sequences. Then freeze the
   method before whole-stream replay, with all failures and paired errors retained.
4. Only if complete results support it, proceed to additional independent scene/
   sensor/error and end-to-end latency challenges, then new closed-loop local
   execution. If the hypothesis fails, retain it and reconsider complementary
   sensing or handling of temporary visual unavailability rather than silently
   resetting pose or searching thresholds on these same failures.

In parallel, continue the original collector25963 through all12 successful layout
receipts before launching the already-reviewed36-fit JEPA/supervised/input-ablation
study. Reliable execution, useful learned online rollouts, memory/backtracking,
unfamiliar-maze success, realistic sensing, real-time control and bounded hardware
evidence remain unfinished. This rejected method does not replace those goals.
