# Tracking continuity verification and shared-gyro dependence

The bridge/rejoin checker is implemented separately from the observer and its
production summary builder in
`lewm/independent_tracking_continuity_verification_development.py`.
The frozen observer, collector, learning study and pending tracking experiment
are unchanged. This is not a new native experiment or navigation result.

## What the new check reconstructs

From complete ordered sensor-result rows, it checks the fixed 100 ms clocks,
single initial reference, current/previous frame identities, retained-reference
membership and eight-reference retention, keyframe counts and promotion. It
recomputes bridge path length and consecutive/total bridge counts. A bridge
cannot promote an anchor, exceed ten frames, or invent anchor agreement. A
rejoin must preserve the preceding bridge history; its recorded position
disagreement is recomputed from the two saved position estimates.

End-of-recording, anchor rejoin and terminal failure are distinct outcomes.
Failure rows remain in the population, terminal observers cannot resume and
their failure reasons cannot silently change. Agreement does not become an
error bound, a claim of independent measurements, or a navigation qualification.

The checker is sensor-row-only: it has no native-data or filesystem access.
Artifact/source authentication, actual observer inference, stress intervention
reconstruction and the full experiment-level verification remain separate.
Unavailable-pose failure causes are not independently proved merely by matching
their saved labels.

## Confirmed scientific limitation: orientation agreement is shared by construction

This is an existing design property, not a newly introduced regression. The
tracker uses `register(..., mode='gyro')`. That mode fixes local rotation to
the supplied relative gyro rotation; RGB-D correspondences determine translation
and acceptance, not an independently estimated rotation.

Let `G_t` be the integrated gyro orientation. Initialization sets `R_0 = G_0 = I`.
For any retained reference `r`, the candidate orientation is

`R_candidate = R_r (G_r^T G_t) = G_t`,

provided `R_r = G_r`, which is preserved inductively by both anchor and bridge
updates. Different references therefore agree in orientation even if `G_t` is
biased relative to physical truth. Small floating-point deviations do not turn
this algebraic dependence into an independent visual-heading measurement.

Consequently the anchor/increment orientation-agreement check cannot establish
correction of a common gyro bias. Visual reprojection/residual tests can still
reject sufficiently inconsistent data; they are not claimed to be powerless.
The absence of such rejection does not establish unbiased heading.

There is also an evidence limit: the continuity record contains the incremental
position and a scalar rotational disagreement, but not the incremental rotation
matrix. The new checker independently reconstructs position disagreement and
range-checks the reported rotational disagreement. It explicitly leaves
`rotation_disagreements_independently_recomputed` false. An in-range but altered
rotation scalar is deliberately tested as **not independently certified**.

## Small numerical confirmation, not physical evidence

The existing exact-correspondence fixture was evaluated with the unchanged
production registration function at a fixed 0.002 rad imposed gyro error and
the existing deterministic frame-2 proposal schedule. Both modes accepted all
16/16 correspondences:

| Mode | Orientation error | Reported gyro disagreement | Translation error | Residual RMS |
| --- | ---: | ---: | ---: | ---: |
| Gyro-conditioned | 0.002 rad | 0 | 4.639063 mm | 1.821273 mm |
| Joint visual rigid fit | 1.46e-16 rad | 0.002 rad | 6.77e-13 mm | 8.23e-13 mm |

The gyro-conditioned fit reports zero disagreement because it uses the gyro
rotation as its fitted rotation. Translation compensates partly for that wrong
rotation: `t = mean(a) - R mean(b)`. Thus, on this fixture, small heading error
creates millimetre-scale translation error despite full inlier acceptance.
These are exact synthetic correspondences, not a realistic image-matching,
noise, latency, physical-motion or calibration experiment. No statistical
generalization, new robustness result or estimator replacement follows.

This is consistent with the earlier
[joint RGB-D fitting-only replay report](go2_joint_rgbd_rigid_pose_result_2026-09-06.md):
joint rotation tolerated its two small gyro-bias perturbations but had worse
nominal maximum position/orientation error than gyro-conditioned fitting.
Those prior result documents were inspected as source evidence here; their
runtime artifacts were not reopened or rerun. Joint-mode bias invariance is
also expected when gyro only acts as an acceptance monitor, not evidence that
the estimator learned or calibrated bias.

## Consequences for the next scientific steps

1. Keep the fixed collection and 36-fit prediction comparison unchanged. This
   observation is not a reason to delay or redefine that experiment.
2. Run the already specified tracking challenge after its existing gates. Treat
   shared gyro bias and shared depth drift as tests of correlated failure, not
   as cases that anchor agreement is assumed to solve. Report whether each
   intervention was actually reached while the observer was active.
3. Use the independently scored physical orientation/position errors to decide
   whether the frozen tracker supports fresh closed-loop turning. Do not infer
   heading accuracy from a small internal disagreement scalar.
4. If bias robustness is needed, compare an independently visual rotation
   estimate with gyro-conditioned tracking in a separately specified matched
   experiment, retaining nominal accuracy, availability, wrong-match failures
   and timing. Do not switch the current tracker on the strength of this toy
   result or tune thresholds until old failures disappear.

The ultimate test remains successful physical execution and memory/backtracking
on novel mazes with deployment-valid sensing, plus matched JEPA and online
rollout contributions. Bridge accounting and algebraic consistency are not
substitutes for that endpoint.

## Verification at this handoff

The final five-file regression passed **126 tests** in 30.51 seconds, including
35 new checker tests and the existing temporal-anchor, replay, independent
verification and joint-registration tests. Independently parsed JUnit counts
were 126 tests, zero failures/errors/skips. The tests exercise real observers
on synthetic packets, bridge/rejoin/terminal histories, repeated bridge spans,
eight-reference retention and deliberately changed evidence. They are not
physical or hardware trials.

JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_continuity_verification_final_v1.xml`,
SHA-256 `70a2bc5cd0d7cedd510a406c73efe58d6d441145cec934246ce875500c2e36c7`.

The numerical registration diagnostic exited successfully without creating an
experiment root or editing source. The original collector remained live on
`l08` (layout 9), with eight layouts / 960 eligible trials last verified. No
matched-study or new physical-tracking result is claimed.
