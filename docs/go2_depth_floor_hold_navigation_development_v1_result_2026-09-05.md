# Depth-floor / active-hold V1: audited advancement and remaining failures

Collector75481 completed both fixed missions, exit0. Full audit28860 passed
1,040 controller decisions and1,051 RGB/depth/relative-state observations.
Prelaunch41905 passed1,155 tests across108 files in77.89 s; preflight94977 and
terminal audit verified316 source,210 input,2 gait and10 native bindings. The
old observable-hold integration failure remains frozen and unchanged.

## Complete mission outcome remains0/2

Both trials passed first scan initialization and active alignment, then executed
a second traversal. North stopped on weak lateral motion constraints after45.6 s
(path3.754770 m, home distance2.524478 m). South completed a second measured local
arrival and began its second scan, then stopped on turn-volume evidence after
59.3 s (path4.034926 m, home distance2.666209 m). Neither discovered the marker,
returned home, recorded contact/native body stop or made a false home claim.
Both independent terminal zero-release checks pass. Trusted graph edges stay zero.

This is progress through the projection/hold integration and the previous
alignment failures, not whole-maze success, a memory benefit, learned navigation
or JEPA-contribution evidence. The population is still two reused development
layouts. The result does not establish robust performance on either layout.

## Active alignment and constrained approach now execute

North alignment completes at tick325 with heading error-0.00422144 rad and
projected rate0.01197890 rad/s. The retained request is0.05071442 rad/s with
integral0.05704658: explicitly active holding, not zero-command rest. South
completes alignment at tick328, enters the second traversal at344, completes
its second local arrival at536 and begins a second scan at552.

Depth-ground projection accepts genuine holding history instead of attempting
the old zero-command gravity bootstrap. The second approach uses an observed
blocking/view-constrained target and logs conservative target tightening.
North's last valid frame retains1,114 floor points, so the previous vertical
floor-support failure is not the present failure mechanism.

## North: weak lateral constraint, not an observed zero displacement

At observation451, the normal spectrum is[0.00579429,0.26426290,0.72994281].
The smallest eigenvalue falls below the predeclared relative rank threshold.
The reported weak direction[-0.06460,0.99685,0.04601] is predominantly lateral.
Only a small left-wall support remains while front wall/floor dominate the view.
The estimator reports rank2, invalidates cumulative position, and control stops.
This is weak geometry under the declared test, not proof of mathematically zero
information: there is a finite small eigenvalue. A threshold reduction without
measurement-error/uncertainty analysis is not justified by this trace.

North has450/456 fully observed intervals, maximum accepted step error1.09151 mm
and no final cumulative position, so its motion check fails. All457 depth-frame
checks pass. The missing-component stop remains part of the scientific result.

Read-only feasibility probe72aa4d used only causal measured force/gyro, the last
observed depth velocity and observed translation projections to estimate the six
weak intervals451–456. It propagated acceleration in the initial body reference
and filled only the weak subspace. Ground truth was used only for scoring. Weak-
component step errors were0.329,0.245,0.033,0.121,0.201,0.242 mm; resulting final
position error was1.297 mm. This is a posthoc short probe over the actual stop/
release trajectory, not continued-navigation evidence, calibrated covariance,
bias robustness, an online implementation or a rescued predecessor result.

## South: the nominal turn envelope loses margin during actual turning

South retains full motion estimates on593/593 intervals; maximum step error
0.81463 mm and final error0.90659 mm. All594 depth checks pass. At tick588, after
about1.13 rad of the second scan turn,2/951 nominal volume samples are near
observed surfaces and no longer meet the existing4-cm ray margin. The controller
zeros the command before a recorded collision; it does not ignore the two samples.

Full actual-input diagnostic73921 localized those samples. Evaluator-only world
positions are approximately[2.08694,2.14581,0.10627] and
[2.08690,2.14580,0.18465] m, about3.3 cm from the wall face at x=2.12 m. They
are not ground-support samples. During the scanned interval56.7–60.3 s, actual
body translation is[+0.03495,-0.01519] m. Thus initially supported nominal turning
volume did not account for the full translation of the learned gait during yaw.
Current-pose checking correctly detected the margin loss, but did not supply a
predictive turning operator. Do not remove its margin or call no contact a pass.

## Next scientific execution plan

1. Implement a separately named causal depth/gyro/accelerometer state estimator
   that explicitly distinguishes directly constrained and inertially predicted
   translation components. Initialize velocity from available observations;
   retain covariance/error-growth information and bias/gravity assumptions.
   Do not report predicted components as rank3 depth observations. Validate
   clocks, dropout, bias, acceleration/turns, weak-geometry duration and recovery
   using predeclared development populations, not only the successful six-frame
   probe. Couple pose uncertainty to transported ray evidence and stopping.
2. Make observation turns account for empirical gait translation and available
   clearance. Use current observed space to choose a better observation position
   or turn action; if necessary acquire another view before losing state. Retain
   the2-ms contact stop and current nominal-volume guard. A measured stationary
   footprint is not a future turning trajectory.
3. Collect matched action-conditioned development trajectories for the actions
   now known to matter: both yaw directions, hold, braking and translation/turn
   combinations at multiple clearances and durations. Use them to compare a
   strong geometric/empirical-dynamics baseline, supervised prediction and JEPA
   prediction under identical sensors, action choices and budgets. Test whether
   predicted turning displacement/risk actually changes chosen actions. This
   gives predictive learning a concrete role rather than indefinitely replacing
   each failure with a new hand-tuned local constant.
4. Integrate the validated estimator and dynamics-aware local choice into fresh
   full exploration/discovery/return trials. Then require matched online-memory
   and genuine multi-step rollout comparisons, independent layout/model seeds,
   sensor robustness and bounded physical Go2 evidence when available. Local
   regression passes do not remove any final-goal requirement.

No new fusion, calibrated uncertainty or learned turning controller is claimed
implemented here. The complete scientific objective remains active and unachieved.

## Exact identities

Output: `.generated/go2_depth_floor_hold_navigation_development_v1_attempt_001`.

- launch.json: `af7c8a2ff20e6fe68adeca5939b5f3efa1da643bd87822a55af1245fa0fe92d1`
- result.json: `9dff970f0f725efca2677d6c4422b124e83f11fcb12b2c88c86a65749c9adea7`
- raw_artifact_audit.json: `8e816c9af34aff14f47f1442a6a4c4bedfd29685bca62d8c61857c6f11722f17`
