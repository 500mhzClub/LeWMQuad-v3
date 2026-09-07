# Previous-measured-pose initialization does not restore recorded tracking

Do not adopt this candidate. The complete fixed comparison yields **473/7,433
available poses**, versus **7,400/7,433** for the original descriptor-based
observer. It loses6,927 previously available frames, recovers none, and fails
early in all six trajectories. This is a completed negative development result,
not an infrastructure failure or a navigation success. The latest physical
simulation room-return result remains0/3; no JEPA benefit is established.

## Completed evidence

- Regression87857 terminates exit0:3,339 tests across245 explicit files,
  292.87s. The saved JUnit report has0 errors/failures/skips, SHA-256
  `fc79834c10e487477fad41ee54cd5bb67e3b895971031e09f3c9c6ead79d50d1`.
  Earlier regression45052's terminal result was unavailable and is not counted
  as passed. No scientific experiment was repeated to recover that test evidence.
- Replay10196, PID2082389, terminates exit0 (d8313f); process absence subsequently
  confirmed (9825a2). The696-source launch and predecessor data/witnesses are
  unchanged. All six sensor streams are persisted before native evaluation.
- Independent verification53017 terminates exit0 (c0ddb1). It authenticates all
  source/predecessor raw/native/witness bindings and27 declared output artifacts;
  joins all7,433 sensor/evaluation/original-witness rows; checks original poses,
  selections and failures exactly; reconstructs native timestamps and pose
  errors; checks full availability, first failures and terminal latching; and
  recomputes every available/paired error and timing summary.
- The same verification checks475 candidate seed attempts against previously
  estimated poses and stored promoted references. Every seed has the immediately
  preceding100ms observation provenance. Numerical comparison tolerances are
  1e-12m and1e-10rad, not changed physical acceptance thresholds. Orientation is
  independently computed using quaternion composition/projection; the frozen
  observer uses accumulated near-orthogonal gyro matrices. Final output hashes
  verify again after analysis.

Output root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_pose_correspondence_replay_v1_attempt_001`.

Launch SHA-256:
`a82c4e75a459896c70ec3f3486c26c00fba321dab55f078bd27ba7f2e9d4a8b6`.
Complete sensor-phase SHA-256:
`0a084ea6379e41dc60ca0eb49139017678b3907b1dac973c1c4799a56766e7d6`.
Result SHA-256:
`1d44e2e72497f479536924d155358de4a806f50573c8f686c0ca02c326b0c7ea`.

## Whole-trajectory result

| Recorded stream | Frames | Original available | Measured-pose available / first failure | Frozen rotation-only available |
| --- | ---: | ---: | ---: | ---: |
| Inner left |1,138|1,127|62 /62|58|
| Inner right |1,014|1,003|98 /98|98|
| Inner low friction |576|576|76 /76|78|
| Intent left |2,413|2,413|58 /58|58|
| Intent right |1,820|1,809|98 /98|98|
| Intent low friction |472|472|81 /81|81|

There are473 both-available frames,6,927 original-only frames, zero candidate-only
frames and33 neither-available frames. The new initializer gains four frames on
one stream and loses two on another relative to rotation-only; the other four
first failures are unchanged. The frozen rotation-only run is not repeated.
These correlated reused development recordings do not establish independent
layout reliability or causal navigation improvement.

Paired early-frame mean position errors are smaller for the candidate in all
six streams, but cannot compensate for losing almost the entire useful tracking
population. On intent low friction its paired maximum is worse:1.359mm versus
0.966mm. Both methods retain the same measured gyro, so matching orientation is
not a new capability. Candidate available-frame observer medians are39.7–40.7ms;
its approximately0.04ms all-frame medians mostly measure latched no-ops. Neither
those numbers nor original medians establish full-loop real-time performance.

## Scientific interpretation

The synthetic multi-frame result—91/91 measured-pose frames versus59/91 for
rotation-only—did not transfer to these recorded scenes. Correct past-pose
initialization is insufficient to make the altered correspondence method
reliable. This rules out adopting the implemented change; it does not prove
that every measured-pose prior, optical-flow method or temporal estimator fails.

A sensor-only check of inner-left frames60–62 finds205 depth-valid tracks at
failure, after124/205 accepted inliers and8/8 grid coverage on frame61. The last
seed is approximately(0.09222,0.00039,-0.00282)m, from the previous estimate.
The exact rejected consensus component is not stored; do not attribute the
failure solely to fraction, grid or displacement from the composite message.
More tracks and a better initial guess do not prove correct correspondences.
Periodic appearance ambiguity and common-depth bias remain separate limitations.

## Next actions toward the actual goal

1. **End this initializer branch.** Retain both negative flow experiments and
   the original observer as the measured baseline. Do not extend this attempt,
   search new seeds, choose methods per trajectory or lower acceptance gates.
   Further local execution work must address an actual missing measurement or
   temporal-state capability, not just another initializer on the same images.
2. **Keep the independent learning experiment on its existing critical path.**
   The original collector is live on layout four; layouts0–2 each have120
   audited eligible departures. Preserve its sources and complete all12 receipts.
   Then run the already reviewed36-fit JEPA/supervised/input-ablation study,
   including the empirical action/time baseline. Do not fit a partial cohort or
   select a winner from the old one-room results.
3. **Before another navigation trial, design one distinct continuity experiment.**
   Separate short-baseline incremental measurements from longer-baseline anchor
   association, or use complementary measured kinematics only where its validity
   is established. First review existing contact-dropout negatives: low-friction
   support measurements already fail, so they are not an automatic fallback.
   Preserve an uninterrupted coordinate frame, explicit unknown state, drift
   accounting and stop behavior. Require sensor-only predictions and independent
   scene/error challenges; no command-integrated pose or silent reinitialization.
   This is a design obligation, not a newly qualified estimator or run authority.
4. **Test dynamics and navigation contributions separately.** Use the independent
   study to determine whether body history/RGB improve action-conditioned motion
   and contact prediction, particularly under changed support. Only a useful
   predictor should proceed to matched online rollout tests. Retain the existing
   memory event interfaces, then evaluate physically executed branch choice,
   backtracking and home association once local execution is dependable.

Independent maze/training-seed evidence, predictive-training versus online-rollout
versus memory attribution, realistic calibrated sensors and self-occlusion,
unpaused real-time operation and bounded hardware evidence remain unfinished.
No test count or completed replay substitutes for those scientific requirements.
