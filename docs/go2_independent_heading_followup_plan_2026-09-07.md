# Independent heading evidence: conditional follow-up, not an estimator replacement

This source-and-result-document review does not launch an experiment, inspect
new runtime payloads, or change any frozen source. Finish the original twelve
layout collection and the frozen four-worker learning comparison first. The
already frozen independent tracking challenge remains unchanged. This note
identifies a subsequent experiment if its results warrant one; it is not an
extra prerequisite for either frozen run.

## Existing evidence and the actual gap

The current multi-reference candidate explicitly calls `register(mode='gyro')`
in `lewm/multi_reference_rgbd_pose_development.py`. Both retained-anchor and
previous-frame measurements in `lewm/temporal_anchor_continuity_development.py`
inherit that rule. Their rotational agreement therefore cannot establish
independent visual heading evidence. In exact arithmetic, with identity initial
orientation, induction gives

    R_current = R_reference (G_reference^T G_current) = G_current.

Both paths share the same integrated gyro history. Floating-point composition
defects are not an independent measurement. Translation still uses visual/depth
correspondences, so this limitation does not negate measured translation
bridging. Neither positional nor rotational agreement calibrates absolute error.

An independent visual rotation solver ALREADY EXISTS in
`lewm/joint_rgbd_rigid_pose_development.py`: joint mode fits a proper rigid
transform to paired depth-lifted image features. Gyro is not fixed in that fit,
but its disagreement still gates acceptance. Reusing this solver is preferable
to implementing another equivalent registration method. It does not estimate
gyro bias, and disagreement with gyro can still stop the observer.

The [earlier joint result](go2_joint_rgbd_rigid_pose_result_2026-09-06.md)
reports unchanged joint predictions under the admitted +/-0.001 rad/s gyro
perturbations, with nominal maximum position error 6.904 mm versus 5.340 mm for
the gyro baseline. This is conditional invariance by construction, not evidence
of bias identification or correction. The
[longer frozen transfer](go2_longer_motion_frozen_pose_result_2026-09-06.md)
tracked all 586 frames of each of two nominal tapes in one geometry. Joint
maximum position errors were 23.162/20.861 mm versus 16.654/17.381 mm for gyro;
orientation errors were also larger. Gyro coverage-query rejections arose from
a documented numerical interface mismatch, not superior joint perception.

The [continuity result](go2_temporal_anchor_continuity_result_2026-09-07.md)
recovers three one-frame interruptions on old stopped tapes. It does not test
the joint solver combined with continuity, continued physical recovery, or a
calibrated sensor-error distribution. Its extra registration also costs time.
These observations support a matched follow-up, not immediate adoption.

## Smallest informative subsequent experiment

1. After the fixed tracking readout, distinguish actual motion-coverage failure,
   correspondence failure, common-gyro drift, and control/dynamics failure.
   Do not treat a tracking-only change as a fix for lower-friction overshoot.
   If the needed motion was not executed, the unobserved regime remains untested.
2. In a separately named future source variant, compare the current gyro
   continuity estimator against joint-RGB-D continuity using the existing rigid
   solver. Keep feature extraction, reference retention, temporal bridge budget,
   correspondence gates, clocks and terminal-failure semantics matched. Mode
   must propagate to both retained-reference and previous-frame estimates;
   switching only an outer label or constructor would leave hard-coded gyro
   fitting unchanged. No in-place modification of frozen candidates or outputs.
3. Before reading any reserved outcomes, specify the complete paired stream
   population, signed bias conditions, missing-current-image conditions, and
   nominal-error/availability trade-off used for adoption. Use fitting data to
   set any new criterion, then acquire distinct validation observations. Reuse
   of exposed challenge tapes is development, never a fresh validation cohort.
   Synthetic gyro disturbances are stress tests, not hardware calibration.
4. Record both independent relative rotations and both composed rotations,
   reference/frame/timestamp identities, inlier support, disagreement and failure
   reasons. The current continuity witness stores incremental position but not
   incremental rotation. Add that evidence only to the new variant. Distinguish
   acceptance-gate invariance, detected disagreement, pose robustness and actual
   bias estimation; this comparison implements only the first three.
5. Report paired availability, rotation and translation errors, bridge/rejoin
   outcomes and full-loop latency for every stream, including rejected ones.
   Do not rank methods only on their different surviving subsets. Report shared
   support errors and each method's missing population separately. Correlated
   frames and stress variants are not independent scene replicates.
6. Adopt only if the predeclared trade-off is met. A negative result should end
   this specific candidate, not trigger relaxed gates or favorable tape selection.
   A positive replay result must be followed by fresh continued closed-loop
   turns, branch departure and physical backtracking with unchanged task goals.

## Relationship to the JEPA navigation goal

Independent heading is a deployment-relevant state-estimation prerequisite,
not a JEPA contribution or a learned high-level policy. Preserve separate
comparisons of predictive training, online rollout and persistent memory using
the same observation/execution stack and independent layout units. Report
actual branch recovery and home return, not just local pose or latent loss.
Hardware calibration and timed full-loop operation remain separate requirements.

Decision now: do not replace the frozen gyro estimator, duplicate its existing
joint solver, or delay the queued learning study. Use the completed challenge
to decide whether this specific follow-up is worth executing.
