# Floor-registered native readout V1

Read one completed, hash-bound floor-registered pilot through
`scripts/read_go2_floor_registered_maze_pilot_v1.py --native-result-sha256 HASH`.
Exclusive output: `go2_floor_registered_maze_readout_v1_attempt_001`.
No new native execution, model fitting, controller selection or output overwrite.
Verify completed native artifact/source bindings before and after; require
8 GiB available RAM and 128 MiB headroom above the 40 GiB artifact reserve.

Retain the predecessor readout's actual traversals, arrival/return result,
strict visibility failures, recovery/waypoint execution, and timing summaries.
The auxiliary-floor classification readout remains explicitly scoped to checks
at the poses used by this controller. It is not an unexecuted original-controller
trajectory or attribution of all changes to that mask.

Additionally reconstruct each recorded floor-registration witness, restoring
only JSON's tuple identity representation. Validate the original joint-fit
witness and the registered composition independently. Compare both estimates
with the same actual native pose at observation sample 749 + 50 * frame.
Report XY, XYZ and rotation errors for each, plus the correction magnitudes.
Never feed this evaluator-only native state to the controller or recalibrate
its estimator from these outcomes. Accuracy on one executed trajectory is not
a calibrated uncertainty bound or a physical floor-identity certificate.

Terminal observations without admitted poses are counted explicitly, with no
imputed error. Missing native endpoints, nonterminal missing poses, chronology
gaps, malformed identities and altered witnesses fail the readout. Focused
tests include a synthetic case where native state favors the correction and
the opposite case where it favors the raw estimate, so an improvement is not
assumed by the metric.

Zero or failed arrivals, unsafe/invalid traversal, sensor failure and slow
timing remain failures. Independent layouts, matched comparisons and realistic
sensing/timing/hardware evidence are still required by the full thread goal.
