# Measured-plane chained-anchor controller prefix V1

Prospective development controller replay protocol. This source preparation
does not launch a scene, change a live experiment, or establish navigation.

Use the exact completed measured-plane dispatch-recovery episode (launch
`93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`),
its 3,124-observation negative collection
`922cf4a1134eb5458010e12f89ba65fe427b839476837ab035d8ceace4181680`,
and its eventual actual parent-result hash after the entire raw worker audit
and parent have ended. Scientific failure is retained and admissible. An
execution/audit failure is not bypassed.

Also require the completed fixed return-anchor pair probe, result
`d3a01cc5b667b7038964adf04ee135179af8a35c7d8bf0ed875b5baae5831c6e`,
and the ended full-history timing waiter with launch
`c63f9bccb0cdfc273b1d3fd721310da5fac384f1350dcfc4625098fd103184a4`.
The exact timing waiter and child must have completed and ended before this
full paired CPU replay starts. Timing improvement is not a prerequisite.
Their completed source/artifact receipts establish execution ordering; their
controller science is not rerun by that ordering check.

The two arms are the actual `MeasuredPlaneResidualController` and the separately
implemented `MeasuredPlaneChainedAnchorController`, with independent fresh copies
of the same corrected no-RGB direct model
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
Each starts with fresh perception, floor/map/contact, mission, residual and
model history. Keep the original geometry, public mission, 4,000-tick budget,
model inputs, 800 ms forecasts, 100 ms requested-command interval and all
inherited physical/sensor gates. This is a perception intervention, not a
performance optimization or a new world-model training attempt.

For each consumed original observation:

1. Check the original requested command and physical sample endpoints, including
   any genuine final partial command or observation without a following command.
2. Reconstruct the original public RGB/depth/gyro/auxiliary packets. Run both
   controllers on those same packets and verify that neither modifies them.
3. Reproduce the complete original recorded decision. When both controllers
   publish a forecast, require the same raw prediction, head, input variant,
   forecast offsets and training-only correction fields. Count actual forwards
   using hooks. A terminal failure after inference may legitimately retain a
   forward count without publishing a selection; it remains a negative result.
4. Save the complete original and candidate decisions, packet identity and
   comparison. Stop immediately at the first changed requested command or
   terminal status, any matched terminal, or the exact original history end.
   Never consume a following public observation after that boundary.

Check both final model states and absent gradients, close the complete output,
then reconstruct every consumed packet, original decision, comparison and the
aggregate/boundary report. Reauthenticate the complete original raw artifact
roster and immutable source union before completion. A separately replayed
candidate-observer proof is not claimed: candidate perception is executed
inside the complete controller path.

Use one deterministic CPU process with one OpenCV/BLAS/Torch thread. Admission
requires 64 GiB available RAM and 43 GiB artifact free space. Recheck an 8 GiB
RAM and 41 GiB disk floor every 100 frames and at the comparison boundary.
Output is bounded to 2 GiB. The output root is exclusively
`go2_measured_plane_chained_controller_prefix_v1_attempt_001` under the owned
development artifact root. Preserve any failed attempt; no automatic retry or
resume. Native scene execution and actual outcomes of changed commands require
a separate prospective experiment.
