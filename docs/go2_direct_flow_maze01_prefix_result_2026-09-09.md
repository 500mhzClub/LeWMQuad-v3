# Direct-flow full-controller maze1 prefix: V2 complete

The corrected full-controller replay passed through the original tracking
failure at observation214. It is a completed prospective replay, not native
tracking recovery or navigation. No observation following the new command was
consumed, and no new physical trajectory or arrival is claimed.

V2 session74784 exited0. Result SHA-256
`345b54f1b3c647be516040f2b8bf03b4ab3c709b9737dc0bdf7c78bedcacdbe3`;
launch `9ec86d78eeb4ef9fa46e2d1031b8b1294cd665b9e413141e9b36cb54a4cf6151`;
decision stream `13b4b1017957396bfc7a1b1a8c6cb43fe4a2e65afe8ed680aec1489aea1fcfea`.
Root: `go2_direct_flow_maze01_prefix_v2_attempt_001` in the navigation
development artifact volume. Wall time after launch349.3820586500224s.
1,673 sources and6,201 cohort artifact bindings, upstream evidence and all
three preserved V1 failure artifacts were verified before/after execution.

The215paired observations cover0–214. All214complete original decisions and
prior physically completed requests were exact. All211preceding raw forecast
banks were exact, including their scoring/veto selections and observed,
mapping, mission and residual receipts. At214 the original visual failure
and zero command became an accepted current auxiliary-camera anchor pose,
reference213, and left-turn request[0,0,0.45]. The controller advanced to214
with no failure or terminal. The fallback preserved the original failed
camera, continuity and reference receipts, changed only correspondence
association, and passed existing rigid/gyro/temporal/bridge limits. Both live
raw-pose and registered-floor contracts passed; the complete learned selection
was produced. No original qualified witness existed at this failed boundary.

The assigned model remained
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`,
with no gradients. Public input arrays were unchanged. Final artifact and
source verification completed. The separate V1 validation failure remains
terminal and is described in
`docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md`.
An additional read-only comparison found all215V2candidate decisions exactly
equal to the saved V1candidates, including214. The correction changed live
versus serialized validation, not the scientific controller behavior.

Final hardware:80,414,588,928availableRAM bytes,88,418,832,384artifactfree bytes,
CPU3.3%,bothGPUs0%,all32logical CPUs in affinity. The reactive cohort parent
was still finalizing then; no native scene was launched by this replay.

## Fresh-native comparison helper prepared

Added `scripts/direct_flow_maze01_native_prefix_development.py`, SHA-256
`f8aeb87fd380fd20dcc72464233ce811f0e7c08a1861ffc4895f7152c1871426`, and
`lewm/tests/test_direct_flow_maze01_native_prefix_development.py`, SHA-256
`213f8792a55c8b03adeaf57fa9a0ca67acc9c2020d88968b1708cb2f23e220cd`.
22tests passed in1.88s. The helper also admitted the actual completed V2 report
and all215saved decisions after authenticating result/launch/stream bindings.

A fresh native tracking trial must reproduce11,450physics samples through
the214endpoint,215paired public observations, all214prior commands and all215
complete prospective decisions. It must retain the original failed-boundary
evidence and compare211preceding raw forecast banks. Tests reject altered
physical/public histories, prior commands, original failure evidence and new
decisions, truncated/extra prefixes, failed V1 admission and unsupported recovery
claims. They exclude physical samples after11449 and allow the changed command
to complete only partially if a new physical stop occurs.

The native collector/audit/launcher for this tracking successor are not yet
prepared or executed. Implement them separately, preserving the original
physical and evaluation calculations and frozen controller. Keep the queued
planning-memory native pilot and residual maze2 native trial ahead of this
new tracking execution unless an explicit later goal decision changes order.
