# Measured-region navigation V1: audited negative whole-task result

The fixed two-layout development collection completed (session29096, exit0).
Full audit15476 passed both trials, replaying500 controller decisions and511
RGB/depth/relative-state records. Audit PASS means faithful acquisition,
controller replay and physical scoring, not navigation success. Tests before
launch passed1,106 across99 files; all282 source,198 input,2 gait and10 native
bindings were checked again by the terminal audit. No frozen source or result
was changed or retried.

| Outcome | North dogleg | South branch |
| --- | --- | --- |
| Whole mission | Failed | Failed |
| Terminal | FAILED_ALIGNMENT | FAILED_SENSOR |
| Elapsed, including release | 36.4 s | 14.5 s |
| Final distance from start | 1.505909 m | 1.500151 m |
| Path length | 1.768009 m | 1.540737 m |
| Local arrivals | One unqualified candidate | None |
| Marker discovery / home return | Neither | Neither |
| Recorded contact / native stop | Neither | Neither |
| Zero-command release | Passed | Passed |

## What changed scientifically

North reached a measured arrival at decision tick150 (16.5-s simulation clock):
0.05913 m short of the measured target, all940 nominal volume samples supported,
zero near-surface conflicts,32 retained views. It then acquired a scan and selected
a branch at tick223. This is progress beyond the old global-image-change local
failure, but is not a recognized place, continuous-volume safety proof, future-gait
qualification, discovery or return. Trusted graph edges remain zero.

North's heading alignment ran from tick239 to359 under the unchanged12-s budget.
Its terminal error was0.02201446 rad, just outside the0.02-rad acceptance condition.
The last active command was0.03300758 rad/s, whereas measured projected heading
rate was-0.00012066 rad/s. This is consistent with inadequate low-command response
under proportional feedback; it does not independently identify a hard actuator
deadband. Do not relax acceptance or relabel the timeout as success.

South failed during TRAVERSE, not SCAN. At ticks137–139 remaining forward distance
was0.07522,0.07127,0.06732 m; yaw requests were-0.10772,-0.15077,-0.19525 rad/s.
The point-target bearing divides lateral error by shrinking forward distance,
making steering increasingly sensitive near arrival. At observation140 the
point-to-plane normal spectrum became[0.00313585,0.36488390,0.63198025], below
the predeclared relative rank threshold in one direction. Registration converged
with0.08562-mm residual but reported only rank2; the missing direction was mostly
lateral in the previous body frame. The estimator invalidated cumulative position,
and the controller stopped rather than filling the missing component from commands.
The causal role of changed approach guidance must be tested in a fresh run; replay
cannot establish that a different command would have retained observability.

## Remaining sensor limitations

North has full estimates on364/364 intervals, maximum step error1.00637 mm and
final position error0.46959 mm against evaluation-only physics. South has139/145
full intervals and no final position estimate; its declared moving-state check
fails even though the observed-subspace residuals are small. Small residuals
do not supply the unobserved component.

Moving-depth checks pass361/365 north frames and146/146 south frames. North
failures at indices250,270,273,329 have maximum errors41.971,21.433,41.825,16.236 mm.
Read-only diagnostic cc0cfb found exactly one failing sampled ray per frame,
at(row,column)=(268,124),(316,4),(276,132),(356,420). Each analytic reference
selects the visual floor; the native ray terminates at a wall-boundary coordinate
only0.60–7.63 micrometres below z=0. Nearest retained motion-cloud points are
60.39,75.88,60.73,22.64 mm away respectively, so the erroneous samples were not
used as motion constraints. These observations support a numerical lower-wall-edge
ambiguity, but do not establish the exact rasterization mechanism or qualify
other rays. The original four FAIL outcomes remain unchanged.
No calibrated depth robustness, hardware safety or real-platform validity follows.

## Next controlled intervention

Implement a distinct line-guidance plus bounded-integral alignment controller:
fixed spatial lookahead avoids endpoint bearing sensitivity; bounded integral
feedback addresses persistent heading error without changing tolerance or deadline.
Keep the measured targets, braking/arrival, rank test, missing-state stop, sampled
clearance, marker, memory and independent mission metrics fixed. Run the original
two full missions under a new named protocol and fresh output; preserve all failures.
If translation becomes unobservable again, add genuinely independent deployment-
valid motion evidence with explicit uncertainty, rather than lower the rank
threshold or assume commanded pure rotation implies zero translation.

Whole-maze success, memory benefit, JEPA predictive-training benefit, genuine
multi-step online planning, independent-layout/seed robustness and physical Go2
evidence remain unachieved. The ultimate goal remains active.

## Exact identities

Output: `.generated/go2_measured_region_navigation_development_v1_attempt_001`.

- launch.json: `f28f476f853c2ee9f42b6762e4749fef1657991b9c99e2aee2a65474fe273503`
- result.json: `a7f9839823d3772d0245559b80c0f5c9d9eac03650458f551526aae8ca1eb0c0`
- raw_artifact_audit.json: `38d2526fc5af870fc750d049674aff83ed55def4dca08e73d0788be0762d7e20`
