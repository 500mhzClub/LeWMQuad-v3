# Observable-approach/heading-hold V1: audited integration failure

Collector23053 completed both fixed missions; full audit74202 passed390 decisions
and402 RGB/depth/state records, exit0. Prelaunch56153 passed1,147 tests across106
files in77.90 s; preflight14752 and terminal audit checked308 source,207 input,
2 gait and10 native bindings. Post-audit checksum verification passed. No frozen
source, packet, command history or result was changed.

Both missions stopped at decision195, the first scan observation after the first
local arrival/hold, with FAILED_SENSOR. Both elapsed20.0 s including terminal
zero release. Neither discovered the marker, returned home, recorded contact or
native body stop, or made a false home claim. North path1.567829 m/home distance
1.526937 m; south path1.565022 m/home distance1.525825 m. Whole-task success0/2.

The error was not missing motion information. All400 motion intervals have full
estimates and all402 depth checks pass. Maximum step errors0.59166/0.59436 mm and
final errors0.15638/0.12411 mm (north/south) are evaluated separately from control.

## Exact integration defect

Read-only actual-packet controller replay7678 reproduced the nested exception:
`CausalGroundPlane.begin` rejects the scan's history because it requires every
applied command to be zero within1e-8. The new active hold correctly logs small
nonzero yaw corrections. The last north request before scan is-0.00052265 rad/s;
the actual history contains magnitudes up to0.00072181. Freshly bootstrapping a
zero-command gravity hypothesis is incompatible with that operational holding
contract. The same latent issue would affect a later traversal's fresh ground
observer. Do not zero the history, discard its validity checks, or call it
measurement noise to bypass the contract.

The new blocker/view cap has static old-input evidence (old south target4.134 m
limited to2.235 m before measured wall3.545 m), but this physical attempt never
reached that second traversal. It therefore does not establish the cap's physical
benefit or whether active alignment handoff works. The earlier synthetic
positive-bias handoff failure and stricter readiness change remain recorded.

## Implemented next integration component, not yet a new physical result

`ObservedDepthFloor` and `DepthFloorHoldNavigation` now use the current supported
depth floor plane for scan/traversal floor projection. They do not pretend to
have a zero command history or substitute foot-contact height for an observation.
The initial whole-mission gravity initialization remains unchanged. Fresh
adapters are installed before the scan/local operator's first observation;
current clocks, plane support, normal, height and failure checks remain explicit.
The measured visual plane is not a collision-ground or hardware certificate.

Focused81500 passed6 tests in1.39 s, including real nonzero holding history,
unchanged input bytes, missing/stale/invalid floor and fresh installation.
Read-only45573 on the actual failed north frame195 accepts1,263 floor points,
measured plane height0.316001 m, preserves the nonzero command history and
produces an unqualified forward floor-extension proposal with987 support points.
That is static input replay, not a rescued or counterfactually successful run.

Next freeze a distinct depth-floor-hold protocol/source and execute both full
missions with a full audit. Keep observed blocker/view limits, missing-motion
stops, nominal turn checks, genuine holding commands, native contacts and complete
discovery/return metrics. No completed attempt may be retried or rescored.
Independent-layout/seed robustness, matched memory/supervised/JEPA/multistep
comparisons and bounded real Go2 evidence remain required. Goal unachieved.

## Exact identities

Output: `.generated/go2_observable_hold_navigation_development_v1_attempt_001`.

- launch.json: `0efc8f897f14616627001ddadff0ba793e10580483f36a925aba6990ed7c2ea2`
- result.json: `ea443bb9ef29cde060fa7023b8d2794020ca3a42db9d48fe53b9e59415c8eb6c`
- raw_artifact_audit.json: `69575badc0209e66299f191ac2d82f92302b8748d942d19ab81841c857073ca0`
