# Fresh Go2 execution V1: result and next experiment

5 September 2026. The eight-case study completed once, without reruns, under the
[pre-collection specification](go2_contact_attributed_execution_development_v1_2026-09-05.md).
Raw-force/trajectory recomputation passed for every case. Two cases meet all
predeclared arrival checks; seven cross without disallowed contact. This is
bounded oracle-guided simulation evidence, not RGB navigation or a JEPA result.

## Observed outcomes

| Approach | Width (m) | Correct contact-free crossing | Usable arrival | Limitation |
|---|---:|---|---|---|
| Straight | 0.75 | Yes | Yes | None under declared checks |
| Straight | 1.00 | Yes | Yes | None under declared checks |
| Offset | 0.75 | Yes | No | Speed 0.10139 m/s versus 0.10 target |
| Offset | 1.00 | Yes | No | Same speed miss |
| Left 90° | 0.75 | No | No | Front-right calf contacts opening corner |
| Left 90° | 1.00 | Yes | No | Heading 0.507 rad; speed 0.151 m/s; angular speed 1.121 rad/s |
| Right 90° | 0.75 | Yes | No | Speed 0.10226 m/s; angular speed 0.291 rad/s |
| Right 90° | 1.00 | Yes | No | Same speed/angular-speed misses |

Arrival thresholds remain unchanged: heading <=0.35 rad, planar speed <=0.10 m/s,
absolute world-z angular speed <=0.25 rad/s, plus the other specified checks.
The small offset/right speed misses must not be exaggerated into catastrophic
failure, or rounded into passes. An instantaneous arrival proxy is not itself
proof that the next maneuver is impossible; consecutive-edge execution is still
needed. Similarly, passing one endpoint is not a guarantee of continuation.

## Measured collision and RGB/body distinction

The narrow left case stops at recorded simulation time 6.704 s (including 1.5 s
settling). Its `FR_calf` contacts `corridor_low` and `front_low` at the opening
corner, with approximately 76.41 and 76.35 N per-contact force magnitude. Those
are simultaneous contact points against adjoining wall boxes, not independent
collisions or an aggregate calibrated impact severity. All native vectors and
object/link identities are retained.

At that instant the selected opening plane is world y=0.600 m. The base is at
y=0.432 m, the rigid-mounted camera at y=0.745 m, and the first contact point at
y=0.615 m. The final RGB already looks down the corridor while the calf contacts
its corner and the base has not crossed. Camera progress and whole-body
clearance are therefore different quantities in this actual simulated example.

This motivates predicting or controlling swept body/leg clearance using motion
history and body state. It does **not** establish that a JEPA is necessary:
compare a direct sensor-history policy and a simpler kinematic/physical baseline
under the same inputs before claiming predictive-model benefit.

## Evidence and limits

- Runtime: existing Go2 gait, Genesis 0.4.6 CPU, one environment, fixed camera
  extrinsics, 2 ms physics. No trained navigation model or JEPA was loaded.
- Every case retains requested/applied commands, joint state, privileged base
  state, phase, raw contact packets, attribution, native RGB and camera poses.
- The raw audit independently reindexes each contact and recomputes force
  magnitudes with a scalar reference, then checks contact labels, timing,
  command limits/slew, crossing, arrival, image hashes and optical-frame validity.
- The source/synthetic suite passes 222 tests. The separate native primitive
  contact assay still supplies six passing measurement checks.
- Some width pairs have identical non-contact trajectories. These are correlated
  deterministic development cases, not eight independent maze replicates. No
  population success rate or confidence interval is inferred.
- Geometry, starts and stopping differ from the interrupted generator. Do not
  compare their raw success/contact percentages as a detector improvement study.
- Genesis reports existing asset warnings about hip/thigh centres of mass and
  neutral joint limits. This is the model used by the existing gait; physical
  fidelity and real-platform transfer remain unqualified.

Artifacts: `.generated/go2_contact_attributed_execution_development_v1_attempt_001/`.
The result SHA-256 is
`2ea3228a6e523a2752617d7d9fca22443a0fb7eb04e1523e5e260b9f13c19b14`.
The raw audit SHA-256 is
`aeaa7b850d23433d7f07ea34ab3b12e33baa8516ed4c780f6e82f80aca819ffe`.
The old qualification material remains preserved and must not be resumed.

## Next scientific step

1. Test pre-turn alignment as a specific way to reduce the calf's lateral sweep
   before entering the narrow opening. Keep the original teacher as a matched
   baseline; do not change contact labels or remove difficult starts.
2. Separately test feedback-based arrival/settling against the fixed 0.5 s brake.
   Preserve the declared motion thresholds for comparison and report timing and
   gait-phase sensitivity; do not select a favorable endpoint retrospectively.
3. Prefer a small factorial comparison of those two mechanisms, with parameters,
   task-time budgets and stopping rules fixed before collection. Use fresh
   experiment identities; reusing these development geometries is not a novel-
   maze generalization test. No old runtime snapshot is required.
4. Execute a second directed edge from the actual first-edge terminal state,
   without teleportation or idealized reset. This tests whether the arrival
   proxies predict genuine continuation and closes a gap in the current evidence.
5. Once local execution is reliable, integrate observed-place routing and causal
   sensor packets, then compare direct history-conditioned selection with JEPA
   predictive training and online rollout as separate factors. Record RGB over
   decision sequences for that future experiment; current endpoint images alone
   are not a sufficient JEPA training dataset.

The final novel-maze and real-platform objective remains open.
