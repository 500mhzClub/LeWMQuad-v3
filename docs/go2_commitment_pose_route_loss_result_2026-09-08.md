# Route and surface-veto reconstruction result — 2026-09-08

The two RGB-direct trajectories stopped making useful progress for different
reasons. Layout 052 became trapped by the nominal inflated-grid connector test;
layout 039 retained a floor route but useful actions were vetoed by low-lying
foot/lower-leg voxel intersections. These are reconstructed controller/map
conditions, not proof of actual collision or permission to remove a veto.

The [V2 diagnostic](go2_commitment_pose_route_loss_v2_2026-09-08.md) accounted for
388 actual rows and exactly reconstructed all 367 available map receipts from
the original public packets and already raw-audited visual evidence. The other
21 rows retained unavailable/failure/drain accounting. Every recorded floor
proposal and every candidate surface check at the 72 prediction contexts matched
exactly. No model, observer, command, map rule or native outcome changed.

On 052, the first post-movement route loss was tick 38. Measured map position
was (0.107899, 0.181206) m and the closed start cell was (2, 3). That cell was in
the nominal 0.45-m obstacle inflation. There were still 346 eligible observed-
floor entry candidates within 1.25 m, but every closed connector necessarily
included the blocked start cell. The same condition held at every remaining
prediction context through tick 118, as the start moved through cells (2, 4),
(3, 4) and (3, 5). Entry candidates increased to 454. More observations therefore
did not remove the connector obstruction in this monotone map. There were no
current-posture or candidate articulated-surface hits in any of its 24 contexts.
The later visual failure is not the initial explanation for losing the route.

This is a conflict in a cell-inflated nominal disk approximation. It does not
prove that the actual robot or a continuous 0.45-m disk at its exact position
intersects an obstacle. A useful next geometric check is exact continuous
segment-to-observed-cell clearance at the same radius, before considering any
change to the radius, observations or articulated-surface filter.

On 039, only tick 43 had a blocked start cell, (1, -3); the route returned by
tick 48. The later holding state retained a 37-cell route with 923 reachable
floor cells and an unknown initial connector. At tick 238 forward had utility
0.074019 m, but its check hit both front feet and `FR_calflower1:0`; hold was
the only candidate without a reported intersection. The current measured
posture itself had no surface hit in the inspected tick-78/tick-238 states.

Across this trajectory's 48 contexts, candidate checks reported 213 conflicting
shape records: 76 `FL_foot:0`, 114 `FR_foot:0` and 23 `FR_calflower1:0`. The
measured floor plane crossed the first-hit voxel's transformed height interval
in 126 of those records. This counts first-hit shape records, not every hit
voxel and not distinct terrain patches. Several other intervals ended only
millimetres below the estimated plane. Height overlap alone cannot establish
ground identity, approve foot support, or waive a calf conflict. The next terrain
check should classify the actual original sampled returns and examine primitive
geometry instead of treating a voxel's entire volume as an identified obstacle.

V1 is preserved as a failed decoding diagnostic. It stopped before producing map
results because JSON converted the identity tuple to a list. V2 adds an explicit
adapter that restores only that field after checking exact identity agreement
with the current public packet; all numerical values and strict pose gates are
unchanged. Six focused adapter tests passed in 0.13 s, including unchanged JSON
roundtrip and rejection of mismatched/Boolean/floating/short identities. Both
eight-frame preflight replays passed before V2 output creation.

The one-/two-thread benchmark took 1.536401/1.217933 s on the same first eight
frames of both cases, with exact result equality. Two threads were selected.
Post-launch work took 46.602 s. Hardware and storage were checked before and
after. All source, input, URDF, readout and failure-witness bindings were
reauthenticated. There was no simulator, training or observer-estimation run.

Roots are under the existing development artifact base. V1:
`go2_commitment_pose_route_loss_v1_attempt_001`; V2:
`go2_commitment_pose_route_loss_v2_attempt_001` (944 source bindings).

| Artifact | SHA-256 |
|---|---|
| V1 launch | `6df147aafb8fa5f2d20f75e286a41bebeb2afe3fc2b2242a7c9f67afcc60db04` |
| V1 terminal failure | `427d8c17bdc982aa8124c9082a3a6c629ec6cb01566432fc03d831a3de432e4d` |
| V2 launch | `8affd00d27753d067ec162b6b7486a800ddec4d5b56a588ec27e7dd04c703bbc` |
| V2 result | `a33b120bdd8b210ab54da3293e903ab07fdcbc006fa388a197127c5298ce1b84` |
| V2 052 reconstruction | `4464c27e4675f913f0c5f726ada0c70128f6226ccdf5215b2a005c202efdb2b6` |
| V2 039 reconstruction | `55a6c74fadf7b42e869f5722d9026b654b6bf955729c00259a5816b9c70b648c` |

The full navigation goal remains active and unfulfilled. These diagnoses do not
establish goal-reaching, independent-maze generalization, physical backtracking,
terrain-support qualification, realistic timing or real-platform readiness.
