# Commitment-pose waypoint native result — 2026-09-08

The changed utility produced forward and arc commands, but reached zero of four
mission goals. All four cases completed raw audits with no physical contact or
strict/hard depth failure. The result exposes remaining map and surface-filter
limitations; it does not establish successful navigation or general JEPA benefit.

The [protocol](go2_commitment_pose_goal_probe_v1_2026-09-08.md) tested the existing
final RGB direct and RGB JEPA fits on both known layouts. Only waypoint ranking
changed: predicted distance and bearing potential reduction minus contact cost
over the next half-second commitment. Scanning, all six complete model forecasts,
surface vetoes, observation pipeline, 1.2-m mission goal, 240-tick budget, native
guards and native goal criteria were retained. This changes both waypoint value
and scoring horizon, not an isolated heading term. Models were not trained or
updated. The development-based choice of the two methods was declared.

| Model / layout | Terminal outcome | Terminal / minimum goal distance (m) | Native terminal displacement in initial body XY (m) |
|---|---|---:|---|
| RGB direct / 039 | Tick budget | 1.230267 / 1.142393 | (0.134145, -0.614418) |
| RGB JEPA / 039 | Tick budget | 1.185256 / 1.184504 | (0.014746, 0.001969) |
| RGB direct / 052 | Visual failure, tick 123 | 1.055304 / 1.053420 | (0.179187, 0.267596) |
| RGB JEPA / 052 | Visual failure, tick 123 | 1.055304 / 1.053420 | (0.179187, 0.267596) |

Both 052 cases selected forward at tick 23, then left arc at ticks 28 and 33.
They had the same recorded action sequence and terminal displacement. At tick 38
their floor route disappeared, so they returned to scanning. At tick 123 neither
retained references nor the previous frame supplied enough rigid-pose matches.
Both completed ten zero-command drain intervals. The largest accepted-pose XY
error was 4.089 mm. There is no pose or calibrated uncertainty claim after failure.

RGB direct 039 entered waypoint mode at tick 33, briefly scanned again at tick
43, and returned to waypoint mode at tick 48. It selected four forward plans,
two right arcs, one left arc, seven in-place turns and 34 holds overall. Its
terminal displacement was about 0.63 m, largely lateral around the obstruction;
its terminal Euclidean goal distance was larger than the initial 1.2 m. Movement
along this unfinished detour is not goal success. At tick 238 forward still had
the highest utility (0.074019 m), but its surface check reported intersections
for the front feet and a lower-leg primitive; hold was the only unvetoed action.
The recorded first-hit voxels are low-lying, with old observation witnesses.
Their surface type must be established before proposing any ground-support
classification. No foot, calf or other surface veto has been waived.

RGB JEPA 039 never entered waypoint mode and retained its previous scan/hold
stall. Its complete requested-command tape, RGB sequence, physical trace and
public body/control/gyro arrays reproduced the serial predecessor exactly,
despite concurrent execution. For each other case, all those recorded inputs,
observer/map evidence and model predictions matched the predecessor through
the observation before the first changed command: tick 23 for both 052 cases
and tick 33 for RGB direct 039. The [readout](go2_commitment_pose_goal_readout_v1_2026-09-08.md)
retains these explicit common-prefix identities. It does not infer counterfactual
outcomes after the command divergence.

Eight focused tests passed in 1.84 s before launch. All four actual runs passed
complete sensor-to-observer-to-map-to-model-to-command replay with unchanged
model states. There were 776 camera frames and 41,600 physics samples; all 776
strict/hard depth checks passed. Every one of 772 complete iterations exceeded
100 ms. Median iteration times were 331.178–347.826 ms, with maximum 712.490 ms.
Four fresh workers ran concurrently, supported by the earlier authenticated
native scaling benchmark. These timings reflect concurrent load; they are not
uncontended timing improvements over the serial cohort. Physics paused during
compute, and no real-time or hardware qualification is claimed.

Preflight found 16 physical/32 logical CPUs, 0.2% CPU use, 82,128,080,896 RAM bytes
available, idle GPUs and 90,916,556,800 artifact bytes free. Resources were
monitored throughout. The four-case cohort and audits took 209.142 s after launch.
All workers are terminal. No case was resumed, replaced or omitted.

Artifact roots under the existing development base:
`go2_commitment_pose_goal_probe_v1_attempt_001` (936 frozen sources; 3,234 bound
artifacts totaling 1,347,493,320 bytes), and
`go2_commitment_pose_goal_readout_v1_attempt_001` (938 frozen sources).

| Identity | SHA-256 |
|---|---|
| Native launch | `4f1e6ce84d5ca60dc0f4ab7e9a6d8266fe6051e74709f923ed1ed34874ad5394` |
| Native result | `485c7d2081cacc2a5176d72bc4d52a34fd795ea0f4c4bc8a73cb0a29091afd6e` |
| Readout launch | `01c4170c5851f896121eada42d997c0aa9370624dac36bb0d70b100b1870ea76` |
| Readout result | `e3626874caca2ace86e3f9c60ade48e720927fe3649ffbce32900ce8cd3d31f6` |

The subsequent [route-loss reconstruction](go2_commitment_pose_route_loss_result_2026-09-08.md)
identified blocked start cells in nominal inflation on 052 and low-lying
foot/lower-leg surface vetoes on 039. Preserve the distinction
between nominal map inflation, possible voxel intersections, measured ground
support and actual native collision. New observer or filter behavior requires
a separately declared source and prospective test. Independent-maze goals,
physical backtracking, reactive/non-predictive and memory baselines, realistic
sensing/timing and bounded real-platform evidence remain unfulfilled.
