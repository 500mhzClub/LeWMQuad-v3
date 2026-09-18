# Completed measured-floor-transport maze0 readout

Session48066 closedexit0 with
`MEASURED_FLOOR_TRANSPORT_MAZE_READOUT_V1_COMPLETE`.
The unchanged native outcome is one settled outbound arrival, no home arrival
and zero verified round trips. This readout adds no navigation episode,
model update or independent-layout evidence.

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_floor_transport_maze_readout_v1_attempt_001`.

Result SHA-256:
`a46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c`.
Launch SHA-256:
`0b824ca244ac6fc2bd6164826e38b0a1489cc8a4501031ff98948b45e30d1c1c`.
Input native result SHA-256:
`1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374`.
All1,658 bound source paths and input artifacts were verified before and
after the readout by its existing launcher.

## Actual execution

The robot travelled10.304553639084356m in301.3 simulated seconds after its
initial observation. Final native initial-frame XY was
`[3.764321979721436,-0.6553267695495932]`, giving home distance
3.820938777565301m. Final distance from the outbound target was
0.6587959465920187m. Its minimum outbound-target distance during the run
was0.021496480659927094m; the separate native arrival-window audit remains
the evidence for settled arrival, rather than this minimum alone.

The final active controller decision at3002selectedhold in RETURN, targeting
an observed-floor frontier waypoint`[3.725,-0.275]`. It reported home distance
3.8184589785535925m with6,857retained floor cells and549occupied cells.
No controller failure or active infeasible wait was reported at this frame.
The subsequent budget terminal and drain commands remain recorded.

Selected-action counts are663left arcs,934right turns,264left turns,
1,099holds,26right arcs and one selection with no action. Counts do not
include warmup/other frames without a new selection and do not by themselves
measure useful movement. Source-backed provisional return-ranking diagnosis
is in`docs/go2_floor_transport_live_return_selection_diagnosis_2026-09-09.md`;
no alternative command's physical success is established by this readout.

## Pose and timing measurements

Of3,004 admitted poses,2,996 use current joint floor registration and eight
use measured visual floor transport, at1904through1911with anchor ages1–8.
On those eight executed observations, registered3D error ranges from
0.0026305504051638527m to0.003427959727538423m. Registration resumes after
that interval. Across admitted poses, registered3D error mean is
0.004262246141651316m and maximum0.008581740674091813m. These are accuracy
measurements on the executed trajectory, not calibrated uncertainty bounds
or guarantees for unexecuted poses.

| Timing measure | Count | Median (ms) | Maximum (ms) |
| --- | --- | --- | --- |
| Observation and controller | 3014 | 1182.855663 | 1996.171491 |
| Iteration including command | 3013 | 1215.364901 | 2028.152257 |
| Iteration including receipt | 3014 | 1237.033925 | 2050.692826 |
| Receipt write | 3014 | 21.5962285 | 32.298629 |

Every recorded observation/controller and full iteration exceeds100ms.
Physics remains paused during computation; no real-time or hardware
qualification is claimed. Original strict primary failure909and hard
measurement failures1924,1925,1930 are unchanged, as detailed in the native
result and saved raw audit reports. No learned-planning or memory advantage
has been established.

## Independent-layout admission

Preflight52560closedexit0 and verified1,658source paths, all required
completed inputs and the fixed case order1,2,3. It created no output or
native scene. Its final hardware refresh measured83,057,709,056bytes
availableRAM and95,268,552,704artifact-volume free bytes, against
34,359,738,368bytes RAM admission and78,383,153,152bytes required free space
for all three remaining cases. All32logical CPUs on16physical cores were
available, overall CPU activity0.3%, bothGPUs0%; only the small existing
background Python process remained. Workspace free space was21,360,185,344
bytes. Native concurrency remains one scene.

The fixed cohort command then started in session11941 using the exact native
result SHA above, without`--preflight-only`. It revalidates the source and
input bindings and refreshes resources before creating its exclusive output
and starting maze1. At launch-command return it was still running without
terminal output; no new maze outcome is yet available. Keep this same handle
and preserve all fixed cases and negative outcomes.
