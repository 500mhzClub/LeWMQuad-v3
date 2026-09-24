# Measured-floor-transport maze0: saved raw audit

The twelfth collected navigation episode now has a saved full raw audit.
At this inspection the original worker remains active on its prospective
prefix comparison and final binding checks. This document does not certify
the root attempt complete or admit a dependent experiment.

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_floor_transport_maze_pilot_v1_attempt_001`.

Saved `full_jepa_novel_maze_00_audit.json`: 10,754,355 bytes,
SHA-256 `449d23341e1449754eaf698757be2b2553cefdcd4345d77e372ac03cb66ea368`.
Collection result SHA-256 remains
`0153c24c6974fb5f8df736ca0a24874766b3ef93b552ad020af56242296bb8cc`.

## Verified audit findings

Raw sensor reconstruction, additional auxiliary RGB reconstruction, complete
model/controller-to-command replay, command audit and unchanged model state
all pass. Collection contains 3,014 paired observations/decisions, 3,013
completed commands and 151,400 physics samples. It exhausted the shared
3,000-navigation-tick budget and completed ten terminal zero commands;
physical and acquisition stops are null.

The independent native evaluator confirms the outbound arrival at frame1868.
Over its one-second settling window, maximum goal distance is
0.03761944072156676m and maximum full-3D speed is
0.03901577611147847m/s. This reproduces the same outbound arrival as the two
preceding maze0 runs; it is not an independent-layout success.

There is no confirmed home arrival, no physical route-retracing pass, no
terminal native quiet pass and no native round-trip candidate pass. Strict
verified round-trip success remains false. Outbound traversal has ten raw
crossings, no invalid crossings and seven loop-erased cells; return traversal
has three raw crossings, no invalid crossings and two loop-erased cells.
Crossing counts alone do not establish progress along the complete return
route or collision-clearance certification.

## Measurement failures retained

The first original strict primary-camera visibility failure remains frame909.
New hard measurement failures occur at frames1924,1925,1930. Each has one
primary stable-interior sampled ray exceeding the unchanged 1mm metric gate:

| Frame | Maximum primary stable-interior error (m) | Auxiliary visibility |
| --- | --- | --- |
| 1924 | 0.00100578011992436 | Pass |
| 1925 | 0.0010578653070654198 | Pass |
| 1930 | 0.0010515414946317136 | Pass |

At those three frames, primary near-occlusion failure is false, auxiliary raw
reconstruction is exact, auxiliary robot-pixel count is zero and auxiliary
stable-interior metric checks pass. These observations do not explain the
primary error's cause or justify changing its gate. Preserve all original
failures and investigate any correction prospectively.

## Pending completion and next execution

Keep session19976, parent2446755 and worker2447506. The saved audit is an
intermediate artifact: require the final prefix comparison, worker terminal
and root result with their verified bindings before launching the readout.
Then follow the existing fixed independent learned-layout cohort and matched
reactive/planning-memory comparison order. Scientific negatives remain in the
denominator; complete raw validity does not imply navigation qualification.

Hardware assessment69385 completed during this audit: 16 physical/32 logical
CPUs with all32 logical CPUs in affinity, 3.4% overall CPU activity,
72,311,721,984 bytes available RAM, 95,296,634,880 artifact-volume free bytes,
21,360,205,824 workspace free bytes, both GPUs at0% activity. Competing Python
processes were the original parent/worker/tracker plus the existing small
background process. Refresh measurements before the next substantial job;
retain one native scene at a time. No restart, source change, new scene,
checkpoint update or independent-layout execution occurred in this inspection.

## Subsequent prefix and worker completion

The original worker subsequently completed successfully with status
`MEASURED_FLOOR_TRANSPORT_MAZE_COLLECTED_AND_RAW_AUDITED`, wall time
7994.413019503001s and peak RSS11,952,197,632 bytes. Its terminal artifact
SHA-256 is `493df96dd733283c1d09682c0cb503671440577201e1431016506cc1cc10b32d`;
worker log SHA-256 is
`69c97bdb6216f2f3a3e6096b0c001e1c85f7cc0488e967ec7bb2650e934af139`.

Saved `full_jepa_novel_maze_00_prefix_comparison.json` SHA-256:
`99b7ff1a75e5c3a87731c809cf9ce9ad111adfd9e20181eea37d584e706ec5ec`.
All95,950 physical samples and1,905 paired public observations through frame1904
match the predecessor exactly. All earlier requested commands match, and all
complete candidate decisions match the prospective prefix replay. At1904 the
candidate requests `[0,0,-0.45]` instead of the predecessor's zero command.
No unexecuted outcome is inferred. Raw physical prefix SHA-256:
`a99b1e6a4021867718906c196f202ba063ac580240a4710685fc82c64f570643`.

Worker2447506 has exited normally; parent2446755 and session19976 remain live
in final verification. Root `result.json` and `failure.json` are still absent
at this update. Keep the existing parent and require its completed result
before running the dependent readout. The audit findings above are unchanged.

## Final root completion

Session19976 subsequently closed with exit0 and root result SHA-256
`1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374`.
The attempt is now complete, with the same negative scientific outcomes.
See `docs/go2_measured_floor_transport_maze_pilot_result_2026-09-09.md`
for final bindings, execution accounting and the subsequent readout/preflight.
