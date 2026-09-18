# Measured-floor-transport maze0 attempt completed

The native attempt completed collection, full raw sensor/model/controller/
command audit, exact prospective intervention comparison and final artifact
verification. Session19976 closed with exit0. The scientific result is negative:
one settled outbound arrival, no confirmed home arrival and zero verified
round trips. No navigation, timing or hardware qualification is granted.

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_floor_transport_maze_pilot_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| result.json | 1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374 |
| launch.json | 8dbd36ce4c300bef2b42b31cb6c04d5633624163684f1a5e88de5fcbb8002369 |
| full_jepa_novel_maze_00/result.json | 0153c24c6974fb5f8df736ca0a24874766b3ef93b552ad020af56242296bb8cc |
| full_jepa_novel_maze_00_audit.json | 449d23341e1449754eaf698757be2b2553cefdcd4345d77e372ac03cb66ea368 |
| full_jepa_novel_maze_00_prefix_comparison.json | 99b7ff1a75e5c3a87731c809cf9ce9ad111adfd9e20181eea37d584e706ec5ec |
| full_jepa_novel_maze_00_worker_terminal.json | 493df96dd733283c1d09682c0cb503671440577201e1431016506cc1cc10b32d |

The root result binds1,654 frozen source paths and18,123 artifacts. Root wall
time is8139.471536834026s; worker wall time7994.413019503001s with peak
RSS11,952,197,632 bytes. There was one native scene, no model training and
no independent-layout execution. Twelve navigation episodes have now completed
and been audited; three reproduce a settled outbound arrival on maze0, with
zero verified round trips across the completed episodes.

## Outcomes and intervention fidelity

Collection exhausted its original shared3,000-navigation-tick budget, with
3,014paired observations/decisions,3,013completed commands,151,400physics
samples and10terminal zero commands. Physical and acquisition stops are null.
The outbound native arrival at1868passes the one-second quiet-window test:
maximum goal distance0.03761944072156676m, maximum3D speed
0.03901577611147847m/s. Neither physical route retracing nor terminal native
quiet nor native round-trip candidate evaluation passes.

The floor-transport intervention preserves95,950physical samples and1,905
paired public observations through frame1904. All earlier requested commands
and all prospective complete candidate decisions match. At1904 the new
controller continues with `[0,0,-0.45]` where the predecessor stopped. The
same assigned model remains unchanged and the entire new sensor-to-command
history replays exactly. Continuing past the old stopping point did not
produce a successful return.

Strict primary visibility first fails at909. New primary stable-interior
metric failures at1924,1925,1930 are retained, each with one sampled ray
exceeding the unchanged1mm gate. Their respective maximum errors are
0.00100578011992436m,0.0010578653070654198m and0.0010515414946317136m;
auxiliary visibility passes at those frames. See
`docs/go2_measured_floor_transport_saved_raw_audit_2026-09-09.md` for the
audited measurement and traversal details. No threshold adjustment or causal
explanation of the measurement error is inferred from these values.

## Next jobs admitted from this completed result

Hardware2507 closed with all32logical CPUs available on16physical cores,
0.3%overall CPU activity,83,455,864,832bytes available RAM,
95,288,840,192artifact-volume free bytes and21,360,197,632workspace free
bytes. Both GPUs were0%busy; only the existing small background Python
process remained. The current resources support the8GiB readout and the
independent cohort's32GiB RAM admission and40GiB reserve plus33GiB planned
remaining-case allowance. These are capacity checks, not enforced quotas.

The existing readout started in session48066 using the exact final result
SHA above. The existing independent-layout preflight started in session52560
with the same SHA and `--preflight-only`. The preflight validates sources,
inputs and current resources without creating an output or native scene;
it can run alongside the CPU-only readout with the measured headroom. Require
both to finish successfully before the independent native launch. Preserve
the fixed layouts1,2,3 order and one native scene at a time. Subsequent
reactive and planning-memory comparisons remain queued as previously defined.
