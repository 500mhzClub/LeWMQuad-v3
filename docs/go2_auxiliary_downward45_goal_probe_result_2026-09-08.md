# Explicit 45-degree auxiliary camera native result

Both fixed corrected seed-2026091001 models completed collection and exact fresh
controller replay on reused family_episode_039. All primary and auxiliary
measurement gates passed. Verified goal arrivals: **0/2**. No independent-maze
evaluation was performed.

| Outcome | JEPA | Direct |
| --- | --- | --- |
| Terminal decision tick | 52 | 46 |
| Active zero-wait ticks | 42–51 | 36–45 |
| Feasible recoveries | 0 | 0 |
| Commands including ten-command terminal drain | 62 | 56 |
| Paired frames | 63 | 57 |
| Native physics samples | 3,850 | 3,550 |
| Minimum native goal distance, m | 1.117060187496089 | 1.1011647990168776 |
| Terminal native goal distance, m | 1.1178615237085892 | 1.1011647990168776 |
| Maximum observed XY error, m | 0.0013957027851192636 | 0.0013957027851192636 |
| Median complete iteration, ms | 807.558889 | 811.5194035 |
| Maximum complete iteration, ms | 1041.319107 | 1034.308685 |

Both terminals were NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS.
Neither collection reported an acquisition or physical stop. All 120 auxiliary
frames passed with zero robot pixels; the complete visible robot remained in
the renderer. All 118 complete command iterations exceeded 100 ms. Physics was
paused during computation; acquisition alone had medians 193.980767/204.2734 ms.

At JEPA ticks 42 and 52, all six candidates passed the first-step and all-eight-
step nominal checks but failed the auxiliary FL_foot:0 surface check. These
were measured-floor-only hits (1–4 voxels), zero auxiliary other/unknown hits,
and no primary hits, without complete measured foot-projection coverage.
The 45-degree camera had covered the predecessor's retrospectively diagnosed
foot locations, but the new online trajectory encountered a different gap.

Direct tick 36 failed every first-step nominal check despite passing all surface
checks: nearest cell [11,-2], distances 0.4462490192666565–0.4487070281492521 m
against the unchanged 0.45-m radius. At terminal 46 these were
0.4373912138492425–0.44060363329014324 m. Zero waiting did not restore feasibility.

Each method first changed its command relative to the 30-degree predecessor at
tick 17. Through the 18 corresponding observations, native physics, public
policy/gyro histories, primary RGB and model forecasts were exact. Combined
observer/map receipts differed with the auxiliary calibration/map. The new
JEPA/direct pair first differed at command 26: all 27 shared-prefix native,
public, primary RGB, auxiliary depth/mask and observer/map records were exact;
model forecasts differed. No unexecuted native outcomes were inferred.

The prospective run bound 1,296 sources and took 131.00014719599858 seconds after
launch. Hardware admission recorded 82,301,906,944 bytes available RAM and
66,879,053,824 bytes free artifact storage. Two one-thread CPU scene workers ran
within the declared allowance. The readout bound 1,299 sources; two comparison
boundary tests passed in 1.80 s. Three native scope tests passed in 1.79 s.

| Artifact | SHA-256 |
| --- | --- |
| Native launch | 9261d6fd4ded629079204062c380739b65db801bf942fe4cbdec833b6760e7d5 |
| Native result | 5007693d6e2b178d13c42164a08d561bf3ad2a60801b9290fb8033db811ce7fe |
| Readout launch | d41db72d6ba465123bb6661935b1ae1a2b74192795e4e92d6685f5b596fd02c4 |
| Readout result | 5d2f6df4c9762ac6d7f28c8546cea93e2d61356cda7a567561c32894c7667d6f |

Roots: go2_auxiliary_downward45_goal_probe_v1_attempt_001 and
go2_auxiliary_downward45_goal_readout_v1_attempt_001.

Next: diagnose whether the actual JEPA floor gap is missing observation or an
overly restrictive aggregation of existing complete pixel patches, using the
recorded 45-degree trajectory and unchanged controller. Do not choose another
camera angle solely to cover predecessor coordinates. Direct-model nominal
stopping remains a separate motion-prediction/control problem. Full navigation,
backtracking, independent layouts, matched baselines and realistic timing/hardware
evidence remain outstanding.
