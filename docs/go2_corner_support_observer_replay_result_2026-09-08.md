# Corner features restore complete recorded-trace pose support

The new corner detector supplies admitted current joint RGB-D poses on all270
recorded observations, including the formerly failing left-frame5. It passes
the prospectively frozen diagnostic thresholds in both complete traces and is
eligible for a separately frozen native probe. No native control was run with
this candidate. Both original mission failures and the failed spatial-SIFT
quota experiment remain unchanged. The navigation goal remains active.

| Complete trace | Original poses | Corner poses | Corner maximum XY error | Corner maximum rotation error |
| --- | --- | --- | --- | --- |
| family_episode_052,16 frames | 5/16 | 16/16 | 0.312mm | 0.000386rad |
| family_episode_039,254 frames | 254/254 | 254/254 | 5.154mm | 0.004562rad |

All actual observations, including terminal zero drains, remain in the
population. The original left-frame5 failure reproduced. The candidate never
failed or reinitialized. Native state was opened for scoring only after each
observer completed; it never supplied a pose to the observer. These are
conditional simulator measurements, not calibrated uncertainty bounds.

At left-frame5 the corner detector selects166 of178 liftable detections. The
selected registration has88 inliers covering12 reference cells and11 current
cells, exceeding the unchanged six-cell requirement. The old fit had21
inliers covering only3 current cells. Detecting corners of the large texture
blocks addresses the measured feature scarcity that quotas could not fix.

The candidate uses Shi-Tomasi detection at quality0.01, minimum distance5px,
block size3, public-depth validity and unchanged lifting; at most50 corners per
4x3 cell and600 descriptors. Upright SIFT descriptors use size8px/angle0.
Mutual descriptor ratio0.7, LK checks, depth lifting, rigid registration,
spatial coverage, gyro consistency, reference retention, bridge budgets and
failure latching are unchanged. No optical-flow-only fallback was introduced.

Eight focused tests passed: source-AST scope, detector bounds/determinism and
ownership, unknown-depth exclusion, real rendered RGB-D pose admission, two
latched-failure cases, coarse-block translation with unchanged registration,
and repeated-checker descriptor rejection. These tests and replay do not
establish robustness to all real-world texture, lighting or sensor failures.

Median active observer time was35.814ms for the short trace and43.989ms for
the long trace. Maxima were47.610ms and60.244ms, with no observer call over100ms.
The original long trace median was57.553ms and maximum139.983ms. The short
original timing covers only its six-call pre-failure prefix. Packet loading,
planning, rendering and gait execution are excluded; the previously measured
full-loop timing deficit is unresolved.

The preflight found32 logical/16 physical CPUs,82.41GB available RAM,96.57GB
free artifact storage,0.2% CPU activity and idle GPUs, with no substantial
competing Python task. Both arms ran serially with one computational thread.
No physics, model training, GPU work, source export or hardware motion occurred.

Root under the established navigation artifact base:
`go2_corner_support_observer_replay_v1_attempt_001`.

| Receipt | SHA-256 |
| --- | --- |
| launch.json | 162ad020a1de3781259dafb284e9724174363985e4bcc75ea0595082faef163c |
| result.json | 97517b61fc33d2cdc484d96f0f47aeef0e730defb866444b08cc0ac41cb97375 |

The terminal binds six artifacts totaling19,174,245 bytes and867 source files.
It records `candidate_eligible_for_separate_native_probe=true`,
`candidate_adopted=false`, `navigation_qualified=false` and `goal_achieved=false`.
The process exited0, and no job remains running for this attempt.

New source entry points are `lewm/corner_support_features_development.py`,
`lewm/corner_support_joint_observer_development.py`, and
`scripts/replay_go2_corner_support_observer_v1.py`. Preserve their frozen bytes.
The candidate can next be integrated in a new named native controller using
fresh prospective source/model/mission bindings. An observer-only rerun of the
unchanged turn-only policy would leave the main action-selection failure.
The new transition-prediction bootstrap scope is defined separately in
`docs/go2_family_transition_bootstrap_inputs_v1_2026-09-08.md`; it preserves the
old failed navigation-design/readiness flags and does not claim goal success.
