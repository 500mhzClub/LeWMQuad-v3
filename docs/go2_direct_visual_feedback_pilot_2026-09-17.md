# Direct visual feedback versus future-image planning

Status: **COMPLETE: feedback 1/2, world-model reference 2/2 final arrivals**.
Native feedback sessions 20013/69724 and reader 13653 exited 0. Both new trials
completed the fixed 20-decision budget without contact. No process remains live.
The two world-model outcomes are retained references, not new replicates.

| Geometry | Controller | Final XY cm | Final heading degrees | Final arrival | Contact |
|---|---|---:|---:|---|---|
| cluster 02 | World model + visual arrival | 2.624 | 3.657 | Yes | No |
| cluster 02 | Direct visual feedback | 2.624 | 3.657 | Yes | No |
| cluster 03 | World model + visual arrival | 2.624 | 3.657 | Yes | No |
| cluster 03 | Direct visual feedback | 6.043 | 1.032 | No | No |

Both controllers use the same frozen V-JEPA encoder, signed goal readout,
supplied goal RGB, six primitive commands, limiter, 500-ms observation/command
interval, 20-decision budget, and visual arrival latch. The feedback controller
does not call the future-image predictor. Shared initialization loads then
releases that model; the shared scalar goal metric is recorded for diagnosis
but does not select feedback actions. No additional fitting was performed.
The inherited per-run `arm: action` field is only a runner initialization label;
these new runs are direct feedback, with zero predicted windows.

The feedback rule was fixed before these trials. With planar goal bearing
alpha and signed goal heading theta, request yaw rate
clip(1.5 alpha - 0.5 wrap(theta-alpha), -0.45, +0.45). When the goal is in
the forward half-plane and farther than 3 cm, choose the closest yaw rate
among forward/left-arc/right-arc. Turn toward the goal when it is behind;
once estimated position is within 3 cm, turn toward final heading with gain
1.5. Choose between the two pure turns in these cases. The common learned
arrival rule alone selects hold. This is one fixed discrete feedback law,
not an exhaustive or optimized reactive benchmark.

Initial 11-frame RGB prefixes match the world-model references exactly.
Cluster 02 feedback selects four left arcs then hold, matching the world
model. Full saved base poses, applied commands and contact flags are exactly
equal across those runs. Prediction is therefore unnecessary to reproduce
that task's successful trajectory.

Cluster 03 first diverges at tick 20: feedback chooses forward while the
planner chooses another left arc. Its sequence is left-arc, left-arc, forward,
left-arc, left-turn, then hold. At tick 35 the signed goal readout estimates
[-1.446, +0.759] cm (1.634 cm magnitude) and 1.097 degrees. Actual error is
5.270 cm and 0.135 degrees: the common arrival detector produces a false
positive and permanently latches hold. Actual final distance reaches 6.043 cm.
At tick 110 the readout itself reports approximately 5.71 cm distance, but
the permanent latch prevents further correction. Thus this comparison reveals
both route-dependent readout error and a consequence of the irreversible latch.
Neither has been tuned away; the full failed recording is retained.

The prior readout diagnostic had zero false positives on its 11 outside-goal
states. This newly executed trajectory demonstrates that this small sample did
not establish reliable arrival recognition. The planner's 2/2 result remains
valid on its trajectories, but is not evidence that the shared detector is
generally reliable. The current difference cannot establish broad planning
superiority, independent maze generalisation, or a JEPA-training advantage.

Wall times were 33.35/33.78 seconds versus reference 37.12/38.00 seconds.
Feedback selection averaged 0.128/0.153 ms, excluding RGB encoding and readout.
Physics paused during computation. These are development timings, not a
controlled compute benchmark or real-time/hardware qualification.

Next retain both frozen controllers for a fresh-layout/goal comparison, report
arrival recognition separately from approach decisions, and avoid tuning on
these two exposed tasks. Full independent maze navigation, exploration, memory,
backtracking and isolated JEPA representation evidence remain outstanding.
Additional native runs need a smaller prospective recording policy: depth is
unused by these controllers and dominates disk consumption. Retain all current
failure data; do not repeatedly spend the storage reserve on redundant depth.

Before launch, 72 GiB RAM was available, GPU utilization was 4%, and no other
experiment was running. Runs used CPU groups 4-7/8-11 and shared the R9700 GPU.
Case 0 was saved on the root volume and case 1 on the dedicated experiment
volume, each with its own measured-size check plus 512 MiB reserve. Approximately
708/585 MiB remained afterward. Only the completed duplicate arrival success's
unused depth was retired before launch; case 0 arrival depth and every failed
recording remain full. See the retention policy and receipt.

Plan: `go2_direct_visual_feedback_pilot_plan_2026-09-17.json`.
Results: `go2_direct_visual_feedback_pilot_result_2026-09-17.json`.
Controller: `lewm/direct_visual_feedback_control_development.py`.
Runner/reader: `scripts/run_go2_direct_visual_feedback_pilot_development.py` and
`scripts/read_go2_direct_visual_feedback_pilot_development.py`.
Exact output directories are recorded in the plan and result.
