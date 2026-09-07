# Paired inner-arrival audited result: 0/3 complete returns

The frozen paired collection completes with0/3 return candidates. All expected
artifacts are present; there is no native physical or storage stop. Both nominal
trials fail during TURN_BACK from visual tracking loss. Low friction completes
no stage and stops on observed excursion. This does not support adopting the
tighter controller as an end-to-end improvement. The independent audit is
complete and confirms exact setup pairing, raw sensor reconstruction and all
decisions, with six passing early local pose/winding holds but no home holds.

| Condition | Completed stages | Last decision tick | Stop |
| --- | ---: | ---: | --- |
| Nominal left |3/7|1127|No qualifying visual reference|
| Nominal right |3/7|1003|No qualifying visual reference|
| Lower-friction left |0/7|565|Observed excursion limit|

There are2,698 decisions,2,725 commanded intervals,2,728 RGB-D frames and
138,500 physics samples (277s). The lower-friction final decision has visual
XY(.48475,.13920)m and yaw-1.79981rad against goal(.4,0)/yaw0. Its x projection
exceeds the original goal-direction overshoot limit .4+.08m. Good tracking
does not correct the nominal table's support-dependent dynamics error.

## Tracking gate localization, unchanged sources and thresholds

The read-only [diagnostic](go2_inner_arrival_visual_failure_diagnostic_2026-09-06.json)
reconstructs the active pairs and fixed recent-reference alternatives:

- Left1127/reference1102:74/95 inliers (.779), five reference cells/six current
  cells, reference translation4.82mm. Only the support-count gate fails among
  the captured fraction/support/translation checks.
- Right1003/reference977:48/63 inliers (.762), six reference cells/five current
  cells, reference translation6.08mm. Again the captured failure is support.
- All eight recent candidates fail unchanged registration gates in each case.

These observations localize a proxy threshold; they do not establish that a
rejected pose was safe or that accepting five cells would give reliable control.
Investigate point/flow/depth survival, conditioning and actual error across
complete development streams. Test any replacement qualification against
independent error/latency/sensor challenges, not only the frames used to design
it. Preserve this batch and the earlier negative balanced-feature replay.

## Completed audit and evidence

Collector56322 is terminal exit0. Launch SHA256:
`53ec5a2a04831ddbc8ae3932038ce90a5f12c797d38087bb6fb95a3e8778667a`.
Collection result SHA256:
`938090fce09f77b9c55c919d6ee3b29d71dde5b5939374a542aa0edf5a7c0456`.

Auditor68877 completed with exit0, with launch SHA256:
`1ebefae89efbcd703c716aaad3515454f225469dba12feb50071709c80aa0b98`.
The completed left audit confirms exact750-sample setup/first-image pairing,
all1128 decisions replayed, all1138 raw-depth checks within1mm, three passing
local native pose holds and three passing signed-winding holds, but no home
hold. Maximum available visual position error10.34mm; native path1.99m.
Right also passes exact setup pairing, all1004 decision replays and1014 depth
checks, and three local pose/winding holds, but no home hold. Maximum available
visual error8.96mm; native path1.66m. Low friction passes setup pairing,566
decision replays and576 depth checks, but completes no holds. Its final native
XY is(.50322,.13945)m and yaw-1.79919rad; maximum visual error5.22mm, path1.19m.
All2,728 raw depth checks are within1mm. Audit result SHA256:
`07ae6d3a3ff3183e015617159cbd499faf751cbf13af646ac1ddcfa2bc0b8810`.

Pulse counts are43/37/23 (103 total). Median observation/control compute is
146.91/145.71/143.10ms, exceeding100ms with physics paused. Other CPU tests and
diagnostics overlapped collection, so timing differences are not an isolated
controller benchmark. Minimum free storage was156.47GiB; the40GiB reserve held.

Compared with the baseline left reaching home and right reaching six stages,
the changed nominal trajectories fail earlier. Exact setup pairing is confirmed
for all three conditions. Do not extrapolate a
reliability estimate from one batch or count a source/model-matched check as a
physical result. Tighter arrival adds control work and changes future imagery.
Six passing early holds do not establish correction of the baseline's failed
late return hold: this batch never reaches that stage. The differing hold
populations cannot be compared as6/6 versus14/15 evidence of improvement.

## Next

Collection and audit handles are terminal; do not restart either attempt.
Do not run another same-room tolerance variation merely because this one failed.
Advance perception failure/uncertainty validation and prospectively balanced
action/state/layout collection. The completed
[dataset integration](go2_pulse_dataset_integration_progress_2026-09-06.md)
supports the next actual learning-runner step, not yet learned control.
The full goal still includes independent-maze JEPA/baseline comparisons,
useful observed memory/backtracking, realtime/realistic sensing and bounded
hardware evidence when available.
