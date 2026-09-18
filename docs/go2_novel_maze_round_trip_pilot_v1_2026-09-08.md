# First prospective native maze round-trip pilot V1

Run `scripts/run_go2_novel_maze_round_trip_pilot_v1.py` once in the exclusive
`go2_novel_maze_round_trip_pilot_v1_attempt_001` artifact root. This attempts the
actual user objective's outbound/return behavior on the first fixed prospective
maze. A component check or close endpoint is not counted as a successful round trip.

## Prospective assignment and controller

The only launched case is `full_jepa_novel_maze_00`, layout index 0, unchanged
`seed_2026091001_full_jepa` with its existing training-only translation correction.
The corrected model identity is
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
No fitting, checkpoint selection, native residual feedback or hidden route input
is performed. The completed variable-controller compatibility result
`5c9d1d60e789329ccb7d2a035388e9fa315f7d702aa7d7dfa438c3b49164792f`
is an exact admission dependency, including both 311-frame recorded comparisons.

The four source-generated layouts retain order 0, 1, 2, 3 and their frozen
seeds/topologies. The first maze has six shortest-path edges, five turns and
7.8 m shortest outbound length; its dihedral topology identity is
`fe9de58f9b859cd8732fd6da53aaab1ca07ede710f2f5906bb4bf0aa2bc4378a`.
Its public instruction is initial-body XY `[3.9000000000000004, -1.3]`, then
`[0, 0]` after the original observed arrival dwell. Only these coordinates and
the required-return boolean enter the mission controller. Physical geometry,
maze edges, shortest routes, native poses and contact guards stay outside it.
Source-disjointness was checked against the 24 counterfactual and eight online
choice maze source generators; this is not a claim of sealed-test evaluation.

The shared outbound-plus-return budget is prospectively **3,000 navigation
intervals (300 simulated seconds)**, with three warmup intervals and ten terminal
zero intervals. Up to 3,013 commands and 3,014 observations are allowed, within
the existing primary reader's 3,611-frame bound. The four-maze source's longest
shortest round trip is 33.8 m; the declared budget allows planning/exploration
time beyond ideal 0.2 m/s translation but does not guarantee sufficient time.
The budget does not reset on outbound completion. No predecessor attempt is
extended, resumed or relabeled.

The tested `ObservedRoundTripController` retains its mapper, surface history,
visual tracker, four-frame learned history and causal residual estimator on the
return transition. It keeps the original six actions, first 100 ms commitment,
all eight forecast path checks, measured floor contact rules, 45-degree auxiliary
camera, 4 cm observed arrival region and ten completed quiet intervals. It can
issue ten valid infeasibility waits before stopping. Sensor/model/view failures
and the global deadline remain latched stops.

## Native measurement and verification

Scene construction is the narrow explicit successor of the visible-robot native
session, changing only the fixed scene pack and labels. It retains the actual
Go2 collision/visual geometry, frozen PPO gait/gains, 2 ms physics, 20 ms body
sensor acquisition, 100 ms command/acquisition schedule, physical contact and
stability guards, ordered robot-visible raster and primary/auxiliary calibration.
Both optical captures refer to the same native sample. The extended auxiliary
packet reader changes only its explicit maximum index. All raw pixel/mask/hash,
calibration and causal clock checks remain required.

Every complete decision is written once to exclusive deterministic gzip JSON
lines. This is lossless receipt storage, not data reduction. The reader rejects
nonconsecutive frames, partial lines and oversized rows. Per-row write time and
iteration time including the receipt write are separately recorded; the small
timing-log write itself is outside that interval. No complete decision list or
duplicate full selection list is retained in the collector/audit result.

After collection, a fresh unchanged model reconstructs every recorded public
sensor packet and complete decision exactly. The independent raw-command audit
checks all requests, float32 slew/readback, physical samples and drain roles under
the new declared bound. Original articulated setup, contacts, raw sensor
reconstruction, pixel/metric/physical visibility and auxiliary robot visibility
checks are retained. The evaluator checks both observed arrival claims against
their own 501-sample native radius/speed windows and 500 intervening zero requests,
and checks physical outbound/return cell crossings, no teleports or closed-edge
crossings, and reverse traversal of the unique tree route. Final success also
requires terminal quiet, no contact/stops and all raw measurement/visibility
checks. Numerical evaluator candidates alone are not success.

## Resources and scope

Inspect current topology/affinity, CPU/GPU activity, competing processes, RAM,
and both output volumes before launch. Use one fresh CPU scene process and one
thread per numerical library; this longer workload has not been measured, and
simultaneous matched scenes would exceed the available storage allowances. Admit
only with at least 32 GiB available RAM and **12 GiB collection plus 1 GiB final
persistence headroom above the unchanged 40 GiB reserve**. These are admission
allowances, not enforced OS resource limits or a proof that the maximum-length
episode fits. During collection, stop if total observed volume consumption since
start reaches 12 GiB or less than 41 GiB remains. Other concurrent writers count
against this conservative volume check. Record resources every 15 seconds and
preserve a storage-limited attempt as unsuccessful/incomplete evidence.

This launch covers the first learned native maze case only. Its matched direct
case and remaining layouts require subsequent explicit resource-reviewed runs;
reactive/non-predictive controls and RGB/planning/memory attribution are also
outstanding. A failed maze is not discarded or replaced by an easier index.
Source and input bindings are checked before/after execution and audit; failures
remain at the exclusive root. No retry/resume, sealed access, hardware movement,
deployment or promotion occurs. Physics pauses during computation, and ideal
sensor/zero acquisition latency assumptions remain explicit. No real-time,
hardware, general navigation or learned-planning advantage is claimed by launch.

## Prelaunch checks

The existing scene source, mission/controller and evaluator tests passed in their
preceding work. Twelve additional native/logging checks passed: narrow native
source and packet/command-audit scope, paired frames at 253, 254 and 3,013,
over-bound rejection before rendering, the full 3,013-command audit and tampered
dispatch rejection, exact/exclusive gzip round trips, bad frame/NaN rejection,
and truncated/missing-line rejection. Initial test collection caught a missing
copied command-audit function; the source extraction was corrected before this
launch and all twelve tests then passed. No native attempt preceded that fix.
