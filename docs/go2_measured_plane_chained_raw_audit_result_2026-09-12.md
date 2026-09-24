# Chained controller raw audit passed; native final integrity checks pending

The raw audit for `go2_measured_plane_chained_maze02_v1_attempt_001`, case
`no_rgb_direct_measured_plane_chained_maze_02`, has been written. Its
`no_rgb_direct_measured_plane_chained_maze_02_audit.json` is 14,324,277 bytes,
SHA-256 `1abe2451e9e90878f57291f9c116ad8e4f07f0875d959998074c4ff147bc1f25`.
That identity was checked again after reading the result metrics.

The original worker PID 2994743, creation time 1789194027.81, remains active
in subsequent physical-prefix/integrity checks. The final worker receipt,
readout, physical-prefix comparison and root result are still pending at this
snapshot. This document records the emitted raw audit, not admission of a
completed native attempt. The prepared timing replay has not been dispatched.

## Verified by the emitted raw audit

Raw sensor reconstruction, additional auxiliary RGB reconstruction, complete
model-command replay, command audit and unchanged model-state checks pass.
Strict physical visibility passes, with no hard measurement-failure frames.
Renderer witnesses cover all 4,014 frames and 8,028 capture endpoints; paired
readbacks match and all witnesses match the raw acquisitions.

The native one-second outbound arrival-and-quiet window passes at frame 3062,
ending at physical sample 153849. Its maximum goal distance is
0.04377755639348901 m and maximum speed is 0.0441619731798587 m/s.

| Native traversal | Crossings | Invalid crossings | Distinct visited cells |
| --- | ---: | ---: | ---: |
| Outbound | 15 | 0 | 11 |
| Return | 6 | 0 | 5 |

Both traversals remain inside the maze and pass the native step bound. These
are evaluator-only native measurements; native state was not an online
controller input. The audit does not claim collision-clearance certification.
No contact count is inferred merely from its null physical-stop field.

The audit reports `verified_round_trip: false`,
`physically_retraced_outbound_route: false` and
`terminal_native_quiet_pass: false`. The completed collection independently
records `MISSION_TICK_BUDGET_EXHAUSTED`; its input identities and provisional
logged-return analysis are in
`docs/go2_measured_plane_chained_provisional_return_log_inspection_2026-09-12.md`.
The new result establishes physical return progress, not a completed journey.

The 4,004 reported observed-pose XY errors have median
0.005998929795609485 m and maximum 0.011689187497715475 m. These are native
evaluation comparisons, not calibrated uncertainty bounds for deployment.

## Timing and remaining scope

Across all 4,014 collection observations, acquisition-plus-control time has
median 1578.8952865 ms, p95 3228.40486895 ms and minimum 217.98203 ms. All
4,014 exceed 100 ms. These measurements include acquisition and are not the
controller-only population of the separate single-pass timing experiment.
Physics was paused during computation. The single-pass optimization was not
adopted in this native attempt.

This remains one reused development layout. The raw audit explicitly reports
no independent-layout execution, no established learned-planning or memory
advantage, and no navigation, real-time or hardware qualification. The pending
final checks must still authenticate the native attempt before the prepared
complete-history timing comparison can use it. The full thread goal remains
incomplete.

## Later prefix and readout receipts

The same original worker subsequently wrote
`no_rgb_direct_measured_plane_chained_maze_02_prefix_comparison.json`, SHA-256
`ceb157224b560ed8386f27cbe64988cd902c256f33de971d4f21b95b1c7edbbb`,
with status `COMPLETE_CHAINED_ACTUAL_PREFIX`. It reports an exact physical and
public prefix of 3,114 observations and 156,400 physics samples, exact
preintervention requested commands, and complete original and candidate
decisions matching the prospective controller replay. The first changed
decision is frame 3113; both its command and terminal status changed, and all
50 physical samples of the candidate intervention command are present and
completed. The common raw-physics prefix SHA-256 is
`79b4e5aac91417004668439879f568ef361f2813f79cd97e2be768d62bc2d231`.
This receipt does not infer outcomes beyond that comparison boundary.

The subsequent `no_rgb_direct_measured_plane_chained_maze_02_readout.json`,
SHA-256 `871498fe21df172371a6c22165038cec9a5d5de6df050fedaaa7ea241e1e3b56`,
reports zero native contact samples across all 201,400 physics samples.
Outbound traversal has 15 crossings over ten distinct open edges; return
traversal has six crossings over four distinct open edges. Both have zero
invalid crossings. Round-trip and complete physical-retrace flags remain false.
The readout identity was checked again before reading the edge counts.

These later receipts resolve the prefix/readout absence in the earlier snapshot
above. At this later observation the original worker is still active in its
further result checks, and the final worker-terminal and root-result records
remain pending. No new timing replay or native attempt has been started.

## Completed worker independently authenticated; parent still running

The original worker subsequently ended and wrote terminal status
`MEASURED_PLANE_CHAINED_MAZE02_COLLECTED_AND_RAW_AUDITED`. Its
`no_rgb_direct_measured_plane_chained_maze_02_worker_terminal.json` SHA-256 is
`979cc7838802a605f4d5025a6e8027c1df203ae28f8174c2fd2ca8525bc9b2ef`.
Worker wall time is 14946.554418632062 seconds. It confirms unchanged model
state SHA-256
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`
and `verified_round_trip: false`.

An independent read-only verification rehashed all 24,120 worker-bound
artifacts and all 2,627 native-launch source files. All matched. It also
verified the closed worker-log hash, matched complete saved collection,
readout and prefix receipts to the worker record, and rechecked the terminal
record's identity after verification. The verification took
6.137415150878951 seconds; it did not rerun a controller or native simulation.

A transient denial when observing the worker's open-file metadata occurred
during its exit. Re-polling the same original owner confirmed that it had ended
and that the successful terminal record was present; no process was restarted.
Parent PID 2992412, creation time 1789193084.73, remains active and was observed
reading the physical-prefix inputs in its own final checks. The root result is
still pending at this snapshot, and the timing replay has not been dispatched.
