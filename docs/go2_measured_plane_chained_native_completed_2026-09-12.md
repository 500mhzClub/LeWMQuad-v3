# Chained native attempt completed: outbound arrival, incomplete return

The original parent and worker of
`go2_measured_plane_chained_maze02_v1_attempt_001` have ended. The original
parent tool session 69592 exited successfully and printed
`CHAINED_NATIVE_COMPLETE`. The final root result has status
`MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE` and SHA-256
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
There is no native `failure.json`. Completion here means that collection and
verification finished; the physical round trip failed.

The root is under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
Its launch SHA-256 is
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`.
The original parent was PID 2992412, creation time 1789193084.73; the worker
was PID 2994743, creation time 1789194027.81. Both exact owners were checked
as ended before the next experiment was invoked.

An independent read-only check authenticated the root result and launch,
rehashed all 2,627 bound source files and 24,124 bound artifacts, compared the
complete worker record with the root condition, and matched the saved
collection, readout and physical-prefix receipts with that worker record.
Every binding matched. The root result hash was checked again afterward.
This verification took 6.067467313027009 seconds and did not rerun a
controller or simulation. The native parent reports total wall time
15781.924906994915 seconds.

The completed population has 4,014 observations, 4,013 executed commands and
201,400 physics samples. It ends with `MISSION_TICK_BUDGET_EXHAUSTED`:

- The measured one-second outbound arrival-and-quiet window passed at frame
  3062, with maximum goal distance 0.04377755639348901 m and maximum speed
  0.0441619731798587 m/s.
- The physical outbound path had 15 crossings over ten distinct open edges;
  the return had six crossings over four distinct open edges. Both had zero
  invalid crossings, and the native readout recorded zero contact samples.
- The run did not complete a round trip or physically retrace the complete
  outbound route. The terminal observed home distance was
  2.621759960041774 m.
- Raw sensor reconstruction, complete model-command replay, unchanged model
  state, strict physical visibility and the prospective physical-prefix
  comparison passed.
- Acquisition-plus-control median was 1578.8952865 ms, with all 4,014
  observations exceeding 100 ms. Physics was paused during computation.

This is one reused development layout, with zero new independent-layout
executions and zero measured round-trip successes. It does not establish
independent-maze reliability, a JEPA/planning/memory advantage, real-time
operation or hardware qualification. Detailed raw-audit, prefix, readout and
worker identities remain in
`docs/go2_measured_plane_chained_raw_audit_result_2026-09-12.md`. That document's
earlier pending-parent snapshots are superseded by this completed result.

## Prepared timing comparison invoked after native completion

The already tested launcher
`scripts/replay_go2_measured_plane_chained_single_pass_full_history_v1.py`
was invoked with the actual completed root result SHA above, under the
original deterministic single-thread CPU environment. Its original tool
session is 70177; its process is PID 3015121, creation time 1789209876.83.
At the first live observation it was authenticating inputs, and its exclusive
output `go2_measured_plane_chained_single_pass_full_history_v1_attempt_001`
did not yet exist. No comparison result is claimed at this snapshot.

Available memory was about 75.7 GiB and artifact storage about 529.4 GiB.
The launcher still applies its own resource and idle checks before creating
the exclusive output. It compares complete recorded histories for the
chained controller and its single-pass implementation; it does not execute
another native episode or adopt the optimization in a robot controller.
The original native attempt was not restarted or replaced.
