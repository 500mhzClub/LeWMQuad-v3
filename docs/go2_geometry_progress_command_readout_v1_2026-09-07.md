# Separate command-representation readout V1

The original geometry-progress collection is terminal and stays unchanged.
Its original frozen raw audit failed before completing its first episode and
remains failed. This is a separately named, posthoc raw readout of the same
24 recorded episodes, not a retry, repair, resume or successful original audit.
Use the exclusive new root `go2_geometry_progress_command_readout_v1_attempt_001`.
No simulation, recollection, training, action change or progress-threshold change.

Failure: the inherited audit cast requested metadata to float32 before exact
comparison with recorded float64 requests. For example, it compared recorded
0.45 with 0.44999998807907104. The actual recorder chain is explicit:
`GeometryProgressSession.command_tick` validates Python-float requests;
`FactorialSession.command_tick` passes those original requests to `_sample`, but
casts the actuator path to float32 before `_clip_block`. `WholeTaskPhysicsSession`
serializes both channels as float64. Requested values retain their original
precision; applied values are float32 values promoted exactly to float64.

The separate validator requires exact float64 requested-command identity from
the preregistered metadata. For applied commands it independently reconstructs
float32 clipping against the previous float32 command plus/minus float32
[0.25,0,0.35] tick limits. It requires exact equality for both applied-command
recorded fields. This replaces the old approximate applied-command comparison
with an exact representation check; no command-error tolerance is enlarged.

All original raw sensor/depth/contact, actuator, setup, native stop, shadow
observer, episode-denominator, future-censoring and predeclared progress checks
remain. Source-level comparison tests bind the unchanged audit-condition logic
and isolate the representation change. Synthetic fixtures use the recorder's
actual serialization and reject even one float64 ULP of request corruption or
one float32 ULP of applied-command corruption.

Before this readout launches, inspect all 24 recorded command tapes under the
new representation rule as a separately reported diagnostic. Freeze new source,
tests, this document, original source and the original terminal failure hashes
in the new launch receipt. One readout; preserve a terminal failure if any.
The scientific reader must identify this separate evidence and the original
failed audit. A passing readout does not establish RGB/JEPA benefit, independent
maze generalization, real-time execution, deployment or goal completion.
