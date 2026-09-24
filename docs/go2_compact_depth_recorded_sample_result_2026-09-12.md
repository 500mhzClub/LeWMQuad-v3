# Compact archives preserve four actual recorded sensor packets

The prospective check completed successfully for exactly frames 0, 3062,
4003 and 4013 of the completed chained native development episode. All four
complete public packets matched after compact encoding/decoding. Raw primary
and auxiliary depth bits and diagnostic segmentation dtype/bits were preserved.
Original inputs were unchanged.

The twelve original depth archives occupy **12,497,036 bytes**. Their eight
compact counterparts occupy **6,255,262 bytes**, a **49.9460% reduction for
these depth archives**. This excludes common RGB and global metadata and is
not a measured complete-run or independent-population saving. Including this
check's launch, report and result, its entire output occupies 6,983,240 bytes,
within the prospective 64 MiB allowance.

The result status is `COMPACT_DEPTH_RECORDED_SAMPLE_V1_COMPLETE`, SHA-256
`ce9fd7809904792182feb1a2348d07ad33577afd13a8a227c2dbca98db00b033`.
Launch SHA-256:
`a3def9d1587baed598abf0b5108f25c9839477affade24c35bf46a6dbef669b9`.
The output is `go2_compact_depth_recorded_sample_v1_attempt_001` under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
Original owner PID 3051435, creation time 1789227889.07, ended. Tool session
21542 exited 0. No sample `failure.json` is present.

The final checker source SHA-256 is
`14d648f01ed3891b6b1c21e66f52690be4922f5aa1ba86d842f921c64306ef7f`.
The protocol remains
`217b33980a5cf3382def78469d3c602a7f958154e8c7e52d2e2b7a702bb88cda`.
The launch binds 2,655 source paths, the 26 selected original input artifacts,
the fixed four-frame selection and the resource/scope definition. The terminal
result binds all ten outputs: eight compact archives, launch and report.

The earlier invocation in session 29519 failed before output creation because
it passed a nested episode directory to the attempt-root verifier. Its exact
failure, original source identity, path-only correction and successful input
preflight are preserved in
`go2_compact_depth_recorded_sample_preflight_correction_2026-09-12.md`, which is
also bound by the actual launch. No launched attempt was overwritten or resumed.

## Independent verification

Read-only verification session 71059 exited 0. It authenticated the result and
launch, confirmed the original owner ended, rehashed the source roster and
all output artifacts, and matched the complete saved report copies. It then
independently reconstructed all four original and compact public packets,
rechecked bitwise equality against the original native archives, compared each
saved packet fingerprint and per-frame byte count, reauthenticated the selected
inputs, and rechecked all outputs and the 64 MiB total-output bound.

The compact verification record is in
`go2_compact_depth_recorded_sample_result_2026-09-12.json`; complete per-frame
bindings remain in the artifact report. The reported 0.7086804718710482-second
run duration is operational wall time, not sensor latency or an isolated
performance benchmark. This short storage check overlapped the ongoing
non-timing controller-prefix replay.

No model inference, new controller observation, rendering, physics, training,
native-format adoption or deletion occurred. These are four selected actual
observations out of 4,014, not a complete-population proof. Full recorded-packet
and sensor-audit compatibility, capture/reconstruction timing and a prospective
integrated native format remain outstanding. The current prefix and already
prepared longer native trial retain their original file formats.
