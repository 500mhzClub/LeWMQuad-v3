# Recorded pulse native targets: binary serialization adapter V1

The original derivation stopped before any labels were produced because the
audited collector records contact as binary uint8, not numpy bool. Preserve
its launch and terminal-failure files. This successor adds only an explicit
binary uint8 -> bool decoder; values other than0/1 and other dtypes fail closed.
Native trace samples, requested commands, target times and label semantics are
unchanged. No source or output from the launched original is edited or resumed.

Apply the complete target-only semantics and limits in
`docs/go2_pulse_native_targets_v1_2026-09-06.md`: all185 old indexed windows,
start-body XY/relative yaw, exact partial horizons, sample-level command prefix
checks, observed pre-divergence contact positives, and unknown noncontact-stop
tails. No new physics, running intent-return data, fitting or model training.

Exclusive output
`.generated/go2_recorded_pulse_native_targets_v1_attempt_001`. Bind the original
failure/launch witnesses plus original pairing/trace/source identities. Before
creating output, validate the decoder and native-trace contract against every
one of the three actual source traces. New focused tests exercise all three
recordings, invalid binary representations and nonmutation; all existing
target-semantics tests remain. Freeze this source, decoder and tests at launch.
Report every window and positive/censored label count without scientific
promotion of a successful derivation to useful prediction or navigation.
