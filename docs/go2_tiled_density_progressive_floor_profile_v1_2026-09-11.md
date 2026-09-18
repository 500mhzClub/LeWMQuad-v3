# Tiled controller complete-history profiling diagnostic

This profiles the remaining controller costs after density routing, progressive
patch batching and tiled dense floor geometry. It consumes the same original
1,428 observations, with 1,425 model forecasts, as the completed paired replay.
It is a development timing diagnostic with recorded sensor inputs and commands.
It adds no physics, navigation success, training, independent-layout evidence,
hardware evidence or real-time qualification.

The runner is `scripts/profile_go2_tiled_density_progressive_floor_late_history_v1.py`.
The exclusive output is
`go2_tiled_density_progressive_floor_late_history_profile_v1_attempt_001`
under the existing navigation development artifact root. No retry or resume.
Before launch, require the actual tiled result and completion-receipt SHA-256
arguments, the fixed launch binding, an ended original owner on its recorded
boot, no original failure, unchanged complete sources and raw/model bindings,
64 GiB available RAM, 41 GiB artifact space and four physical CPUs. Keep one
full CPU replay and one native scene at a time. Source-only preflight does not
require the future result and neither admits raw/model inputs nor creates output.

The existing frozen tiled completion checker is reexecuted read-only. Its exact
function body is cloned with a private argument parser supplying the explicit
result SHA, a new absent capture destination, an in-memory receipt writer and
muted progress. Its existing receipt is not overwritten; no capture file is
written. All original owner, raw/model, source, artifact, row, timing, report,
state-witness and negative-sensing checks remain. The reconstructed receipt must
equal the recorded receipt in every field except the verification UTC timestamp.
This is repeated after profiling. It is admission reauthentication, not a second
paired controller execution or a new scientific attempt.

The original complete-history profiler body is reused unchanged, with private
bindings for the tiled controller, its complete original-decision normalizer,
the exclusive output and progress prefix. No imported globals are changed.
Every input and complete candidate/original decision hash must equal the
completed tiled replay. The model state and absent gradients are checked.
The original sensing failure at frame 1173 is retained; this is not a qualified
sensing history. The profile does not independently repeat the seven retained
state equivalence witnesses already established by the paired replay.

The fixed ten-observation windows are early navigation 3–12, repeated hold
395–404 and late navigation 1418–1427. All intervening observations execute
normally to retain the complete causal history. Each window produces a cProfile
`.prof` and JSON function/module summary. Profiling includes controller observe
only; acquisition and decision normalization remain outside the measured region.
Profiler overhead is not removed, and shared-host interference remains. Use
exclusive time to account for costs; cumulative call times overlap. No speedup
or deadline claim can be inferred from these profile measurements alone.

Validation covers the exact reused profiler body, all 1,428 decision identities,
missing or changed binding rejection, private CLI and receipt capture without
filesystem mutation, the actual frozen checker's live-owner rejection, full
negative-witness comparison, exclusive output and source-only preflight.
