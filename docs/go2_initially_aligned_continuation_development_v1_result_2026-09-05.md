# Initial alignment: unchanged task success, earlier failures, full audit passed

All16 fixed simulation trials completed, and the full raw audit passed all4,652
controller decisions. Every method achieves2/4 two-leg successes and2/4 initial
alignment timeouts. No native contact, stability stop or sensor-contract failure
occurs. This is not an improvement in navigation reliability: the previously
contacting fixtures now fail before the first traversal.

## Comparison and scope

The [fixed protocol](go2_initially_aligned_continuation_development_v1_2026-09-05.md)
adds observation-driven initial alignment to the same four corner/tee development
fixtures. Both-leg physical criteria, frozen models, matched fixed-forward,
scan/branch selection and final release remain unchanged. The global budget
includes the added alignment and hold. No predecessor trial was rerun or edited.

| Method | First leg /4 | Two legs /4 | Initial timeout /4 | Traversal choices |
|---|---:|---:|---:|---|
| Fixed forward |2|2|2|32 forward|
| Direct prediction |2|2|2|30 forward,2 forward-right|
| Supervised recurrent prediction |2|2|2|30 forward,2 forward-right|
| JEPA recurrent prediction |2|2|2|30 forward,2 forward-right|

The three learned arms have identical raw physical trajectories. There are only
three exact trajectory groups overall: eight identical failed runs, two successful
fixed-forward runs and six successful learned runs. Rendered corner/tee motifs
do not make repeated dynamics independent replications. No maze-level confidence
interval or general learning advantage follows from these counts.

Successful fixed-forward tasks take46.2 s after settling, versus46.0 s for all
learned arms. That small local difference is shared by all learned conditions;
it is not a JEPA-specific result. The predecessor successful tasks took43.0 s,
but this successor adds observation/hold time and changes gait state. It is not
an isolated heading-tolerance comparison.

## Why the intended correction fails

Both negative-offset/heading fixtures produce the same initial observed bearing,
+0.087266 rad. Alignment processes121 decisions over its12 s deadline. Error
falls to a minimum0.018523 rad and ends at0.018842 rad, inside the0.02 rad band.
However, the longest run of qualifying observations is three frames: only0.2 s
between first and last, short of the required0.3 s dwell.

The trace shows repeated switching: entering the tolerance band sets the yaw
command to zero; observed negative heading rate around0.009 rad/s then carries
the error outside the band; proportional control resumes around0.03 rad/s.
The dwell resets. These measurements identify a boundary-switching failure in
the implemented controller/plant interaction, not a lack of initial rotation.
The ideal synthetic stream did not include this measured zero-command drift.
They do not, alone, establish its underlying gait or actuator cause.

All eight such runs stop after12.9 s including release and never start either
traversal. There is consequently no evidence that alignment corrected their
arrival position or made their scan collision-free. Counting their absence of
contact as a successful clearance intervention would reward not doing the task.

The two positive fixtures propose zero initial body-relative bearing, so their
alignment completes after four quiet frames without a substantive corrective
turn. At first-traversal start, true relative heading differs from the stored
target by0.003237 rad; initial-phase translation is0.002132 m. The fixed-forward
scan starts at(1.228508,0.166462) m and drifts up to0.092123 m. These successful
cases do not test whether a nonzero correction improves the failed arrival pose.

## Evidence and preservation

Collection35621: COMPLETE16, exit0. Full audit21831: PASS16 and4,652 decisions,
exit0. Evidence includes247,800 physics/live-fast-gyro samples,24,780 ordinary
sensor samples and4,732 actual RGB packets. The audit verifies raw/native
contacts, causal sensors/histories, camera transforms, exact controller/model
replay, command tapes, ledgers and unchanged physical endpoints. It excludes
only the two declared learned timing fields from decision equality.

Before launch,876 focused tests across79 files passed (session57549,23.16 s).
Launch binds201 source/test/protocol paths,170 inputs and two gait bindings.
Those paths remain unchanged. Read-only diagnostic51244 failed on the wrong
selection-field name before reporting results; corrected read-only76114 completed
with the actual `selected_action_name` field. Neither diagnostic wrote artifacts,
changed outcomes or reran physics.

Root: `.generated/go2_initially_aligned_continuation_development_v1_attempt_001`.

- Launch: `369870d7e4546d49b3cb1c7740c56f4b39c392dce0acfc8232df35880f6222b9`.
- Result: `91b003a0218c3108226300624f6a9038d77f51a6ddcbe12af0d2ec40b68c67cf`.
- Full audit: `82907c2038aacdb82ae462a40e7b8f4a2e820c10a6a398ba7763d0f49c8ac744`.

## Next steps toward the full objective

1. Build a separately scoped alignment-control successor that addresses the
   measured drift/switching mechanism. Test bounded persistent feedback during
   the acceptance dwell before increasing model capacity or sweeping thresholds.
   Retain the heading/rate/dwell/deadline criteria, then measure the actual
   post-zero-hold pose; feedback-time alignment alone does not establish release
   stability. Add synthetic persistent-drift cases and both turn signs. Treat
   the proposed control change as a hypothesis, not a guaranteed solution.
2. Keep the next physical test continuous and matched. Count failed initial
   operators as task failures. If corrected alignment still leaves bad scan
   poses, test observed centering/repositioning or a separately declared sensor
   configuration, without privileged wall coordinates or retrospective margins.
3. Build uncertain episodic place/branch memory from actual observations, and
   integrate a rendered, initially hidden beacon and directed return in a small
   whole-task prototype. Keep hypotheses separate from qualified graph edges;
   do not require perfect place recognition before testing uncertain decisions.
4. Include scan/alignment/repositioning operators in future predictive training
   and comparisons: they remain hand-controlled and outside the frozen learned
   action bank. Independent mazes, appearance/sensor robustness and real-Go2
   evidence remain required. Local control progress is not final-goal completion.
