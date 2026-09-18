# Temporary phase instrumentation prepared; no replay launched

The completed single-pass replay reports whole-controller observation times,
not component durations. Its median is 487.501 ms in frames 0–255 and
632.893 ms in frames 2048–3071. Both exceed the 100 ms target. These windows
do not isolate the cause of latency or establish that retained history causes
the increase; other native work overlapped part of that replay. The complete
timing result is recorded in
`docs/go2_measured_plane_single_pass_full_history_result_2026-09-12.md`.

The existing `PhaseTimedSinglePassReceiptController` constructs an older map
and memory implementation. Substituting it for the current measured-plane,
body-projected, single-pass controller would therefore change what is being
measured. The new `lewm/measured_plane_phase_timing_development.py` instead
temporarily wraps bound methods on the caller's existing controller objects.
It preserves their classes and shared map/memory identity, calls the original
bound methods, and restores preexisting instance overrides or removes its own
overrides on context exit, including when an original call raises.

Explicit phases cover the controller, visual motion, floor registration,
mapping, memory operations, model forward, selection and result construction.
The existing nested `PhaseTiming` sink reports inclusive and exclusive times;
exclusive times partition the timed root without counting nested calls twice.
Instrumentation overhead remains part of these measurements. Methods are
wrapped only during synchronous calls on an owned controller; this is not an
attachment mechanism for an already running experiment.

All five focused tests passed in 26.48 seconds. They cover original return
identity, side effects, exceptions escaping the context, restoration of an
existing instance method, and invalid-binding rejection before installation.
Actual synthetic image-to-action cases exercise the current optimized
controller with JEPA/full and direct/no-RGB models. Complete decisions and
the existing complete observed-state fingerprints match an uninstrumented
controller after each observation, including a duplicate-packet sensor-failure
latch. Both cases preserve object types and identities and model state, remove
all temporary method overrides, and account for the same single model forward.

| Prepared file | SHA-256 |
| --- | --- |
| `lewm/measured_plane_phase_timing_development.py` | `6ab0372bb0e1be5e5167fa45f8015dd8cf14cdd8106c1185004d0e414db7fff8` |
| `lewm/tests/test_measured_plane_phase_timing_development.py` | `5acb87892f008614725495faad774a99bbf36acbede0c2689e432fc022689080` |

No native or recorded-history profiling experiment was launched. No source
bound to the active chained-controller replay or prepared native experiment
was modified. The original replay remains the immediate dependency for the
fresh tracking-repair simulation. A future owned profiling experiment must
bind its actual controller and inputs and compare its complete decisions and
state; these small synthetic tests are not long-history equivalence or
component-cost measurements. This preparation establishes neither a speed
improvement nor navigation, real-time, hardware or deployment qualification.
