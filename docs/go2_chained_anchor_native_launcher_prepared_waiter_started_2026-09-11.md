# Chained-anchor simulation launcher prepared and queued

Prepared a prospective fresh-physics launcher for the original no-RGB JEPA
tracking case with the chained-anchor observer. The model, 3000-tick budget,
planner, memory, floor registration, robot, renderer, sensing, command set,
physics loop, and navigation/visibility success gates remain those of the
original tracking case. The prepared collector and auditor were preserved.
This experiment does not adopt the policy changes tested by other diagnostics.

The new input gate binds the authenticated controller replay, the completed
original tracking simulation and waiter, and the existing queue through
contact-plus-flow. Full completion reconstruction is required at admission and
finalization. Intermediate checks authenticate the same source, result, worker
and artifact bindings without recursively rerunning all prior queue audits.
The assigned model loader rechecks the original model and coefficient artifacts;
the completed original training admission is reused explicitly.

Validation completed before queuing:

- 99 tests for the new input/launcher gates and existing chained-anchor physical
  prefix and collector/auditor source invariants passed in 3.90 seconds,
  session 67810. An earlier 97-test run also passed before the final distinction
  between full admission and routine bound checks was added.
- Native source preflight passed with 2374 bound source paths and no simulation
  output, session 77303. Full runtime admission remains deferred until the
  diagnostic queue completes.
- The ordered waiter passed 40 tests in 2.41 seconds, session 84072. Tests cover
  exact process/completion identity, changed or missing predecessor outcomes,
  timeout without restart, a single child, preservation of child failure,
  complete raw artifact accounting, and reconstructed physical/readout results.
- Waiter preparation checked the actual ended controller replay and the live
  original contact-plus-flow waiter, with 2378 source paths and adequate resources.

Launcher preparation:
`docs/go2_chained_anchor_native_launcher_preparation_2026-09-11.json`, SHA-256
`634822ede9d6d9841184fca4cc4835d3e58cf41548d32d1f9a085046d62e559e`.
Waiter preparation:
`docs/go2_chained_anchor_native_wait_preparation_2026-09-11.json`, SHA-256
`e683428afe9bda46d3cb628759b5d6873e657726202145e83712969502fe73b0`.

The waiter was started in session 32044. Its launch SHA-256 is
`cf6703e83197d6c75df35b2b53834a47a53a293a06c64737ce94ade2ac0b87c1`.
Direct process inspection confirmed PID 2845479, creation time 1789129072.88,
exact Python argument vector, and original boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. It had no child and had created no
native simulation root. Its first event recorded the controller replay ended
and the original contact-plus-flow waiter still live.
Execution evidence:
`docs/go2_chained_anchor_native_wait_execution_2026-09-11.json`, SHA-256
`5bebc1c25f01bad39497a7d12139b5c0855b098d3c2ac9d4defa23327c88320b`.

The active diagnostic order is extended budget, sustained turn, contact plus
flow, then chained anchors. At the final direct check, extended-budget launcher
PID 2843773 (creation time 1789128335.77) was still alive, spending time on input
verification; no physics progress was established. The sustained and
contact-plus-flow waiters also remained live. No process was restarted.

The next scientific evidence must come from completed fresh simulation.
No new goal arrival, return trip, independent-layout result, JEPA advantage,
real-time control, or hardware result is established by this preparation.
The independent-layout policy review must account for all these diagnostics
before selecting and freezing a study controller.
