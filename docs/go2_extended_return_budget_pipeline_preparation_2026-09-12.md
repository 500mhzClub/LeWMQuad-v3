# Longer mission recording and audit pipeline tested

The prospective 8,000-navigation-step controller is now connected to the
collector, public sensor replay, complete decision stream, command audit,
camera witnesses and native outcome evaluator. No longer native trial has
been launched or queued. Actual recorded-prefix equivalence and resource
admission remain prerequisites for that trial.

## Source identities and validation

| File | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_maze_pipeline_development.py` | `c5f90b5dae863a187f304e6b1bb67c5fb02b1da0d2b807a3ed9db97d7c5c8430` |
| `lewm/tests/test_extended_return_budget_maze_pipeline_development.py` | `f4472a9e2b64c1cd9379734b003104ad53ba07d3104212fea512eb1c36936c5e` |

The focused suite passed **11 tests in 5.94 seconds**, tool session 59640,
with the original deterministic single-thread environment and
`pytest -q -p no:cacheprovider`. An earlier observed run, session 97107,
had **one failure and ten passes in 5.97 seconds**: the test incorrectly
expected `ValueError` when opening an existing recording. The original
exclusive writer correctly raised `FileExistsError`. Only that assertion
changed; the writer and all pipeline implementation remained unchanged.

These tests overlapped the already declared non-isolated chained/single-pass
CPU comparison. Afterward, all **2,639** sources bound by its actual launch
were independently rehashed and matched. Both new files are outside that
live source roster.

## Scope established by the tests

The private bindings preserve the original collection loop, command and raw
audits, sensor reconstruction, renderer queries and evaluator code. The new
limits are 8,014 observations, 8,013 commands and 401,400 physics samples.
The native evaluator remains a script-side audit dependency, separate from
the controller and its public sensor inputs. Both collection and full raw
controller audit construct the extended controller.

- The complete 8,014-row decision stream is lossless, exclusive and bounded.
  Both writer and reader reject an additional row; the original shorter
  reader still rejects the longer population.
- The renderer session's cooperative dispatch applies the extended bound
  before primary capture and preserves the failure latch. Synthetic camera
  readbacks generate all 16,028 endpoints and audit them against 8,014
  capture records. Final-frame pixel, sample-index and paired-pose changes
  are rejected. No native renderer was executed in this test.
- The replay constructor admits the complete extended manifest and rejects
  excess population. Its population-only arrays are not valid sensor data.
  A separate persisted synthetic RGB/depth sample reconstructs and validates
  the actual public auxiliary packet at frame 8013, including its clock and
  pixel identities; over-limit and boolean frame identities are rejected.
- A complete synthetic 401,400-sample command population verifies all 8,013
  command endpoints and the ten-command terminal drain. Corrupting the final
  applied command is detected.
- The evaluator admits the longer negative trajectory and checks a late
  outbound quiet window. One sample above the speed threshold fails that
  window. The deliberately discontinuous synthetic path never becomes a
  round-trip success, and an extra physics sample is rejected.
- The original collector loop runs through every extended observation using
  a fake session and the actual extended mission. It remains nonterminal at
  the old deadline, expires at frame 8003, drains ten commands and persists
  the final observation at physical sample index 401399. The artifact roster
  includes the final auxiliary image. This is loop/mission integration,
  not a real controller, renderer or physics execution.

The collector's proposed output allowance is **28 GiB**, retaining the
original storage checks. This value is not a measured maximum, RAM bound or
completed resource admission. Full retained raster history, native sample
persistence, paired replay and complete raw audit still need an explicit
resource assessment before dispatch.

The separately prepared single-read auxiliary acquisition optimization is
not composed here. Acquisition behavior remains the original dual-camera
implementation.

## Next evidence required

Complete and authenticate the live 4,014-observation chained/single-pass
comparison. Then replay the extended controller on the authenticated native
prefix, changing only declared budget and implementation fields and stopping
at the first decision intervention before consuming a following observation.
Bind the resulting evidence and resource assessment into a fresh longer
native protocol and launcher. The subsequent native trial must independently
verify its physical prefix, complete return/retrace and all negative outcomes.

No independent-maze reliability, learned planning or memory advantage,
real-time execution, hardware qualification or completed goal is established
by this preparation. Earlier component details are in
`docs/go2_extended_return_budget_controller_preparation_2026-09-12.md` and
`docs/go2_extended_return_budget_preparation_2026-09-12.md`; their recording
and audit integration items are now implemented to the scope above.
