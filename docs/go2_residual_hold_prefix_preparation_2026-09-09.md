# Hold reconsideration candidate and prospective replay prepared

The closed residual maze2 diagnosis found 2,643 selected holds in 3,000 active
decisions. Its original fallback ran only once because it applies to no-action
selections. A feasible hold suppresses that fallback even when an alternative
could become feasible under the correction already used for scoring. This is
a hypothesis about the decision policy; a feasible alternative and better
physical outcome have not yet been established on actual replay or simulation.

Implemented ResidualHoldFeasibilityController as a separate successor retaining
the original observation, advance, mission, map, residual owners and no-action
fallback. Its selector reconsiders a feasible waypoint hold using the original
utility values. Another action must have strictly higher utility and pass the
corrected first-100ms XY geometry, all eight path segments, original and corrected
surface vetoes, and phase restrictions. Preserve raw predictions and residual
targets, later prediction points, yaw/contact forecasts and the nominal 0.45m
radius. No forced movement, hold timeout, cost change or clearance certification.

The runner requires the completed residual native result's supplied SHA-256,
all source/artifact/environment bindings, raw audit and its physical prefix
against original maze2. It directly reuses the prepared residual readout's
paired admission. A closed collection alone cannot admit execution. Original
negative outcomes remain admissible and are not recast as successes.

Replay begins at observation zero and stops at the first changed command,
original terminal, or 3,004-observation bound. It compares complete original
decisions, forecasts, observed map, mission and residual state. At intervention,
only the selection choice, explicit hold-feasibility receipt, selected action
and requested command may differ. Wait state, terminal and plan offset remain
identical. It checks public input immutability, unchanged model weights and
absent gradients, rejects truncated decision streams, and never reads the next
packet or decision after intervention. Actual changed-command outcomes still
require fresh physical simulation.

Verification:

- 19 policy/controller tests pass in 2.01s, session84891 exit0. They cover
  better corrected-feasible movement, all original/corrected vetoes, utility
  ties, existing no-action recovery and causal/original-evidence tampering.
- 28 replay tests pass in 4.10s, session31370 exit0. They cover complete state
  preservation, altered forecasts/utilities/receipts, first-change and original
  terminal boundaries, fixed maximum, truncated streams, input/model mutation,
  completed-audit admission and all seven inherited verifier input fields.
  Earlier 27-test run68806 passed before adding truncated-stream rejection.
- Source-preparation check1589 exit0 verifies 1,684 source/environment bindings
  using the existing native launch as inherited identity evidence. This is not
  completed-native admission or an actual replay preflight. No model loaded,
  native scene, experiment output or new physical result.

Reviewed source SHA-256 values:

| Path | SHA-256 |
| --- | --- |
| lewm/residual_hold_feasibility_development.py | 62dc7768b1b644b0c79c8471a3972543a5c1cf3ac4c90a154943b01b5ee6443c |
| lewm/residual_hold_feasibility_controller_development.py | c6fee017146e7fba66df2d0d5b8f728b773d4b3b61b12309e3d959187ac287a8 |
| lewm/residual_hold_prefix_development.py | 1f57ec627a34966ffa6a62cb6879edcd5b2c2109a549bb75434448302634d433 |
| scripts/replay_go2_residual_hold_prefix_v1.py | 68aa9dc34d4009f2992fb8486ef809be572bf27bed7478f55ed6d786f996ff27 |
| lewm/tests/test_residual_hold_feasibility_development.py | 0e705b04d7e9ce19bf40123e1fe2c414b71330236e6023abb77217e79c12c068 |
| lewm/tests/test_residual_hold_prefix_development.py | db67657c80160b13d270ab011f8814229684eb5f00fa79ebfa3b1adba9faf194 |
| docs/go2_residual_hold_prefix_v1_2026-09-09.md | 04893a665d5cb407272b518436ddfd13dc29dcb7114f2e899ecae42fcc2bc7f3 |

Hardware check76601 exit0: 16 physical/32 logical CPUs, all32 affinity, CPU3.4%,
72,575,741,952bytes available RAM, GPUs idle, 78,142,590,976bytes artifact free,
21,359,267,840bytes workspace free. Sole native worker2485788 remains live,
10,758,172,672bytes RSS; most recent CPU use5,689.52s. Parent2485335 remains
live at348.34 CPU seconds. No final native result/failure or raw audit yet.
One8GiB CPU replay can fit beside the existing32GiB native allowance, but its
completed-input prerequisite is pending. Refresh resources at actual launch.

Next command after admitting the actual native result, using the established
single-thread Python environment:

```text
scripts/replay_go2_residual_hold_prefix_v1.py --native-result-sha256 ACTUAL_FINAL_SHA
```

Output is exclusively go2_residual_hold_prefix_v1_attempt_001. The physical
queue remains completion of current residual audit, prepared tracking maze1,
then supervised layouts1–3 when storage admission passes. The supervised
78,383,153,152byte requirement currently exceeds available artifact space.
No deletion, frozen-source mutation, attempt restart or promotion occurred.
