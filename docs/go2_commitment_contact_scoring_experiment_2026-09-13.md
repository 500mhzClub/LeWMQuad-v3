# Candidate experiment: contact score at the commitment horizon

The current controller scores 100 ms progress against 800 ms contact. Recorded
JEPA turning and supervised holding decisions motivate a separate experiment
that uses the existing 100 ms contact prediction with the 100 ms progress value.
The coefficient remains 1.2 m. No forecast, residual correction, candidate command,
phase allowance, surface check or 800 ms nominal path check changes.

`lewm/commitment_contact_controller_development.py` implements the intervention
for both causal intermediate-waypoint and causal exact-goal scores, composed
with the current stop-conditioned controller. Other score contracts and nominal
clearance reentry retain their existing behavior. Existing baseline runners and
assignments remain unchanged. The dedicated
`scripts/run_go2_commitment_contact_independent_case_v1.py` reuses the existing
one-case runner with a distinct `commitment_contact` mode and output name.
`scripts/commitment_contact_independent_pipeline_development.py` uses the same
independent collector, physical initializer and complete raw audit with the
new controller. No native intervention case has executed at preparation time.

The focused test file is
`lewm/tests/test_commitment_contact_controller_development.py`: eight tests pass
in 1.88 seconds. These cover action changes under the new horizon, unchanged
forecasts and causal progress, preserved later-path/surface/phase vetoes,
first-interval contact penalties, no-feasible-action stopping and fresh
controller composition.
Two additional behavioral integration tests exercise the new selector through
both collection and replay dispatch; both pass in 1.98 seconds. They use small
in-memory stand-ins for acquisition and do not execute a native scene.

| Recorded example | Existing choice | Candidate choice |
| --- | --- | --- |
| JEPA decision 500, intermediate waypoint | Right turn | Right arc |
| JEPA decision 3400, exact home goal | Right turn | Forward |
| Supervised rollout decision 200, intermediate waypoint | Hold | Forward |

All three retain the original forecasts, residual and geometry evidence and
have six feasible candidates. The detailed values are in
`go2_commitment_contact_recorded_score_examples_2026-09-13.json`. These examples
run only the new scoring helper: no checkpoint load, complete controller replay
or alternative physical trajectory was performed. They establish neither
navigation improvement nor real-time performance.

The prospective pair applies the same intervention to layout 0 in this order:

1. `seed_2026091001_full_jepa`.
2. `seed_2026091001_full_supervised_rollout`.

Both retain the original sensing, first-seed fitted model, 8,000-decision budget,
stopping rule and physical evaluator. The pair runs after the remaining
original supervised, direct and nominal cases. Queue receipts are
`go2_independent_layout00_commitment_contact_jepa_queue_2026-09-13.json` and
`go2_independent_layout00_commitment_contact_supervised_queue_2026-09-13.json`.
Each starts after its predecessor finishes operationally, including negative
scientific outcomes. Operational failure stops the chain; there is no automatic
retry. Hardware is assessed by the existing runner at actual launch, and native
collection/audit remain serial. Each output uses
`go2_stop_conditioned_independent_00_commitment_contact_<model>_v1_attempt_001`
under the existing navigation development artifact base.

This is a development hypothesis informed by inspected baseline decisions,
not a new blind evaluation. Compare the intervention with each model's original
case, and report physical navigation and strict sensor validation separately.
All these runs concern one maze replicate. Do not tune a separate coefficient
for each model or reinterpret recorded examples as successful navigation.
Shorter contact scoring can miss later risks; preserving nominal path checks
does not certify articulated motion or calibrated uncertainty. Prediction-off,
memory, realistic timing/sensing and hardware evidence remain separate work.
