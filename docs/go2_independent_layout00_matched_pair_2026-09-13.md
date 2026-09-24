# First independent layout: fixed JEPA/reactive pair

Before either scene has executed, the following order is fixed for independent
layout 0 from the existing eight-layout inventory:

1. `frozen_reference`, model `seed_2026091001_full_jepa`.
2. `reactive`, no high-level model.

Both use `scripts/run_go2_stop_conditioned_independent_case_v1.py`, the same
layout specification, public mission, frozen low-level gait, sensor contracts,
8,000 navigation-tick allowance, stop-conditioned arrival rule and physical
round-trip evaluator. Each case collects a fresh closed-loop trajectory and
performs its raw sensor/controller audit once. The reactive arm is a comparison
of complete methods; it does not isolate JEPA training or forecast ranking.
Different commands may produce different physical trajectories and run lengths.

The existing JEPA queue (PID 3131637, creation time 1789253560.06) waits for
the stopping-rule trial to end operationally. The reactive queue (PID 3140904,
creation time 1789257654.51, tool session 74530) waits for that exact JEPA owner
to finish operationally, then executes the reactive case in the same process.
It does not require a successful JEPA round trip. Operational failure stops
the queue and preserves evidence. Neither case retries or overwrites a prior
attempt. Scene collection and auditing remain serial, and the existing runner
assesses hardware before starting each case.

The machine-readable queue receipt is
`go2_independent_layout00_reactive_queue_2026-09-13.json`. At queue creation,
neither independent case had a launch artifact and both queues were waiting.
This document is a prospective assignment, not an execution result.

Even a successful pair is only one independent maze. Reliability across
layouts, matched direct/supervised/JEPA training comparisons, predictive
planning and memory ablations, continuous execution with realistic sensing,
and bounded hardware evidence remain outstanding.
