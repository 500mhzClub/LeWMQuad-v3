# Supervised adapter case: collection complete, raw audit pending

At 2026-09-10 13:02 UTC the full supervised-rollout case finished collection
with `MISSION_TICK_BUDGET_EXHAUSTED`. It recorded 3,014 observations, 3,013
completed commands and 151,400 physics samples. The observed mission remained
outbound, with no goal arrival and a final observed goal distance of
4.682274535840152 m. The worker terminal and full raw audit were still absent.
This is a provisional failure readout, not an additional audited episode.

Session 61388 verified the original batch launch and all 1,908 bound sources,
bound the completed collection result/physics/command/timing files, checked all
command endpoints and observation indices, and applied the existing native
evaluator to the full recorded trajectory. It found zero maze-edge crossings,
zero observed/native arrival windows and zero recorded native contact samples.
The native round-trip candidate check was false. Full sensor reconstruction,
controller replay and strict visibility auditing remain the worker's pending
work; none is implied by this provisional evaluator check.

Across the complete command tape there were 2,944 zero commands and 69 left
turns, with no forward or arc commands. This includes three zero warmups and
ten terminal zero commands. Thus 2,931 of the 3,000 navigation commands held
position. The first navigation turn at frame 3 was followed by 119 holds at
frames 4–122. There were 139 contiguous command runs in total.

All 3,000 recorded navigation iterations exceeded 100 ms. Their median
`iteration_with_receipt_wall_ms` was 2,754.8513305 ms, p95 3,967.9963159 ms,
and maximum 4,697.879044 ms. This is the collection's recorded iteration
scope, including receipt writing; it is not a controller-only benchmark or
an isolated measurement. The simulation pauses physics during computation.

Session 35222 examined only saved decisions 0–14, bound the complete compressed
decision stream and canonical consumed prefix, and recorded all twelve early
selection summaries. At inspected hold frames 4, 13 and 14, all six actions
passed the recorded phase, surface-intersection and eight-segment nominal
clearance checks. The original utility ranked hold first; no residual recovery
was active, and the target was an intermediate waypoint. At frame 14 the hold
utility was -0.004067977393876686 m; right turn was -0.004152912397042293 m,
left turn -0.0046093942177728095 m, and forward -0.04184926901677317 m.
Those inspected holds were score-selected, not forced by a geometry rejection.
This does not classify every later hold or establish that all physically
executed actions were safe.

This adds development evidence relevant to the already queued hold-reorientation
intervention. That intervention permits a lower-scored originally feasible turn
after repeated discretionary holds. Its native pilot remains queued on the
JEPA case. No policy was applied to these supervised observations, no model was
rerun in this diagnosis, and no changed command or subsequent candidate outcome
was simulated. A successful escape or navigation result remains unproven.

Bindings:

- Batch launch: `97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`.
- Provisional readout JSON: `162c5f43fade5297cf3f6fbbac3e5fe111cb3f3c04d068ed05db8de703589155`
  in `docs/go2_all_phase_adapter_full_supervised_maze02_provisional_readout_2026-09-10.json`.
- Early-hold diagnosis JSON: `582775ef57457c60c6a7cf9f403172581f6de690e47891e7a0b076ad1ab94da0`
  in `docs/go2_all_phase_adapter_full_supervised_maze02_early_hold_diagnosis_2026-09-10.json`.
- Original compressed decisions: `eb5a044d40a14f3cfd442ab33eb1d4a8d6c7913555a6b61231bd3027ee50c66c`.
- Canonical original 15-row prefix: `2167d3603e557cb9963093d3c1298927c0efafe560aeee3772ba0d0ce535cdd3`.

An initial readout command in session 17150 stopped at the artifact-root guard
because an episode subdirectory was passed as an attempt root. No report was
written. Session 61388 used the exact existing attempt root with episode-relative
bindings. No experiment was restarted and no evidence was changed.
