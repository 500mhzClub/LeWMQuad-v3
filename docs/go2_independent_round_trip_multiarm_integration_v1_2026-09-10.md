# Independent round-trip multiarm collection and raw replay V1

This extends the separately checked independent-layout session/evaluator to
all four fixed comparison arms. It prepares collection and audit functions;
there is no entry point, population launcher, native execution or frozen
execution protocol. The existing six-model maze2 waiter owns the next scene.

The fixed assignments remain in
`go2_independent_round_trip_comparison_assignments_v1_2026-09-10.json`, SHA-256
`39754702e7c709e9e2e9e47d2b01ea748d8f94c40a3041f53e2cead6b1678bda`.
All four arms run all eight layouts in the existing balanced cyclic order.
There are 32 episodes and eight layout units. This work does not select a new
model, layout, controller budget, camera setup or seed.

New implementation:

- `lewm/independent_round_trip_multiarm_contract_development.py` binds the
  exact arm/case receipt, planning-map scope, residual configuration and model
  state. A reactive model, changed learned state or any model gradient fails.
- `scripts/independent_round_trip_multiarm_episode_development.py` creates
  the assigned fresh controller/model before opening a new episode directory
  or initializing Genesis, then retains the original sensing/physics loop.
  Learned and reactive cases use their respective command-tape role. The model
  is checked before collection and before writing the collection result.
- `scripts/independent_round_trip_multiarm_audit_development.py` rejects a
  mismatched collection receipt before raw access, creates a fresh assigned
  replay controller/model, and compares every complete decision reconstructed
  from raw public packets. It dispatches to the existing learned or reactive
  command audit and retains the original setup, contacts, geometry, renderer,
  auxiliary sensing, visibility, timing and private round-trip evaluation.
- `lewm/tests/test_independent_round_trip_multiarm_development.py` exercises
  orchestration and corruption rejection without native execution.

The learned acquisition loop is structurally identical to the original
independent residual loop except for the fixed role selection. The complete
raw packet/controller replay loop is structurally identical. The original
sensor, setup, stop, footprint, renderer and evaluator functions are shared
without patching their production globals. Original live, completed and
separately checked source files are unchanged.

Collection and audit take a fixed Case, robot geometry and previously admitted
correction record. They accept no arbitrary controller, checkpoint, model
object, layout index or episode-name override. Public mission coordinates
remain the only goal/layout information sent to controllers. The factory
passes no private graph, route or native pose.

The new treatment receipts describe configured model/residual capability.
Reactive replay records no high-level model and does not claim a model-state
check. Its future-pose feasibility gates differ from learned planning. The
current-pair JEPA arm retains contact, registration, localization, temporal
model, residual and mission histories; it is not fully memoryless.

Validation completed: process 72592, exit 0, 50 tests passed in 9.96s. All 32
fixed assignments exercised a synthetic 15-observation, 14-command episode
with three warmups, one navigation command and ten terminal zero commands,
then replayed it with a separate controller. All remained negative outcomes.
The tests retain the actual decision compression, timing receipts and native
command/slew audit, while replacing scene, controller, perception and physical
qualification boundaries with explicit synthetic fixtures. Separate tests
reject corrupted decisions, command roles, native command samples, timing,
treatment identities and model integrity. These are orchestration checks, not
raw reconstruction of a real newly collected scene or navigation evidence.

Still required: a launcher with full original inventory, fit/correction and
completed six-model development-result admission; fresh worker processes;
one-scene ownership and refreshed whole-population resources; matched startup
checks; complete scientific-failure retention and paired readout. Inspect the
queued expanded-model outcomes before freezing or executing this population.
Any correction must use existing development evidence, not outcomes on these
unexecuted layouts while retaining an unseen-layout claim. Timing and bounded
real-platform evidence remain part of the original goal.
