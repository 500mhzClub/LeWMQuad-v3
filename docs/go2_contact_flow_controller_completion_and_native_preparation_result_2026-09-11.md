# Contact tracking recovery: completed replay and prepared physics test

The full-controller replay completed with result
`a4404e737a29f2499ee5313fcb116bc7a980f6fff5c2f75122ba6be6df437560`.
It reconstructed 562 original decisions using two fresh copies of the original
full supervised-rollout model. The candidate matched all 561 earlier decisions
and 558 earlier forecasts. At frame 561 the original controller stopped for
missing visual evidence; the candidate recovered current visual evidence and
requested a left turn `[0, 0, 0.45]`. Model state remained unchanged. No following
observation was consumed and the changed command was not executed.

Completion verification rechecked all saved comparisons and all 562 actual
public sensor-packet fingerprints, original completed-worker input bindings,
source bindings and replay artifacts. It did not rerun the neural replay or
full training ancestry. The verification is recorded in
`go2_contact_anchored_direct_flow_controller_completion_2026-09-11.json`, SHA-256
`1b75c5f714dbeadfd0cdf05b770899f963f67baba52d1feb1ddf98704af2e6bd`.

The input module passed 25 focused tests and checked its 2,200 source bindings.
It rejected the live original sustained-turn waiter before accessing runtime
inputs. Its preparation record is
`go2_direct_flow_commitment_contact_native_inputs_preparation_2026-09-11.json`,
SHA-256 `4cca1b1448e9e07d2b4cf5e121a75cfc40440bb7f9b9f9197ee25d36d401ce37`.

The new physics launcher, existing candidate collector and physical-prefix
checks passed 106 focused tests. Source preflight checked 2,204 bindings and
available hardware without creating an output. It retains the original model,
layout, sensors, 3,000-tick budget and strict success criteria. A fresh run must
match the actual preintervention trajectory and execute the recovered command
before any outcome can be credited. See the new launcher's protocol and
`go2_direct_flow_commitment_contact_native_launcher_preparation_2026-09-11.json`.

The original tracking child remained live during preparation, with its CPU and
disk counters advancing and no physics output yet. The original budget and
sustained-turn waiters remained live behind it. Preserve their order. The
contact-plus-flow pilot has not launched; its automatic waiter is not prepared.
Next, prepare a waiter bound to the completed raw replay and the original
sustained-turn waiter, admitting the complete existing chain before one fresh
physics attempt. Keep failures and all frozen source definitions intact.

There is still no verified round trip, independent-layout benefit, real-time
qualification or physical-platform evidence from this work.
