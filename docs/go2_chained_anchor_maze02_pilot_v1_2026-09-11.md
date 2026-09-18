# Prospective chained-anchor maze2 simulation

This development experiment asks whether reacquiring retained visual anchors
before the original short bridge expires improves subsequent closed-loop
navigation. It runs the original no-RGB JEPA model
`seed_2026091001_no_rgb_jepa`, state SHA-256
`fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f`,
on reused development layout 2. No-RGB denotes the predictive model input;
the original public RGB-D localization frontend remains active.

The sole controller change relative to the completed direct-flow tracking
simulation is `ChainedAnchorResidualController` and its bounded retained-anchor
association. The original planner, learned model, action set, floor registration,
memory, mission, safety checks, and 3000-tick budget remain unchanged. The
collector and raw auditor are the previously prepared and tested separate
chained-anchor sources. No sustained-turn, contact-scoring, or extended-budget
policy is adopted from the other queued diagnostics.

Prerequisites are the exact authenticated controller completion
`6b0c3298bb4df47aa59affc80d5dcb9236c1e4d9ae5cabd07550800c94d26425`,
its fixed original tracking simulation and completed waiter, and completion of
the existing diagnostic queue through the original contact-plus-flow waiter.
The latter's future result must bind its original launch
`3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72`.
Original process identities and boot are checked; a live prerequisite prevents
input admission and output creation. Scientific failures remain valid completed
diagnostics, but incomplete collection, failed integrity checks, or missing
terminal evidence cannot authorize a new run. No existing attempt is restarted.

The original model training admission is reused from the hash-bound, completely
audited tracking result. All assigned model and correction artifacts are
reverified by the unchanged assigned-model loader. This experiment does not
retrain the model or reconstruct the training corpus. The queued native outcome
is checked with its existing full completion authenticator, including its raw
artifact roster, physical prefix, and readout. Its prior model/admission
checks remain those of that authenticator.

Full queue completion reconstruction occurs at initial admission and again
before the new native result is finalized. Intermediate worker checks verify
the same bound sources, completed results, worker artifacts and receipts,
including proof that the full check was executed at admission. They do not
recursively reexecute all preceding queue audits each time a model is loaded.

The experiment creates one exclusive output root,
`go2_chained_anchor_maze02_pilot_v1_attempt_001`, and one fresh CPU physics
worker with fixed single-threaded OpenCV/BLAS and deterministic Torch settings.
It preserves the original ordered renderer, native binary and geometry bindings,
scene specification, public mission, robot, gains, friction, warmup, drains,
storage and memory guards, and raw artifact persistence. Physics pauses during
controller computation; wall-clock timings must be reported honestly and do
not constitute real-time operation.

The complete fresh candidate decision stream must reproduce the authenticated
replay through frame 853. All 854 public sensor packets and 43400 preceding
physics samples must match the original tracking simulation. Actual command
tapes must agree through the boundary, including 50 completed boundary-command
samples. The intervention changes the anchor evidence; both original and new
controllers request the same right turn at that frame. Following outcomes are
measured only in fresh simulation and are never inferred from recorded inputs.

Collection is followed by independent reconstruction of every raw controller
decision from public sensors and a fresh copy of the same model, full command
and contact checks, strict visibility checks, and original native traversal,
arrival, and return criteria. The completed report preserves failures and
reports actual contacts, progress, goal arrival, return, and computation time.
The source-preflight mode checks source/resource readiness without runtime
output, model loading, or admitting incomplete queue inputs. Full preflight and
execution require all prerequisite results.

This is one reused-layout diagnosis. It does not establish independent-layout
generalization, JEPA advantage, causal memory or planning benefit, real-time
control, or hardware readiness. It does not select the independent study policy
or grant sealed evaluation or real-robot execution authority.
