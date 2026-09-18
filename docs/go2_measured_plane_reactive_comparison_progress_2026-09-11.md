# Measured-plane reactive comparison: verified replay and registered native run

## Completed sensor replay

`go2_measured_plane_reactive_prefix_v1_attempt_001` completed successfully and
its ended-owner result, source bindings, artifact roster, all saved decisions and
report were subsequently verified by
`scripts/reactive_measured_plane_native_prefix_development.py::completed_prefix`.

- Launch SHA-256:
  `ed9e51645c2e4176c46c4f330b1a7513b1b1e22afc8408114d8ac1ffc8b8cc5a`.
- Result SHA-256:
  `87defc11bf27970855e5b2c1ce17f6839ffed6d8641fa1cc998e2afedab9176b`.
- Original owner: PID 2920772, creation time 1789165065.36; process ended.
- Bound sources: 2,502. Four complete observations, frames 0–3.
- Actual model forward calls: learned 1, reactive 0. Assigned learned model
  unchanged; reactive controller instantiated neither model nor residual.
- First changed command: frame 3. Learned left arc `[0.16, 0, 0.45]`; reactive
  forward `[0.2, 0, 0]`. Both nonterminal, both using the same observed waypoint.
- Complete visual, floor, map, mission, goal-distance and auxiliary-floor
  receipts matched without normalization. Original learned decisions reproduced
  completely. All public packets remained unchanged.
- No frame-4 observation was consumed. No changed reactive command was physically
  executed in this replay, and no navigation outcome was inferred.

The nominal predictive prefix independently chose forward at frame 3 too. That
single shared action does not make nominal prediction and reactive control the
same method, or establish their later trajectories or relative performance.

## Registered native sequence

All roots below are under the existing RecoveryStorage development artifact
directory. At the direct process inspection following reactive registration,
the exact learned parent and both waiters were live on boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. That boot matched each bound launch file.
Parent PID, creation time and complete command were checked against each launch.

1. Learned measured-plane run:
   `go2_measured_plane_dispatch_recovery_v1_attempt_001`.
   Parent PID 2916106, creation time 1789162140.42. Launch
   `93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`.
   Latest complete timing row at inspection: tick 2,518. No result or failure
   file yet. Collection and complete raw audit remain unfinished.
2. Nominal predictive waiter:
   `go2_nominal_measured_plane_native_wait_v1_attempt_001`.
   PID 2919853, creation time 1789164727.75. Launch
   `a31dc3ecde7c7e570d6b499ee783381e55eb1d87a97f702bcc65794458292994`.
   Bound sources: 2,580. Native child root was still absent.
3. Reactive waiter:
   `go2_reactive_measured_plane_native_wait_v1_attempt_001`.
   PID 2922235, creation time 1789165929.81. Launch
   `b896918786c965a1b6e7d70aa796ca26f0983917cc40f26623327d2402b0150e`.
   Bound sources: 2,595. Its first event recorded the exact nominal waiter live
   at 2026-09-11 22:32:14 UTC. Native child root was still absent.

Each native comparator requires its predecessor to finish with a complete raw
audit. A scientific negative is admissible; an execution or audit failure is
preserved without automatically dispatching the dependent scene. Each waiter
retries transient observations of the same owner without treating them as
completion or restarting a job. The one-scene guard remains in force.

The reactive native pipeline loads no high-level world model. It retains the
shared low-level locomotion policy and the same scene, sensing, perception,
persistent observation map, mission, 4,000-tick budget and physical evaluation.
It is a whole-method comparison; future feasibility gates and action selection
differ from the predictive arms. Physics still pauses during computation.

## Validation and remaining work

Focused new tests passed: reactive replay 33; physical pipeline/prefix 16;
native launcher/input admission 23; final waiter 11 (83 total). Final waiter
source/resource preflight passed with 2,595 sources and the nominal owner live.
These tests include synthetic worker execution and negative cases, not a
completed reactive native scene. The actual recorded sensor replay above is
separate runtime evidence.

Next inspect the exact learned parent and its eventual full result. Allow the
registered nominal and reactive experiments to run sequentially; do not edit
their bound sources, restart them on observation timeouts or infer outcomes from
growing decision files. Compare complete audited progress, round-trip outcome,
physical retracing, contacts, sensing failures and timings once all arms finish.
Independent-layout replication, JEPA/supervised controls, isolated planning and
memory effects, realistic timing and hardware evidence remain outstanding. The
overall navigation goal is active and incomplete.
