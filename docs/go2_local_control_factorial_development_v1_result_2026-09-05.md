# Local control factorial V1: result and next decision

All 32 fixed development trials completed; raw-artifact audit PASS. The
[preregistered design](go2_local_control_factorial_development_v1_2026-09-05.md)
and executed source remain unchanged. No trial was retried or discarded.

| Controller | First usable arrival | Two contact-free crossings | Two crossings + final usable arrival | Trials with contact |
|---|---:|---:|---:|---:|
| Baseline | 2/8 | 5/8 | 0/8 | 3/8 |
| Prealignment | 1/8 | 1/8 | 0/8 | 6/8 |
| Arrival feedback | 0/8 | 4/8 | 0/8 | 4/8 |
| Combined | 0/8 | 4/8 | 3/8 | 2/8 |

No arm achieved two usable arrivals. **No trial achieved the sustained final
0.20-second arrival window.** The three combined-arm endpoint successes
(both offset widths and wide left turn) occurred at the deadline, not after
demonstrated stable stopping. Their first-arrival proxies failed, but actual
continuation succeeded under the declared task endpoint. Thus the first proxy
is neither necessary nor sufficient for this particular next traversal: the
baseline's two successful first arrivals did not yield a successful full task.
Do not promote the combined controller as reliable on the basis of 3/8.

Prealignment was not a uniformly beneficial intervention. It reduced two-crossing
completion and increased measured contacts on this panel. Arrival feedback
frequently exhausted its deadline (11 edges for feedback alone, 12 combined).
These are correlated development geometries, not 32 independent mazes. The
second boundary is within the same straight exit corridor, not a second junction.
The controller uses privileged simulated pose and velocity; RGB is recorded at
boundaries, not used to choose actions. No JEPA or sensor-fusion benefit is tested.

## Audit and reproducibility

The [auditor](../scripts/audit_go2_local_control_factorial_development_v1.py)
recomputed every native force/contact flag using a scalar reference, every
edge endpoint and crossing, command limits and slew, and every decision from
its available observation prefix. All four arms have exactly equal 750-sample
settling prefixes for each case. Global 2-ms timestamps and command continuity
show no reset between edges. Bound source/gait/artifacts and RGB pixels and
proper optical frames agree. This is a measurement audit, not an independent
physical replication.

A separate read-only comparison found all eight baseline first-edge arrivals,
shared checks and stop reasons exactly equal to the preceding eight-case study.
The baseline formula also passes literal-formula unit tests. This supports the
implementation bridge without treating the reused geometries as unseen data.

Artifacts are retained under
`.generated/go2_local_control_factorial_development_v1_attempt_001/`.
Result SHA-256:
`c9e1ad69a3bb36a92eb61f28592414ae35fc0f68b95304b71c13e275e8c986e3`.
The focused suite has 260 passing tests including seven audit corruption controls.
All tracked frozen files remain unchanged.

## A more fundamental execution-interface finding

Source inspection found that the checkpoint configuration specifies joint
proportional/derivative gains of **20/0.5**, and the upstream training environment
explicitly installs them. The navigation scene/adapter does not install those
gains. A fresh native readback, without stepping physics, measured **100/10** on
all 12 actuated joints. These are Genesis defaults, not the checkpoint's gains.
The checkpoint/configuration hashes match the platform manifest. Evidence:
`.generated/go2_actuator_identity_development_v1_attempt_001/result.json`,
produced by [the readback script](../scripts/inspect_go2_actuator_identity_development_v1.py).

The relevant inspected sources are `lewm_genesis/lewm_genesis/scene_builder.py`,
`rollout.py`, the explicit local upstream `go2_env.py`, and installed Genesis
`utils/geom.py` and `utils/urdf.py`. The platform's earlier policy-contract checker
uses the upstream training environment, not the navigation scene. Therefore its
validation label does not establish the integrated adapter's actuator identity.

This is a confirmed parameter mismatch, **not yet proof that correcting it
improves navigation**. The current study is still interpretable for its actual
plant: all arms shared those gains, and forces and trajectories are valid.
Do not erase or retrospectively relabel its negatives. Training and navigation
also differ in physics step/substeps and some scene settings; gain restoration
alone cannot claim full training-environment parity or hardware qualification.

Next: a separately identified, fixed matched comparison of default versus
checkpoint gains under the unchanged baseline controller and two-edge task.
Read back actual gains before and after execution and bind initial physical
state before the intervention. No controller search, gait retraining, changed
collision thresholds or novel-maze claim belongs to that comparison.
