# Actuator gain pair V1: result

All sixteen fixed trials completed. The independent raw-data reduction passed
for every trial, including exact initial-state pairing and per-joint gain
readback before and after execution. No retries, threshold changes, source
changes during execution or discarded trials. See the
[preregistered protocol](go2_actuator_gain_pair_development_v1_2026-09-05.md).

| Measured outcome | Default gains 100/10 | Checkpoint gains 20/0.5 |
|---|---:|---:|
| Two crossings + usable final arrival (primary) | 0/8 | **8/8** |
| Two contact-free crossings | 5/8 | 8/8 |
| First usable arrival | 2/8 | 6/8 |
| Two usable arrivals | 0/8 | 6/8 |
| Sustained final 0.20-s arrival window | 0/8 | 8/8 |
| Trials with measured disallowed contact | 3/8 | 0/8 |

The local controller, gait weights, geometry, seed within each pair, command
limits, timing and physical thresholds were unchanged. Only the actuator gains
were explicitly restored to the pinned checkpoint configuration. These results
identify a consequential integration mismatch on the measured development
panel; adding a JEPA loss or larger predictor was not needed to address it.

The unchanged default-gain arm reproduced **all eight complete edge-result
records** from the prior factorial baseline exactly. Thus the improvement is
not a comparison against a newly weakened baseline. Native readbacks confirmed
the intended values throughout each trial, and initial pre-intervention pose,
twist and joints matched exactly within pairs. The full audit reconstructed
contacts from raw forces, causal inputs and commands, crossing/arrival results,
global clocks, image bindings and proper optical frames.

## What remains unresolved

Both corrected right-turn trials missed the first heading criterion:
0.4041104 rad versus the fixed 0.35-rad threshold. Their first-arrival speed and
angular speed passed, and both continued successfully to a sustained final
arrival. This reinforces the need to validate arrival proxies against actual
continuation. Do not relax the threshold retrospectively or claim every local
handoff contract is satisfied.

Corrected two-edge active times were 6.0 s for straight, 6.4 s for offset,
8.8 s for left turns and 9.0 s for right turns (plus 1.5 s settling). Width pairs
have identical corrected trajectories and are not independent replications.
The eight cases are four development motifs at two widths, not independently
sampled unseen mazes. No maze-level confidence interval or transfer success rate
is justified. The next edge is a second boundary within a corridor, not another
junction. There is no demonstrated recovery from dead ends or place ambiguity.

The controller still reads privileged pose and velocity. RGB is recorded but
does not drive actions. This is not a JEPA, visual navigation, sensor-fusion or
hardware result. Restored gains do not establish all training/runtime physics
settings are equal: timing/substeps and some scene settings still differ. The
effective integrated reference now works on this panel despite those remaining
differences; characterize them as needed, rather than claiming complete parity.

## Retained artifacts and implementation

Output root:
`.generated/go2_actuator_gain_pair_development_v1_attempt_001/`.
Result SHA-256:
`3c2412e40f1c3cbb33ffeb37b2b9017ab76a8dc1fd6a04aa8931d5c4bff70b20`.

Implementation: [explicit gain adapter](../lewm/actuator_gain_development.py),
[paired runner](../scripts/run_go2_actuator_gain_pair_development_v1.py),
[raw audit](../scripts/audit_go2_actuator_gain_pair_development_v1.py).
The complete explicit focused suite passes **279 tests**. Existing tracked
frozen code remains unchanged. The correction is implemented in the new
development execution path, not silently patched into historical experiments.

## Next decision

Use checkpoint-matched, read-back-verified actuator gains in the new development
reference. Do not rerun or rewrite completed studies. Proceed to the
[RGB/multisensor execution next steps](go2_rgb_multisensor_execution_next_steps_2026-09-05.md):
broader local geometry/continuation evidence, causal sensor capture with strict
privileged-state separation, and a strong direct policy before a matched JEPA
comparison. The ultimate scientific goal remains open.
