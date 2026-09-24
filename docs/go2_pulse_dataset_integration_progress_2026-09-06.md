# Pulse-timed dataset integration: actual materialization, still no training

The new dataset interface joins all185 bound recorded pulse windows to their
native target rows and actual RGB/body/control tensors. The complete diagnostic
materializes185 windows,917 motion targets,917 contact targets and917 future
images. No histories are missing in this corpus; all contact labels are negative.
No optimizer step or model fitting occurs. This is necessary training plumbing,
not a learned-policy or predictive-performance result.

## Implemented and checked

`lewm/pulse_timed_dataset_development.py` validates one-to-one pulse identities,
command-duration indices, partial target clocks and independent future-image,
motion and contact masks. It keeps missing-history windows in the declared
index but excludes them explicitly from materialization/sampling. A declared
layout with no eligible samples is not silently discarded. Returned targets
are copies; mutating a batch cannot change subsequent labels.

Layout identity and role are explicit publisher-bound metadata. One layout
cannot occur in multiple train/selection/development-evaluation roles. The
interface does not certify geometric independence from arbitrary layout names.
The recorded diagnostic checks common four-wall geometry and conservatively
groups all three trajectories into one train-role layout. Appearance, start
and friction changes do not create independent maze layouts. There are zero
selection layouts and zero evaluation layouts; no sealed role is accessed.

The deterministic sampler cycles layouts, then action-duration cells, then
examples within each cell. It does not condition on future observations,
motion values, contact labels or mission success. All three model arms receive
the same12-update/6-sample schedule, seed2026090711. This schedule is recorded
but not executed as training. Strict scheduling rejects missing cells rather
than fabricating examples. Layout/action balancing is not state/support balance.

| Action-duration cell | Existing examples |
| --- | ---: |
| Forward2ticks |2|
| Forward5ticks |53|
| Positive yaw2ticks |13|
| Positive yaw5ticks |39|
| Negative yaw2ticks |13|
| Negative yaw5ticks |65|

The72 scheduled draws give12 draws per action cell. Consequently the two
short-forward examples are repeatedly reused; balanced exposure does not
increase their independent information. Low-friction positive-turn/short-
forward coverage remains absent within this common layout.

Fourteen new dataset tests plus fourteen existing matched-loss tests pass,
including actual recorded/native-materializer equality, mutation isolation,
missing-history/cell accounting, split leakage rejection and target-independent
schedules. Full206-file regression: **2,611 passed in207.05s**, handle59585 exit0.

## Bound evidence

Diagnostic `.generated/go2_pulse_dataset_diagnostic_v1_attempt_001` completed
with exit0, handle30275, after full source/input/raw-artifact verification.
Its [protocol](go2_pulse_dataset_diagnostic_v1_2026-09-06.md) and dataset source,
diagnostic and tests are frozen by the launch.

- Launch SHA256: `6b21ead70f5bf98291ccc57640859843bfdfd6ac48ff1ffc75012862eff6b31e`.
- Result SHA256: `293870ecbb04615029aef389a76fcc4c62477f0d1ffbb8ca2b649664a48cc02b`.
- Schedule-file SHA256: `d6bd0ce92bfe31963abd92c215fe1908876ea6e006aac84dca582d1da93dad19`.
- Common schedule identity: `27172bbfa8eda762990df20f357d28c1dcbe2cea67f8827396ba0b77a1d4ab96`.

## Next toward a learned navigation system

Implement the actual optimizer/EMA/checkpoint/evaluation runner using this
interface and matched schedules. A bounded training-plumbing pilot on these
room data must be labelled development-only, without pretending it measures
new-layout generalization or collision-risk discrimination. Do not wait for
perfect room-return control before checking the learning pipeline, but do not
substitute this correlated corpus for the independent scientific study.

Prospectively collect multiple connected-maze geometries, body histories and
supported command-duration cells, including the missing support/action cases
and properly censored obstacle/contact outcomes. Freeze actual layout roles
and multiple training seeds before fitting/selection. Compare direct,
supervised-rollout and JEPA task-relevant predictions on matched sensor/data
exposure, then separate online-rollout and memory contributions in navigation.
Keep RGB/history/action ablations, usable uncertainty, realistic sensing and
timing, physical backtracking and bounded hardware in the final goal.
