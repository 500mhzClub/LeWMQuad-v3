# Tracking V1: precision-interface failure and incomplete executed coverage

## Outcome

The original V1 remains a terminal failed attempt. A completed read-only
diagnostic supports a numerical-interface mismatch as the immediate cause:
the recorded native quaternions have binary32-scale norm deviations, while
the coverage checker imposed a stricter boundary than the existing sensor
rotation helper. This does not establish every aspect of native validity.

A separate post-hoc numerical coverage readout, applying the existing sensor
rotation convention, finds **0/8 tapes satisfying all intended motion coverage**.
Correcting the representation mismatch therefore cannot turn this challenge
into a scientific pass. The recorded full schedules are not proof that the
requested displacement and turn angles were executed.

## Precision diagnosis

The adapter authenticated the exact retained launch, collection, base/stress
phase and failure records, the outside request/complete log/terminal, and all825
frozen source bindings. It then reconstructed all96 complete sensor streams
before opening native pose arrays. Artifact and source identities were checked
again after inspection. No observer inference, native simulation, checkpoint
loading, source export or original-artifact writing occurred.

Across182,800 native samples:

- All731,200 quaternion components are exactly representable as binary32, though
  stored as float64. The recorder explicitly widens `robot.get_quat()` values
  and reorders WXYZ to XYZW without renormalization.
- 765 samples exceed the frozen absolute norm tolerance1e-7, including48 during
  settling and13 at camera capture samples. Every tape has such deviations.
- Maximum absolute norm deviation is1.1591634696550557e-7. NumPy norm and an
  independent `math.hypot` calculation differ by at most2.22e-16, so the rejection
  is not explained by the diagnostic's norm-reduction implementation.
- Maximum raw-versus-normalized yaw difference is2.1727786791991832e-7rad,
  approximately0.00001245degrees. This is representation sensitivity, not
  observed tracker error or an uncertainty bound.
- Coordinate permutation cannot change a quaternion norm; changing XYZW/WXYZ
  labels alone cannot resolve this particular rejection.

Source chain:

- `scripts/whole_task_physics_session_development.py`, GeometryFreePhysicalSample:
  native quaternion widened to float64 and reordered into the recorded pose.
- `scripts/independent_tracking_snapshot_development.py`: acquired rows stacked
  and persisted, without quaternion normalization.
- `scripts/independent_tracking_native_contact_guard_development.py`: effective
  native configuration requires float32.
- `lewm/physical_execution_development.py`, rotation_xyzw: accepts norm deviation
  up to1e-5 and normalizes a bounded input before constructing a sensor rotation.
- `lewm/independent_tracking_coverage_development.py`: rejects above1e-7 and uses
  the unnormalized quaternion in its yaw expression.

Together these support a precision-contract mismatch rather than grossly
malformed orientation records. They do not identify the exact native arithmetic
instruction responsible, calibrate simulation accuracy, prove integration
stability, or replace the unfinished full sensor/physics audit.

Completed diagnostic54799: exit0,363.04s. Its initial new-adapter preflight83980
failed before sensor/native admission because the new identity calculation
omitted the frozen canonical newline. Only the new adapter was corrected;
a regression test covers that issue. This was not a second native attempt.

Full [diagnostic JSON](go2_independent_tracking_quaternion_diagnostic_2026-09-07.json),
SHA25644f7d6d5307320d2acca00178747693dbfe109179de008bb694e8f723b3897ff.

## Distinct post-hoc coverage, not original-attempt qualification

The new representation-aware helper reuses the existing sensor rotation
acceptance rule, normalizes only a private numerical view and retains the
original norm-gate failure. Original records and frozen source are unchanged.
The existing production coverage arithmetic and separately implemented numerical
reconstruction agree on that private view. Turn, translation, stop, clock,
sample-completeness and physical-stop criteria are unchanged.

This second readout reuses the exact completed diagnostic's sensor-admission
evidence and reauthenticates all raw receipt bindings, all96 sensor-stream
identities, phase/terminal records and825 sources before/after native analysis.
It does not rerun sensor transforms, complete the full raw sensor/physics audit,
score observer accuracy or establish predecessor nonidentity. Treat the numbers
below as descriptive post-hoc coverage, pending broader validation.

| Scene / support / direction | Approach m | Second translation m | Outward turn deg | Return turn deg |
| --- | ---: | ---: | ---: | ---: |
| Offset niche / nominal / left |0.2190|0.1966|125.57|-162.69|
| Offset niche / nominal / right |0.2190|0.1462|-164.29|128.83|
| Offset niche / lower friction / left |0.2008|0.1596|160.48|-180.56|
| Offset niche / lower friction / right |0.2008|0.1388|-180.87|160.48|
| Unequal baffles / nominal / left |0.1960|0.2086|126.89|-165.08|
| Unequal baffles / nominal / right |0.1960|0.1372|-163.00|126.78|
| Unequal baffles / lower friction / left |0.1970|0.1033|164.43|-180.76|
| Unequal baffles / lower friction / right |0.1970|0.1002|-180.55|163.97|

The unchanged requirements are at least0.20m for both translations, at least150deg
in each signed turn direction, and stopping over the complete last second.
All8 stopping checks pass. Only4/8 approach checks,1/8 second-translation checks,
and4/8 paired-turn checks pass; no tape satisfies all requirements. Four nominal
positive-direction turns are approximately126–129deg. Requested yaw integrals
are not a substitute for those measurements. These are two scene clusters with
four treatments each, not eight independent mazes or a reliability estimate.

Completed readout69712: exit0,9.98s. Full
[coverage JSON](go2_tracking_sensor_convention_coverage_2026-09-07.json),
SHA256a62f9f4d0e6998640c961a380f2bf9a5f55bd504a18ae259ddcbd32ec782e80d.

## Implementation and tests

New files only; no frozen implementation was edited:

- `lewm/tracking_quaternion_precision_diagnostic_development.py`
- `scripts/read_go2_independent_tracking_quaternion_diagnostic_v1.py`
- `lewm/representation_aware_tracking_coverage_development.py`
- `scripts/read_go2_tracking_sensor_convention_coverage_v1.py`

Sixteen diagnostic tests pass, including malformed data, no mutation, complete
clocks, capture/settle denominators, canonical identity and native-access ordering.
A synthetic float32 normalization example reproduces the incompatible gate;
it does not prove the precise native implementation cause. Seven coverage
tests pass: independent arithmetic, preserved no-motion failure, stop criteria,
missing-sample rejection and refusal to normalize grossly invalid inputs.
These23 synthetic tests are not scientific mission results. The post-hoc reader
itself has completed its real-data identity and numerical checks; those are not
an independently implemented complete reader or physical audit.

## Next iteration

1. Preserve both the terminal failure and these post-hoc results. Do not rerun
   the fixed recording or reinterpret0/8 intended coverage as a pass.
2. Complete a separately identified raw sensor/physics and paired accuracy
   analysis under an explicitly justified representation convention. Reuse
   recorded inputs; preserve all96 streams and partial/unavailable estimates.
   Do not silently monkeypatch the original frozen scorer or normalize away
   evidence. Reuse shared calculations while identifying independence limits.
3. Use error/availability and actual motion together to choose the next closed-
   loop execution change. Requested-command duration alone demonstrably does
   not ensure motion coverage. Address signed-turn response, translation and
   support-dependent dynamics with sensor feedback and stopping bounds; do not
   merely lengthen every tape or tune against a favorable frame subset.
4. Keep visually necessary learning-task design moving alongside execution.
   The existing RGB/JEPA negative comparisons and action shortcut remain; the
   numerical fix does not solve those scientific gaps.

The [long-term execution plan](go2_long_term_goal_execution_plan_2026-09-07.md)
remains the goal: actual novel-maze navigation, useful prediction/planning/memory,
independent-layout comparisons, timing and platform evidence. None is replaced
by this diagnostic or post-hoc coverage result.
