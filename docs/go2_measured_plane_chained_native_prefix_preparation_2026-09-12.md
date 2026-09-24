# Chained measured-plane native prefix preparation

Development source preparation on 2026-09-12 Europe/London. No native launcher
or new scene was registered in this step. The existing controller replay waiter
remains responsible for producing an actual completed intervention boundary.

## Implementation

`scripts/measured_plane_chained_native_prefix_development.py` derives the
intervention index from the completed controller replay report. It does not
choose a frame or action. The candidate must be nonterminal and the replay must
have stopped at its first changed request or terminal status. A terminal-only
change is explicitly distinguished from a changed physical command.

Given source/artifact-authenticated completed inputs, the prefix comparison:

- Checks every physics array exactly through the preintervention boundary.
- Checks all completed command endpoints and unchanged earlier commands.
- Checks the actual 50-sample requested-command interval at the boundary.
- Reconstructs paired public packet identities and compares the complete
  original and candidate decisions with the saved prospective replay.
- Reconstructs each comparison to verify that no earlier command/terminal
  change has been skipped. It consumes no following actual controller or
  public observation and does not compare following physical outcomes.

The separate `prefix_availability` function accounts for early negative runs:
unreached observation, unissued command, incomplete command, or insufficient
physical samples. Count metadata alone always reports
`actual_physical_prefix_reconstructed=False`. Only the full raw comparison can
establish a reproduced prefix. A future native worker must preserve an early
negative result and cannot label that unavailable comparison as successful.

`scripts/measured_plane_chained_native_pipeline_development.py` prepares the
collector and raw audit by privately substituting the already queued candidate
controller constructor in the existing measured-plane native pipeline. The
original function code, closures, all other dependencies, physics loop, sensor
and visibility audits, evaluator, 4,000-tick budget and artifact enumeration
are retained. This was checked directly without launching a scene. Definition
metadata keeps the simulation, timing and qualification limits explicit.

## Validation

All 33 focused physical-prefix tests passed in 2.31 seconds (session 37815).
They cover dynamic boundaries, matching full physical/public/controller
prefixes, terminal-only changes, incomplete or corrupt command intervals,
corruption of either controller's decisions or any preintervention physical
channel, changed packet identities/clocks, missing/extra saved rows, earlier
interventions and honest accounting for early negative runs.

Two initial corruption fixtures accidentally shared Python objects between
the simulated native and saved-replay documents, so a mutation changed both.
The fixtures now use independent copies, as actual separate JSON files do.
No production equality check was relaxed. The preceding 25-test set and final
33-test set both passed after that fixture correction.

This is synthetic prefix/audit evidence, not proof that the new controller
will produce a valid boundary or a successful round trip. The completed replay
must still be source/artifact authenticated before any native input admission.
The future native launcher must check both raw artifact rosters before using
this helper and perform the full physical/sensor/controller audit afterward.

## Current work and next step

The original raw worker PID 2916239, creation 1789162190.15, remains live and
was observed advancing its read-only compressed decision stream to byte
381,947,904. Its parent result was still absent at the last check. That byte
position indicates progress in stream replay, not completion of all audit
stages or a reliable remaining-time estimate.

The nominal and reactive native waiters, full-history timing waiter, and
chained-controller replay waiter remain in their existing order. The new
native helper files do not modify any of those frozen source unions. They are
not yet frozen into an execution protocol.

After the queued controller replay is complete, inspect its actual boundary,
including any negative outcome. If it provides an admitted candidate boundary,
prepare and verify a separate native launcher with that exact result identity,
this physical-prefix check, complete raw audit and current resource admission.
Account honestly for an early negative or incomplete boundary. Preserve all
existing failures and never infer the changed command's outcome from old data.

Any eventual comparison claiming a world-model/planning/JEPA/memory advantage
must match the adopted perception across arms. The current nominal/reactive
queue and prepared independent roster use the previous measured-plane
perception; they are not automatically matched controls for this new combined
candidate. No independent population or advantage claim is made here.

## Current source identities

| Path | SHA-256 |
| --- | --- |
| `scripts/measured_plane_chained_native_prefix_development.py` | `301273e657fe117db67c2cdb192db794bf67476a1f583878f89ddabe83019bb3` |
| `scripts/measured_plane_chained_native_pipeline_development.py` | `257c18951ef034a8c3988666723ef6b4cca216947e147b74a040ceec50c0b793` |
| `lewm/tests/test_measured_plane_chained_native_prefix_development.py` | `cfedbe184bc133ded14688bc63bd5ca9f4f217fbdce1256f76d5bcb023947a1a` |
