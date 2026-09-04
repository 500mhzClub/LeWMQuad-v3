# PHYSICAL_HANDOFF_STRATIFIED_GENERATOR_SUCCESSOR_V1

Development-only, outcome-observed generator successor to the completed V4
panel-adequacy result at `4e52b30bde4cd521b6bfc1d37d4d4744f7eede56`.

Each of the frozen four-family by sixteen-stratum streams receives a fresh,
deterministic sequence of at most 64 physical candidates.  A stream
stops only immediately after its fourth `QUALIFIED` state or after attempt 63.
It never stops after the first qualified state.  Candidate allocation is
teacher-only and cannot depend on fanout, ranker, or held-out outcomes.

The successor changes only candidate identity, seed allocation, and stream
depth.  Initial-boundary checks, semantic and behavioural snapshot contracts,
teacher routes and qualification, physical contact, deterministic panel
selection, 48/16 role assignment, exact reset, canonical ports, three target
contracts, the twelve-action bank, frozen encoder and frozen current-visual
ranker, metrics, gates, and handoff classifications remain frozen from V4.

Separately, the qualification-candidate implementation lifecycle destroys all
owned Scenes exactly once in reverse creation order, calls the public
`lewm_genesis.scene_builder.shutdown_genesis()` boundary, and lets the next
candidate build reinitialize Genesis from that candidate's frozen seed and
backend.  This lifecycle-only engineering correction changes no physical
value, render, step, evidence, formula, threshold, gate, disposition, or
model.  Its authority content digest is
`dd01d16f90709fd25034ab6bba0f2815498b2642e37a2c4dae7de6ef9e8e24a7`.

All 64 streams must reach four qualified states before any panel, encoder,
ranker, fanout, or held-out outcome is opened.  Otherwise the generator
terminal is published and downstream evidence remains absent.  No learned
model training is authorized.

`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/physical_handoff_stratified_generator_successor_v1_engineering_log.jsonl` is an external operator-owned concise log
used only after an adjudicated pre-panel implementation or persistence fault
requires full namespace deletion and deterministic restart.  This direct
runner never creates or appends it on generic exceptions; normal execution
leaves it absent and uses ordinary process stdout/stderr.
