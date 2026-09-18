# Route-loss diagnostic V2: explicit JSON identity restoration

V1 failed at the first map observation, before producing map results: JSON
records have an identity list, whereas the unchanged current-pose contract
requires a tuple. Preserve its launch, source and terminal failure exactly.
V2 uses a fresh root `go2_commitment_pose_route_loss_v2_attempt_001`, binds those
failure identities, and adds one explicit decoding adapter. The adapter accepts
only a three-integer JSON list equal to the current public packet's identity,
copies the evidence, and converts that field alone to a tuple. It changes no
identity value, pose value, witness, clock, acceptance gate or command.

All map reconstruction, proposal/candidate checks, closed-start inflation
diagnostics, first-hit voxel height intervals, benchmarks, accounting and
resource limits remain as defined in the [V1 protocol](go2_commitment_pose_route_loss_v1_2026-09-08.md).
V1 sources are included unchanged in V2's source bindings. This is a corrected
post-outcome decoding diagnostic, not a new native attempt or a recovery of a
failed observer/controller. Six focused adapter tests require an unchanged JSON
roundtrip, independent copied values, and rejection of mismatched, Boolean,
floating, short or already-tuple identities. Perform a bounded eight-frame
preflight replay on both exact native inputs before creating the V2 output root.
All frozen native outcomes and surface vetoes remain unchanged.
