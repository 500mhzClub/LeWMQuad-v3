# Temporal online adapter: completed recorded replay

The fixed [protocol](go2_temporal_online_adapter_replay_development_v1_2026-09-05.md)
completed with all three checks passing. No model was fitted and no proposed
command was physically executed in this replay.

- 1,824 proposed choices across 240 method/recorded-stream combinations.
- 4,080 comparable ensemble-member predictions matched the frozen offline
  predictions exactly (maximum absolute difference 0).
- 160 learned-method choices using noncanonical initial sibling images were
  explicitly excluded from prediction equality, not replaced with training RGB.
- 58 physical reference streams: maximum SO(3) orientation error 0.035727 rad;
  maximum projected heading error 0.018589 rad. Both are below the predeclared
  0.04-rad limits. These are ideal simulated sensors, not hardware drift bounds.
- 22 focused tracker/adapter tests passed, including repeated decisions, history
  tampering, cadence faults, coordinate transport and ensemble population.

Artifact root:
`.generated/go2_temporal_online_adapter_replay_development_v1_attempt_001`.
Result SHA-256:
`262fa3019560d2e5f1ed4e376b45d831bc97e3127d7adcf07b59eb274f9a8444`.
Launch SHA-256:
`1196a6eb25a1283f2cd80011913256787ee15990c0c2b4f87fcd24a22dfe0d45`.

This closes an input/coordinate integration check, not the policy-utility claim.
Recorded observations followed recorded actions, not the adapter's proposals.
Next execute the separately specified fresh successive-decision diagnostic;
preserve the negative JEPA comparison and every failed physical trial.
