# Causal subtrajectory development V1: completed data preparation

The fixed [derivation](go2_causal_subtrajectory_development_v1_2026-09-05.md)
completed and passed independent raw-label/command audit and full policy-only
tensor-loader checking. No new simulation or model fitting was performed.

| Original role | Layouts | Causal windows | Valid motion targets | Valid contact targets | Positive contact targets |
|---|---:|---:|---:|---:|---:|
| Train | 16 | 610 | 2,574 | 2,826 | 252 |
| Validation | 8 | 304 | 1,278 | 1,410 | 132 |

All120 original branches are represented. Of960 possible contexts,914 were
actually observed before contact; later contexts on stopped branches were not
invented. Each input contains four past RGB/body/control packets. At time0,
siblings share their audited canonical context; later windows use only their
own actual branch state. Every window retains its layout role.

There are7,312 horizon slots:4,236 are within their remaining fixed action plan,
and3,076 are outside it and masked out. All within-plan contact outcomes happen
to be observed in this corpus;384 are positive and censor motion. Non-contact
right-censoring is supported and tested synthetically, not an observed additional
failure type in these derived windows. Command padding is explicitly unknown,
not an executed zero command. Post-branch release is not included in the plan.

## Verification

The independent auditor checked every window and all7,312 horizon slots,
including3,753 branch-local policy-packet loads, prospective command reconstruction
against actual applied commands, role/context identity, contact censoring and
future timestamps. A separate quaternion-vector reference agrees with all3,852
valid relative-motion labels within4.45e-16. All891 mutually valid initial labels
agree with the old corpus within2.23e-16. These numerical checks validate label
semantics; they do not establish generalization.

The learning-facing loader subsequently loaded all914 actual windows and passed
shape, causality, finite-input, unknown-plan and censored-target checks. Its
observations contain only RGB/body/control; labels and future observations stay
under targets, and layout/role bookkeeping stays under metadata. This loader does
not read raw world-state artifacts. The combined explicit source suite now passes
**430 tests** across29 files, including19 new subtrajectory checks.

## Scientific consequence

This supplies temporal supervision missing from the initial-context-only models.
It does **not** increase independent layout count, provide every candidate's
outcome at a later state, train a temporal JEPA, or demonstrate replanning. The
earlier negative JEPA and limited supervised-rollout results remain unchanged.

Next implement and test a history/plan-mask model interface, then freeze a matched
temporal direct/supervised-rollout/JEPA comparison and layout-balanced training
schedule before fitting. Keep observation exposure, action costs and checkpoint
selection comparable; do not combine temporal coverage with an unreported
coefficient/seed search. Report action-only controls and scene-sensitive outcomes,
not just latent loss. Subsequent fresh physical trials must establish successive
sensor-only choices, then observed place/frontier memory, beacon discovery and
return. The arena-qualified gyro primitive is a candidate reorientation component,
not a substitute for visual clearance or relative odometry.

## Evidence identity

Root: `.generated/go2_causal_subtrajectory_development_v1_attempt_001`.
Derivation58578, raw audit72827 and tensor check86812 all terminated exit0.
Combined test session33338 terminated exit0 with430 passes in4.26seconds.

- Launch: `69b3037630636922974a6571e7b3be0421693c7e0fdd9a42f263f89f28ff3885`.
- Windows: `8c229c5e12b08dda2b98caac9f79b33f1b3b539a99013b802cd9ec4b3a2b6131`.
- Result: `a224d2b212b7a544a34e8f7d62c03bf831f9bdf795e1fd726b26fea5b5eee76b`.
- Raw audit: `a172810608cc96a8268b2470ce429dfcfdefff2e8be51c8becaaa2fc1e1f7b01`.
- Tensor check: `7da56176c4760c3a73766b554e9147d4d3ed29af1991a7f29c18089686bf1a61`.

Keep all bound source/specification and artifacts fixed. No sealed material was
accessed, frozen tracked source changed or completed experiment rerun. Final-goal
status remains active and unachieved; this is completed data-preparation progress.
