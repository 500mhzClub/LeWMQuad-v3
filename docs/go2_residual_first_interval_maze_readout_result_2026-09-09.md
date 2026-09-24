# Residual first-interval maze 2: completed paired execution readout

The correction allowed one previously rejected command to execute, but did not
resolve navigation stagnation. Both executions failed to reach the goal or return
home. This is a comparison on an already used development layout, not evidence
of new-layout generalization or navigation qualification.

## Identity and completion

- Artifact root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_residual_first_interval_maze_readout_v1_attempt_001`.
- Result SHA-256: `aa1abf8ce0110dca271bc5f93fabf51a62bb41315fa5971afe16f87c25c59888`.
- Launch SHA-256: `82137678a68de81bd1d46b084361aba98404db534c3b7169495553bf634f1a8c`.
- Completed native result: `55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466`.
- Original learned cohort: `a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`.
- Readout session 26695 exited 0. The final result, launch binding and all 1,679
  source bindings were checked after completion.

The runner authenticated the completed raw audits, source/environment bindings,
same assigned model, same maze and execution settings, and the 23,900-sample
physical prefix through the first changed command at frame 463. The original
trajectory remains intact. Evaluator-only native poses were used for this
post-execution analysis, never as controller input. No model was trained or
loaded by the readout, and no new scene was executed.

## Paired outcomes

| Measure | Original learned maze 2 | Residual first-interval successor |
| --- | ---: | ---: |
| Paired observations | 514 | 3,014 |
| Completed commands | 513 | 3,013 |
| Simulated duration after first observation | 51.3 s | 301.3 s |
| Native XY path length | 1.503742253 m | 2.021193215 m |
| Closest native goal distance | 3.946789929 m | 3.844697442 m |
| Final native goal distance | 3.946789929 m | 3.847441560 m |
| Native open-edge crossings | 1 | 1 |
| Goal arrivals / verified round trips | 0 / 0 | 0 / 0 |
| Strict physical visibility | pass | pass |
| Raw sensor, model replay and command audits | pass | pass |
| Full iteration median, including receipt | 962.827517 ms | 1162.687231 ms |

The original execution stopped at frame 503 because no candidate satisfied the
phase, surface and nominal constraints. The successor reached its mission budget
at frame 3003. Among its 3,000 active choices, 2,643 were hold; there were no
forward actions. Maximum observed XY pose error was 0.004313192 m in both runs.
All measured full iterations exceeded 100 ms. These timings are from the actual
executions, not the separately measured receipt-copy/index optimization.

## The one executed correction

There was exactly one fallback intervention, at frame 463: right arc
`[0.16, 0, -0.45]`, completing its 100 ms command interval. Its correction used
only residuals from frames 455–462. The raw first-step XY error against actual
native body motion was 0.011743219 m; the corrected error was 0.009737254 m.
This single executed sample does not establish a calibrated error bound.

The corrected predicted minimum clearance was 0.450622582 m against the
unchanged 0.45 m nominal radius. The 0.623 mm numerical margin is not a physical
clearance certificate. No alternative unexecuted outcomes were inferred.

## Consequence for the next experiment

The roughly 0.102 m improvement in closest approach and longer execution do not
constitute useful maze completion. The next causal test is the separately frozen
hold-reconsideration replay: determine whether the same correction makes a
strictly higher-utility candidate feasible while retaining every existing gate.
It must stop at its first changed command; any subsequent physical outcome needs
a fresh native execution with its own prefix comparison. Tracking improvement
is being tested separately on maze 1.

The aggregate remains 21 completed, raw-audited native episodes and zero verified
round trips. The goal is active and unachieved.
