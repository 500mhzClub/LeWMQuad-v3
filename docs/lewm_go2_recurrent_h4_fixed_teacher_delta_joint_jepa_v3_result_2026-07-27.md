# Go2 recurrent H4 fixed-teacher delta joint-JEPA V3 result — 2026-07-27

## Outcome

- Terminal decision: `STOP_MAIN_POOL_RECURRENT_H4_FIXED_TEACHER_DELTA_JOINT_JEPA_V3_PROBE`.
- This was a clean scientific falsification, not an execution failure.
- The run completed exactly 1,000 optimizer updates and 16,000 training-sequence presentations in 491.676 active seconds.
- The preregistered minimum-mean-error rule chose update 750 / 12,000 presentations as the single selected checkpoint. It remained noncollapsed and was scientifically evaluable, but failed 11 pass gates and is not eligible for promotion or reuse.
- The independent read-only terminal review opened no checkpoint bytes, and the execution record reports zero retry/resume checkpoint inputs. No held-out, test, sealed, label, navigation, or deployment input was opened.

## What was tested

- Frozen source commit: `011b665` (`add fixed-teacher recurrent H4 delta JEPA V3`).
- V3 used the exact V1/V2 main-pool indexes, physical sequence order, seed, optimizer groups/rates, 1,000-update cap, and validation observations.
- The accepted N320 teacher encoder was frozen for the entire probe:
  - zero EMA updates;
  - identical initial/final teacher state SHA-256;
  - no target gradients or optimizer membership.
- Each optimizer update jointly trained the online visual encoder, ordered-history recurrence, and action predictor through one JEPA backward pass.
- The model predicted direct, non-recursively accumulated fixed-teacher future deltas relative to current frame `e2`.
- Exactly one zero-initialized final delta projection made update-0 output equal persistence without V2's scalar gates.
- The complete objective was:
  - fixed-teacher future-minus-`e2` delta regression, weight 1.0;
  - all-three-history-frame online-to-fixed-teacher alignment, weight 1.0.
- Absolute prediction, variance, cyclic wrong-action, persistence-hinge, and history-hinge training weights were zero. Wrong-action, hold, persistence, and history remained evaluation-only controls.
- A bound namespace loader prevented package initializers from expanding the reviewed runtime import graph. The exact five-file source closure was verified before input handling and persisted in receipts.

## Main metrics

| Update | H4 error / persistence | H4 action gap | H4 history gap | H4 persistence gap | H4 hold gap | Target / online rank |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | 0.0000 | approximately 0 | 0.0000 | 0.1750 / 0.2075 |
| 250 | 6.1480 | 0.1161 | -0.0650 | -5.1480 | -0.0579 | 0.1750 / 0.2023 |
| 500 | 6.2336 | 0.4692 | -0.1441 | -5.2336 | -0.7126 | 0.1750 / 0.2026 |
| 750 | **1.4388** | **0.0718** | **-0.0951** | **-0.4388** | **-0.0854** | 0.1750 / 0.2053 |
| 1,000 | 2.7916 | 0.2782 | -0.3250 | -1.7916 | -0.3465 | 0.1750 / 0.2009 |

- V3 solved V2's representation failure:
  - teacher rank was exactly 0.174981 at every observation, with zero metric drift;
  - online rank stayed near 0.20 at every observation;
  - all near-zero-variance fractions were zero.
- V3 reduced the registered normalized validation error:
  - selected mean H1-H4 error was 1.5469 times persistence, versus 2.5285 for selected V1;
  - selected H4 error was 1.4388, 36.9% lower than V1's 2.2802 and 34.1% lower than nominal but collapsed V2's 2.1820.
- The action signal emerged without an explicit synthetic ranking loss:
  - H4 action gap was +0.0718;
  - scene-bootstrap lower bound was +0.0393;
  - all eight scene families were positive.
- The model still did not learn usable successor dynamics:
  - every horizon remained worse than persistence;
  - H4 persistence gap was -0.4388 with bootstrap lower bound -0.7326;
  - only `open_obstacle_field` beat persistence at H4; `rough_local_dynamics` nearly tied it, and the other six had negative H4 persistence gaps;
  - ordered history was negative in all eight families;
  - the hold control was positive in only one family.
- Update 1,000 increased action sensitivity while worsening H4 error and history. Extending the same optimization therefore trades prediction accuracy for action separability rather than closing the dynamics gap.

## Interpretation

- V3 demonstrates that low-rank collapse was not inevitable, and its fixed-teacher geometry remained stable throughout.
- It also removes moving-target collapse as an explanation for the remaining failure. Under a stable teacher geometry, the recurrent history/direct-delta predictor still could not beat copying `e2`.
- Success on the open obstacle field and near-tie on rough local dynamics, contrasted with large errors on enclosed/composite mazes, is most consistent with a GRU spatial/temporal binding bottleneck. The positive evaluation-only action gaps rule out a total absence of action conditioning, but do not by themselves prove full nine-action identity, data sufficiency, or that the GRU is the sole cause.
- The recurrent history path is not helping partial observability: in every family, real ordered history was worse than at least one of the reset or reordered-history controls.
- A second seed, longer run, nearby loss weight, covariance patch, scalar gate, or V4 recurrent-H4 retry is not justified.

## Registered branch boundary and recommended next falsification

- Stop the recurrent-H4 latent-dynamics branch. Do not open, resume, reuse, extend, or promote any V1/V2/V3 checkpoint.
- Do not start navigation training or access held-out mazes from these results.
- Do not reopen the already closed local-correspondence, warp, cost-volume, rigid-BEV, or spatial-transport families.
- The recommended separately preregistered and authorized successor is one fresh, capped **RGB fixed-teacher dense spatiotemporal cross-attention H4 JEPA**:
  - retain all normalized `e0/e1/e2` patch tokens instead of compressing them through a GRU;
  - add 2D position, time, and past-action tokens;
  - use four horizon-specific patch-query sets conditioned on cumulative future actions, each directly cross-attending to the complete three-frame history;
  - emit direct zero-initialized `e2`-relative fixed-teacher deltas;
  - jointly train the online encoder and attention predictor with V3's teacher-delta and three-frame alignment losses only.
- This successor must contain no recurrent state, BEV, warp/offset field, cost volume, retrieval objective, transport template, EMA, variance term, action classifier, or synthetic ranking loss. It must use fresh N320 initialization, the same exact 16,000-presentation schedule and V3 pass gates, and no predecessor tensor input.
- Any such successor should be limited to exactly one attempt. Failure should close deterministic dense-H4 predictor architectures and force a target-state or uncertainty reformulation rather than another predictor variant.

## Terminal bindings

- Completion file SHA-256: `c2e89d629160e92619fc0b72c3a22ff6191440ba81de6a2659e55eb80b7fd7cb`.
- Completion content SHA-256: `ea9531f261ab1ab0ae4329ea58ce0983e7b10ec595ab0d452c50649db4ce75a4`.
- Result file SHA-256: `d025076a09dce02045eebccf442eee45ca969c2b214e7dba0fed8f9ab1d26b05`.
- Metrics file SHA-256: `f851873f6ba316233f7fcafeb5ae175a7410524f26e3a5b4b90eff21a93a1d4b`.
- Artifact file SHA-256: `2fe99e37b6ab91a4e36abb6b5823369fa79f378aa7ad7cb0e2896df19d5774db`.
- Access file SHA-256: `7f291b5c95f421f244fb8245ed916323ccce35cb4e53c2e4af441ba5c231bdab`.
- Reservation file SHA-256: `4d0a92cf38b67b20572268113490b15cbdb2dcb88712bfdd5684829f81d4ebfc`.
- Fixed teacher initial/final state SHA-256: `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`.
- Independent read-only terminal review: pass; exact counts, source closure, fixed-teacher identity, decision recomputation, cross-bindings, and all forbidden-zero fields reconciled without opening checkpoint bytes.
