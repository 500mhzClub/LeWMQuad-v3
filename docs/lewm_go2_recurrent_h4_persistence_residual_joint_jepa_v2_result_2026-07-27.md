# Go2 recurrent H4 persistence-residual joint-JEPA V2 result — 2026-07-27

## Outcome

- Terminal decision: `STOP_MAIN_POOL_RECURRENT_H4_PERSISTENCE_RESIDUAL_JOINT_JEPA_V2_PROBE`.
- This was a clean scientific falsification, not an execution failure.
- The run completed exactly 1,000 optimizer updates and 16,000 training-sequence presentations in 534.967 active seconds.
- No trained checkpoint was eligible: online effective rank was already below the registered 0.10 floor at update 250, and both online and target representations were below it from update 500 onward.
- The terminal result therefore selected no checkpoint. The four written checkpoints are terminal diagnostic artifacts and are not authorized for opening, reuse, resume, or promotion.
- No held-out, test, sealed, label, navigation, or deployment input was opened.

## What was tested

- Frozen source commit: `622eb97` (`add persistence-anchored recurrent H4 JEPA V2`).
- V2 retained the exact V1 main-pool train/validation indexes, N320 encoder-only initialization, seed, optimizer, EMA schedule, 1,000-update cap, and validation observations.
- The online encoder, ordered-history module, and action predictor were still trained jointly through one JEPA objective and one backward pass.
- V2 changed the failed V1 prediction mechanism:
  - the current online `e2` spatial tokens were the exact identity anchor;
  - ordered history could add only a zero-gated learned correction;
  - future actions accumulated zero-gated learned residuals from that anchor;
  - a persistence hinge required the real prediction to beat target-encoder persistence by 10%;
  - a detached gate-off-history hinge required real ordered history to improve prediction by 3% of persistence distance.
- The V1 cyclic wrong-action contrast and all validation counterfactuals remained active.
- The complete five-file execution-source closure was SHA-256-bound before input handling and recorded in both reservation and artifact receipts.

## Main metrics

| Update | H4 error / persistence | H4 wrong-action gap | H4 history gap | H4 persistence gap | Target / online rank |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | 0.0000 | approximately 0 | 0.1750 / 0.2075 |
| 250 | 3.1692 | 0.0388 | -0.0343 | -2.1692 | 0.1217 / **0.0869** |
| 500 | 3.1640 | 1.5293 | -0.0314 | -2.1640 | **0.0812 / 0.0609** |
| 750 | 2.1820 | 4.0917 | -0.0390 | -1.1820 | **0.0618 / 0.0504** |
| 1,000 | 3.1023 | 4.0507 | -0.0867 | -2.1026 | **0.0512 / 0.0451** |

- Initialization behaved exactly as designed: with both residual gates at zero, every future prediction equalled current-latent persistence.
- Training immediately moved into correlated low-rank collapse:
  - online rank failed at the first trained observation;
  - target rank followed it below threshold by update 500;
  - near-zero-variance fractions mostly remained zero, showing that dimensions became redundant rather than simply becoming constant.
- The existing per-dimension variance floor did not detect or prevent that covariance/rank collapse.
- Raw H4 action separation became very large and was positive in all eight families, but it was not useful prediction:
  - the correct-action prediction remained worse than persistence at every trained observation;
  - the H4 hold gap was -0.1124 at the nominally best update 750;
  - every trained observation was scientifically ineligible because of collapse.
- Ordered history remained harmful in all eight families at every trained observation.
- The nominal update-750 H4 error was slightly lower than V1's selected 2.2802, and its history gap was less negative than V1's -0.1335, but those values cannot be promoted because V1 remained noncollapsed while V2 did not.
- Terminal raw prediction loss fell to 0.0705 despite poor normalized validation prediction. The auxiliary persistence and history losses also became small. This is consistent with the encoder and EMA target co-adapting into an easy correlated low-rank shortcut rather than learning useful dynamics.

## Interpretation

- V2 falsifies unrestricted joint training of the persistence-residual/history-hinge mechanism under the existing variance-only anti-collapse objective.
- It does not show that the main-pool data or accepted encoder features are inadequate. The failure is localized: the new identity/residual route opened a representation-geometry shortcut that V1 did not take.
- The large wrong-action gap is not sufficient evidence of a world model. An action code can be separable while its correct prediction is still worse than copying the current visual state.
- More updates, a second seed, a resume, or another V2 checkpoint would not answer the failure and are not justified.

## Registered next step

- Do not open, resume, reuse, or promote any V2 checkpoint.
- Do not scale this mechanism, start navigation training, or access held-out mazes.
- Permit at most one fresh, capped **fixed-teacher latent-delta JEPA** falsification on the identical fixed schedule:
  - freeze the accepted N320 target encoder for the whole probe, so the target geometry cannot follow the online network into collapse;
  - still train the online encoder, ordered-history module, and action predictor jointly in one JEPA backward pass;
  - align the current online embedding to the fixed teacher and predict fixed-teacher future deltas relative to fixed-teacher `e2`; zero delta remains the persistence baseline;
  - remove the cyclic wrong-action, persistence-hinge, and detached-history-hinge training losses, which V2 showed can be satisfied without useful prediction;
  - retain action, hold, persistence, and history controls for evaluation only, where they must emerge from lower real prediction error;
  - keep the same seed, observations, data, and 1,000-update / 16,000-presentation ceiling, with fresh N320 initialization and no V1/V2 tensor input.
- Gate the probe on fixed-teacher identity/rank, online noncollapse, beating persistence at every horizon, a positive H4 persistence bootstrap lower bound, and naturally positive action/history controls. Complete the exact fixed schedule unless execution itself fails; a collapsed observation is simply ineligible.
- This is a materially different target geometry, not a V2 loss-weight retry. If it cannot remain noncollapsed, beat persistence, and make ordered history useful within the same cap, stop the recurrent H4 latent-dynamics branch rather than issuing another nearby version.

## Terminal bindings

- Completion file SHA-256: `8e7472b1a824cfd23f4832d29f893dfcb686e48b8082ffc32096642b49121255`.
- Completion content SHA-256: `06203ace3c5d1cfa30e8c642633c101fcf81cb5febab0b300cf7ea01309daeca`.
- Result file SHA-256: `12da66dd29193c083e38b5ce7a1ef909a8a03431aef39b4b5bf3d4d25c05c479`.
- Metrics file SHA-256: `0bb944129bd0b3418d17d067a1c3119a4091568469359014214d67ecacd07bf9`.
- Artifact file SHA-256: `a2da087cfad9966f680c3f2d5ffa052e16868a3c65adaf1cc3f86d20b37029a5`.
- Access file SHA-256: `7c673e86cd46f0b7d5aebbdc94c6b31474f061daeb976b173a954019b64a2fa0`.
- Reservation file SHA-256: `6a623263a4c19dd58105409a1c0f60da3c7d14cdc187ca7c2fbd812612e7359a`.
- Independent terminal-receipt audit: pass; exact counts, cross-bindings, source bindings, and all forbidden-zero fields reconciled without opening checkpoint bytes.
