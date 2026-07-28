# RGB Swept-Progress Survival Joint-JEPA V1 — Terminal Result

- Terminal status: `FAIL_FULL_ARM` (valid completed scientific run, not an execution failure).
- Run completed once on 2026-07-28 with no retry or resume.
- Result file SHA-256: `19944ff6ff205932b991498f0693049c7b9ac5761ee7e4ede233af07de006e88`; content SHA-256: `81409c1d8429caba62ac213fb4d93250202804587738524627d83c00fa67e8c6`.
- Training-trace file SHA-256: `35b9b178d3a660056b98436ff7b5121822bb9a25e4881d867e5b372cb292ac11`; content SHA-256: `a491d8698bce22ac218c6a7daac5779ebb3fa4d95331d17132be76c9d148df4a`.

## Accounting and integrity

- Exact cap completed: 1,000 updates, 16,000 presentations, 4,000 microbatch graphs/backward calls/predictor objectives, and 1,000 optimizer and EMA steps.
- Trace contains exactly 1,000 contiguous finite rows. Result and trace canonical self-hashes and the result-to-trace binding verify.
- Exact frozen label, N320, schedule, GPU, and both sweep-mask bindings verify.
- Forbidden input count: `0`; G2/navigation/final-evaluation open count: `0`.
- The produced checkpoint is unqualified, development-only, not resumable, and was not promoted. The matched no-JEPA arm did not run and no JEPA treatment-effect claim is made.

## Scientific result

- The swept-progress mechanism passed every registered navigation-relevant selection gate:
  - normalized chosen/oracle prefix utility: `0.899706` (floor `0.85`);
  - selected zero-prefix rate: `0.042607` (ceiling `0.05`);
  - unequal-prefix pair concordance: `0.865619` (floor `0.75`);
  - every one of eight families passed its utility, zero-prefix, and concordance floors.
- The full arm passed every registered control comparison:
  - coordinate-matched persistence equal-scene delta `+0.151843`, bootstrap lower bound `+0.090934`, 7/8 positive families;
  - shuffled predicted-action slots `+0.332015`, lower bound `+0.274352`, 8/8;
  - wrong RGB `+0.092125`, lower bound `+0.055331`, 7/8;
  - train action-mean prior `+0.068612`, lower bound `+0.031280`, 7/8.
- Selection expected-progress MAE was `0.248126 m` overall and `0.211608 m` on informative states; weighted progress-calibration gap was `0.035277 m`.
- Semantic balanced accuracy (`0.822839`), free recall (`0.885680`), and unknown recall (`0.938535`) passed.
- Exactly two semantic-retention checks failed:
  - occupied recall `0.644302` versus the `0.70` floor;
  - rough-scene occupied recall `0.580587` versus the `0.65` floor.
- Training did not collapse. First-100 to last-100 mean losses improved: total `6.1285 -> 4.0656`, JEPA persistence `2.4462 -> 1.2128`, survival `0.6820 -> 0.3428`, and ranking `0.8426 -> 0.5253`. Semantic loss improved only `2.1577 -> 1.9847` and remained the limiting term.

## Interpretation and next decision

- This falsifies the claim that the complete V1 stack already retains enough dense obstacle semantics, but it does not falsify the swept-progress mechanism: its action ordering, RGB dependence, persistence advantage, calibration, and family generalization all passed strongly.
- Extending the identical schedule is not justified; optimization had largely plateaued by updates 800–900.
- One bounded V2 successor is justified by the strong improvement and the narrow, mechanistically distinct miss. Preserve architecture, data, N320 initialization, schedule, masks, cap, JEPA persistence, survival, ranking, controls, and gates; add only a normalized occupied-vs-rest safety auxiliary on the existing semantic logits, jointly from update 1.
- If V2 passes the complete gate, run the matched no-JEPA arm before any causal JEPA, G2, navigation, or held-out claim. If either semantic recall still fails or the progress/control gates regress, close this successor.

## Post-result authority deviation

- After the CLI had reported `FAIL_FULL_ARM`, the root audit command mistakenly included `checkpoint_update_1000.pt` in a three-file `sha256sum` invocation.
- Scope: one sequential byte-level read sufficient to compute SHA-256 `b8da881507c6b95d5a2b8cfaf42327a09f18465050373f36725737d5fc0ce8d6`, which was already present in `result.json`.
- No checkpoint deserialization, tensor/model-state inspection, copy, evaluation, promotion, retry, resume, or scientific use occurred.
- The read cannot alter or invalidate the already-completed training or gate result, but it violated the inherited no-read boundary for a rejected checkpoint. No further access to that checkpoint is authorized or will occur.
