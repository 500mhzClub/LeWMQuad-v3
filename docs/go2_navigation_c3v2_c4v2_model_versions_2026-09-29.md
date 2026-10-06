# Model versions C3-v2 and C4-v2, 29 September 2026

These are new model versions recorded under the [C3-v2 pre-declaration](go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md) (§6 and Amendment 1, commit 47f88bda). The harness `v4_completed_support` is unchanged.

**C3-v1's validation result (13/20) remains the pre-registered capability result.** C4-v1's validation result (19/20) also stands. Neither new version has a validation result: validation was not re-run, and their closed-loop evidence is the fresh check only.

Artifacts are under the programme's artifact root:
`RecoveryStorage/…/go2_navigation_capability_v1_attempt_001/`

## C3-v2: JEPA controller with a refit motion readout

| Part | Status and binding |
|---|---|
| Frozen encoder (V-JEPA 2.1 ViT-L) | Unchanged: `vjepa2_1_vitl_dist_vitG_384.pt`, `7ea9b7cb…` |
| Frozen action-conditioned predictor | Unchanged: `go2_horizon_dense_predictor_v1_attempt_001/action_final.pt`, `5d39753f…` |
| **Motion readout** | **New:** `c3v2_readout_fit_v1/readout_v2_final.pt`, sha256 `11a61e419159e7e0591c60c0e5eb6921eff926f2edb81c247988e7276094923c` (3,413,325 bytes) |

**The readout fit.**
- **Records:** plan `a298e1f1…`, result `693bf99f…`, trainer `scripts/train_go2_c3v2_readout_development.py` (commit d16dfc72).
- **Initialisation:** `mixed_data_final.pt` (`bbbb05fd…`). This is the same parent C3-v1's readout (`maze_data_final.pt`, `aa853c6f…`) was fine-tuned from. C3-v2 is a sibling of C3-v1, not a continuation of it.
- **Recipe, identical to C3-v1's:** AdamW with lr 1e-3 and weight decay 1e-4, gradient clip 1.0, and 440 updates of batch 64 (32 old contexts plus 32 maze-pool contexts). Seed 2026092205, with the same schedule seeds, and the fixed final checkpoint.
- **Data, the only change:**
  - The maze pool adds 2,768 new fit-split rest/turn contexts, making 5,216 instead of 2,448.
  - The readout's future input is the frozen predictor's pooled prediction for the executed command tape, instead of the encoder's features of the actual future frame.
- **Training data:** `c3v2_data_v1/train_samples.json`, `6b376596…` (11,182 contexts).

**Offline acceptance: passed all seven §5 criteria.** Result `c3v2_acceptance_v1/result.json`, `ec0f7727…`.

## C4-v2: direct supervised motion predictor, refit on matched data

| Item | Binding |
|---|---|
| **Checkpoint** | `c4v2_fit_v1/direct_v2_final.pt`, sha256 `68de16afbb9033811871292949a458cf39436ae274768033d7ca7609803b12bd` (69,592,837 bytes; 17,397,283 parameters) |
| Records | Plan `23c5fd07…`, result `e2576140…`, trainer `scripts/train_go2_c4v2_development.py` (commit d16dfc72) |
| Recipe | C4-v1's plan unchanged: random initialisation, seed 2026092513, 1,760 updates, fixed final checkpoint |
| Data | The same 11,182 contexts as C3-v2 |
| Predecessor | C4-v1, `c4_fit_attempt002/direct_final.pt`, `d491584b…` |
| GPU time | 7,470 s for this fit; 12,340 s (3.43 h) of C4's 12-h cap used in total |

**Held-out isolation audit: passed** (`c3v2_data_v1/c4v2_heldout_isolation_audit.json`, `6f114088…`). No held-out frame or recording directory is used in training, the trainer reads only the training file, and the checkpoint is the fixed final one.

**The §5 measures (reported, not gating):** C4-v2 meets them against C4-v1 except "no loss elsewhere". On other held-out windows its yaw error is about 20% worse, while its XY error improved.

## Entry into E1

Under the pre-declaration's §6 and Amendment 1, C3-v2 and C4-v2 enter E1 as a pair only if C3-v2 has zero disallowed contacts and zero hard-clearance violations in the fresh check. Otherwise C3-v1 and C4-v1 enter. **The outcome is recorded in the capability-phase report.**
