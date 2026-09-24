# RGB Swept-Progress Survival Joint-JEPA V4 — Candidate Admission Result

- Terminal status: `ADMITTED_PRE_G2_CANDIDATE`.
- Candidate-admission preregistration / source / execution binding: `b5b4ca50b50257872c9ee12a96b901710e35bac9` / `fd3348b7c94f8f00617e19bc1b0601ffd92cce1d` / `cf46c9dd46489d06b7d42005f071e9ff0953e6b3`.
- The sole authorized admission command ran once and exited successfully. There was no retry or resume.

## Bound candidate

- Candidate checkpoint: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_admission/attempt_v1/candidate_checkpoint.pt`.
- Candidate byte count / SHA-256: `25,673,535` / `f8a330d1a4834e4cc61f7acae00069f866a37a5693464e6fbb93b998a971d37a`.
- The candidate size and SHA-256 exactly equal the checkpoint binding in the canonical passing V4 result. The copy was written from the same verified in-memory bytes; the original checkpoint was not reread.
- Candidate receipt file/content SHA-256: `7b21e9a908c05f56c344a74682ee0a3d912c449920d57ee9298619f53c9f66f1` / `247e9f1d81cb143631c4be4b85173f707516ff5cf32a0e9e08ca6d8100420f8f`.
- State inventory: 224 unique sorted tensor entries; canonical inventory SHA-256 `582219fcd6e73141c3f9c4e6dad4aca53500d0623d3f7265c4fcfae7c4fb195e`.

## Validation result

- The exact canonical V4 result remained `PASS_FULL_ARM`, with all 24 checks true and exact 1,000-update / 16,000-presentation accounting.
- The checkpoint retained its exact development-only, unqualified, nonresumable schema and receipts.
- CPU `weights_only=True` deserialization occurred once.
- Strict full-model reconstruction passed with zero missing or unexpected state keys.
- One fixed CPU RGB smoke batch produced finite exact shapes for the online latent `[1,64,64,64]`, semantic logits `[1,3,64,64]`, all-nine action-predicted latents `[1,9,64,64,64]`, and survival logits `[1,9,16]`.
- The fixed nine-action vocabulary was present and inference did not mutate model state.
- Independent post-run audit reproduced the receipt self-hash, candidate size/hash, state inventory, load/inference receipts, access counts, and authority. It read only the candidate receipt and candidate copy.

## Access and authority

- Exact original-runtime access: one canonical V4 result read and one result-bound checkpoint read/deserialization.
- Candidate writes: one byte-identical checkpoint copy and one canonical receipt.
- Zero dataset, trace, training, backward, optimizer, EMA, accelerator, calibration, G2, navigation, held-out, sealed, rejected-checkpoint, or production operations occurred.
- Success grants only `pre_g2_candidate=true`. G2 qualification, navigation qualification, promotion, deployment, training, and resume remain false.
- The original V4 runtime root is permanently closed to further access. All downstream development must use only the admitted candidate copy and its binding.

## Scientific meaning and next gate

- This result does not add a new encoder or predictor claim. It proves that the already-passing, jointly trained V4 JEPA artifact exists, matches its committed result, can be reconstructed exactly, and exposes both physical semantic evidence and action-conditioned predictor/survival outputs.
- G2 remains unopened. The next step is development-only probability calibration and conservative physical-evidence threshold selection, fitted on `probability_calibration` and independently checked on `checkpoint_selection` using the candidate copy. A one-shot G2 binding is allowed only if that development gate passes.
