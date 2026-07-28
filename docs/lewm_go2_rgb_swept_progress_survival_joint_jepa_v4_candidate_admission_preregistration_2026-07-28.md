# RGB Swept-Progress Survival Joint-JEPA V4 — Pre-G2 Candidate Admission Preregistration

- Status: frozen before implementation and before any V4 runtime-artifact access.
- Purpose: admit and package the already-passing full V4 model as one load-valid pre-G2 candidate. This is artifact custody, not retraining, tuning, checkpoint selection, or a new scientific result.
- Authority: the user's standing permission to progress autonomously within the RGB-only fully learned joint-JEPA navigation goal. This supersedes only the earlier operational non-access status for the passing V4 artifact, for the single exact read below. Every rejected checkpoint remains forbidden.

## Bound scientific result

- V4 preregistration / source / execution binding / terminal result commits: `9f9ab784b4bfa827585ec095f2a7f7a30333480a` / `aaa47a138d0eeb78aa20d9524e67f813f7a74a41` / `5a48b878c97717e27bf7e4bdb1c6a13c1687117e` / `8b3a8063b087c81030189deadc6c5f6e1c7d44c3`.
- Bound V4 result file/content SHA-256: `bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4` / `27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8`.
- Frozen V4 model/executor SHA-256: `1c5a26f02a856d9a84903063c53bf23095142d86885787556b09388c508711ef` / `243ef91ccec4e1fcdfa5a0c3f112bf4c645f46ba7de8692c1dddcb47f87c9f40`.
- The committed result proves `PASS_FULL_ARM` at the fixed terminal update with all 24 development checks passing. Its canonical JSON cryptographically commits to `training.checkpoint.path`, `byte_count`, and `file_sha256`, although that checkpoint binding has not previously been independently verified.
- The completed no-persistence result at `14eeda5d5d0611c0571807cb3a637e06feb35512` remains non-input provenance. It grants no positive persistence claim and cannot replace or initialize this candidate.

## Exact allowed input and access

- Read only `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder/attempt_v1/result.json` and require the bound file/content hashes, canonical JSON, exact V4 result schema, `PASS_FULL_ARM`, exact 1,000-update / 16,000-presentation accounting, and all gate checks true.
- Extract the checkpoint basename, byte count, and SHA-256 only from that verified result. Require basename `checkpoint_update_1000.pt`, resolve it beneath the same attempt root, reject a symlink/non-regular file, and read it exactly once into memory.
- Require the in-memory bytes to match the result-bound size and SHA-256 before deserialization. Deserialize those same bytes on CPU exactly once with `torch.load(..., weights_only=True)`.
- Do not list or search the V4 output directory. Do not read its trace or any other runtime artifact. Do not name, open, hash, load, or inspect V1–V3, no-persistence, rejected, G2, navigation, held-out, sealed, or production material.

## Fail-closed checkpoint admission

- Require the exact V4 checkpoint schema and exact payload key set; `development_only=true`, `qualified=false`, `resume_authorized=false`, initialization source `exact_n320_encoder_only`, predecessor experiment checkpoint read false, exact seeds, auxiliary-objective receipt, decoder receipt, and terminal accounting.
- Require a nonempty tensor-only `model_state_dict`; every tensor must be CPU-resident and finite. Record a stable sorted inventory of name, shape, dtype, byte count, and tensor-byte SHA-256 plus one canonical inventory digest.
- Reconstruct the frozen V4 class using the checkpoint's stripped `encoder.*` tensors and exact `predictor.swept_progress_head.sweep_masks`, then strict-load the complete state with zero missing or unexpected keys.
- Run one fixed CPU synthetic batch of shape `[1,3,112,112]` through online RGB encoding, semantic decoding, and all-nine action-conditioned predictor/survival inference. Require exact registered shapes, float32, finiteness, the fixed action vocabulary, and no state mutation. No optimizer, backward, EMA, dataset, calibration, metric, navigation, or accelerator operation is allowed.

## Output and authority

- Fresh write-once root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_admission/attempt_v1`.
- On success, write an exact byte-for-byte candidate copy named `candidate_checkpoint.pt` plus canonical `candidate_receipt.json`. Bind the source/result/checkpoint identities, state inventory, strict-load and synthetic-inference receipts, exact access counts, and the copied checkpoint's identical size/SHA-256.
- On failure, write only canonical `failure.json`; any copied bytes remain permanently unqualified. Candidate and failure receipts are mutually exclusive.
- Success means only `pre_g2_candidate=true`. It must retain `g2_qualified=false`, `navigation_qualified=false`, `promotion_performed=false`, `deployment_authorized=false`, `heldout_or_sealed_opened=false`, and `resume_or_training_authorized=false`.
- Once a candidate copy is admitted, all downstream source and qualification work must use only the copied candidate binding; the original V4 runtime root receives no further access.
- No choice between checkpoints is allowed: this previously selected full V4 terminal artifact is the sole candidate. A binding/load failure does not authorize using the no-persistence control or a rejected artifact.
- This admission does not authorize G2 access. A reviewed V4-specific inference/input adapter and a separately frozen one-shot G2 execution binding are still required.
