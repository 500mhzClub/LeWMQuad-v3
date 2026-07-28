# RGB Swept-Progress Survival Joint-JEPA V4 — Physical-Evidence Calibration Execution Binding

- Status: frozen before the sole scientific execution and before any calibration/selection payload access.
- Preregistration: `e983e0abd9349426f69262563e12d90a4488180e`.
- Runner source / access-receipt closure: `2e32cee0233bcc214707f6cb53cf0721815c73b5` / `440ff2ac103025f8dc15c186737b63d1e2519ad8`.
- Runner SHA-256: `cee7c9c70e6bb9d2bacc6528ef77d009c80e2f484400de9f6445ebfd0c010313`.
- Focused test SHA-256: `891aae8a11307a8e4f95a24602d2887b3218b3007dd2772e2bfebedac5f8edf3`.
- Fresh output root `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence_calibration/attempt_v1` was confirmed absent immediately before this binding.

## Review and test closure

- Independent preregistration/source audit: pass; no concrete science or execution blocker.
- Independent integration audit: the tiny NumPy/PIL/Torch runtime supplies every attribute used by `RawInputs` and `DirectBevNarrowLoader.endpoint_batch`; constructor argument order, authorization shape, path normalization, and endpoint-batch signature all match the frozen implementations.
- Independent failure-receipt review found that post-data operational failures originally omitted the raw-access ledger. The final closure fixes that issue, records full loader and consumed-file receipts on both success and failure, validates forbidden counters as zero, and tests a post-data failure explicitly.
- Focused runner suite: `6 passed`.
- Combined runner, calibrator, threshold metrics, direct adapter, candidate admission, frozen V4 model, and frozen V4 executor suite: `48 passed`.
- Frozen dependency hashes validate successfully and `git diff --check` passes.

## Sole execution

- Run exactly once from the repository root with:

  `PYTHONPATH=.:lewm_worlds /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence.py`

- The runner must use only the admitted candidate copy and the two frozen development payload roles. It must fit once on `probability_calibration`, select thresholds once there, and apply the unchanged calibration and thresholds once to `checkpoint_selection`.
- No train-role RGB/raster, predictor inference, model backward/optimizer/EMA, N320, original V4 runtime, G2, navigation, held-out, sealed, rejected-checkpoint, accelerator, or production operation is authorized.
- Scientific pass or fail is terminal and must be recorded. There is no resume, threshold relaxation, selection-role tuning, or scientific retry.
- A pass grants only authority to prepare a separately reviewed one-shot G2 binding; it does not open G2 or qualify navigation.
