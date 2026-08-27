# PLAN_AWARE_MONOTONE_JEPA_COST_V1 execution correction 2

Status: prospective execution-only correction; no scientific result has been rerun.

This separate amendment corrects canonical UTF-8 result persistence and terminal custody. It does not alter the frozen scientific contract, model, loss, data, metric, gate, classification, or next-decision rule.

Scientific contract freeze: `9c1c3adcfb8382c33e8da8895dc345e006e92e43`; contract digest: `1667f325be2c835a6222dc90bb684f373a06b365d59b70e9746fd7adb052c382`.
First execution-correction freeze: `14625958c0fcc21af05b33fd30c6cc2fc8537745`.
Required correction-2 commit subject: `Correct canonical UTF-8 result persistence and terminal custody`.

Both immutable failed archives are bound and nonreusable. One wholly fresh final attempt is authorised; every file, checkpoint, tensor, ledger and receipt from both archives has reuse count zero. Every compared replay artifact must be a canonical attempt-contained path, may not be a symlink, and may not share a device/inode pair with its failed-archive counterpart. A further retry is not authorised.

The second failure occurred during terminal persistence after Stage B. The amendment was authored without opening scientific metric values, row outcome values, or tensors; it is gate-status-aware. Canonical self-digests use compact sorted JSON with `ensure_ascii=False`, `allow_nan=False`, UTF-8 bytes and no trailing newline inside the digest.

The fresh attempt must reproduce Stage A before Stage B. Immediately after Stage B and before any Stage C child, it must reproduce all 2,304 Stage-B ledger rows, normalized scientific projections, 48 context identities/shapes/dtypes/array-content digests, and 96 stable NPY byte bindings against the immediately preceding failed archive. NPZ container bytes are deliberately not compared.

The scientific evaluator writes a nonpublishing terminal staging handoff and exits. The launcher proves its process-group and device cleanup, then starts a separate finalizer. The finalizer records zero scientific evaluator/helper matches while disclosing its own and the still-live launcher's exact identities, and atomically publishes canonical and tracked outputs. Reproduction after that atomic rename reads the canonical root while preserving the original attempt identity bound in terminal staging. Only after both producer processes exit may an independent external checker record literal zero exact matches for all producer roles; that witness remains outside the canonical storage fixed point.

The external checker also requires canonical and tracked result/report byte equality, an official deep-check pass, and a clean result commit that is the direct child of the correction-2 freeze, has subject `Evaluate plan-aware monotone JEPA route cost`, and changes only `docs/lewm_plan_aware_monotone_jepa_cost_v1_result_2026-08-27.json` and `docs/lewm_plan_aware_monotone_jepa_cost_v1_result_2026-08-27.md`.

Amendment content digest: `4eabc30357a39774aec25fc9be4594e6ef9f634e1269460614b8a2311af88322`.
Overlay schema digest: `cc256c5b11f07dbc03df4495f0117986ccf02337631ceb4714d5027ae03968a6`.
Fixture digest: `8f7ff65c39b6fd06a84c12bd28a72eb2fbf98efb551aa5dcf5476eade1228a06`.
