# Amended RGB restoration recheck — PASS

The recheck had not started when the amendment arrived: the output root was absent and no recheck owner was live. All pre-execution amendments were incorporated. The frozen successor manifest is `go2_decision_headroom_rgb_recheck_amended_v2_2026-09-23.json`, SHA-256 `ac8061a77319b6eb88035a9566d95740906a9d3652d365e7da615023d4a2586e`. The approved correction only invalidates the Genesis visualizer/rasterizer caches after restoring physical time/state, before restoring RNG state. Controllers and models were unchanged.

| Strict criterion | Result |
|---|---:|
| Attempted branches | 42/42 |
| Source-replay RGB, both cameras | **96/96 bitwise equal** |
| Source physical restoration | PASS at all recorded horizons; original 1-mm / 0.1-degree and contact criteria |
| Candidate repeat agreement | PASS, all 36 pairwise branch comparisons |
| Source-trace replays after candidate branches | 4 |
| Retries / extra branches / new source trajectories | 0 / 0 / 0 |

Per state the order was source replay 0, all six candidates in canonical order with three repetitions each, then source replays 1 and 2. Thus the last two replays exercise restore-after-candidate, rather than only restoration into a new scene. The process exited successfully. No further physics is running under this recheck.

The command-history assignment remains source_00/frame 12 at 2.7 simulated seconds. Reactive source_01/frame 132 at 14.7 seconds replaced frame 12: it was the latest already-collected snapshot with a complete forty-command source trace. Frames 12, 52, 92 and 132 were eligible; selection used timestamp alone. Relative to reactive frame 12, the selected snapshot is 12 seconds later, its base moves 27.32 mm and yaw changes −56.13 degrees. Exact base poses and both camera transforms are in the manifest. Both states are still on exposed layout 00; this is broader temporal/view coverage, not new layout or action-controller coverage.

Source and replay use the same paired native camera call, 640×480 uint8 RGB arrays and lossless PIL PNG save path (compression level 1). There is no collection-time resizing or colour conversion. Original decoded source-image bytes were checked against their collection metadata. Historical source images were not altered. The frozen encoder subsequently uses its existing RGB/PIL bicubic 512×384 preprocessing and ImageNet normalization on both sides.

All 96 image pairs have MAE, RMSE, median, 95th/99th-percentile and maximum absolute channel error **zero**, and zero changed-pixel fraction. Each retained histogram has all 921,600 channel samples in bin zero. This is a distributional equality result, not a tolerance-based visual judgment.

The optional encoder diagnostic completed within the same budget: 160 primary images, no dense tensors retained, no ranking or fitting. All 48 replay/source feature comparisons have raw and normalized MSE/RMSE zero. The 240 between-candidate comparisons include 90 identical shared-prefix comparisons at 100–300 ms. At later horizons candidates differ:

| State | Mean pairwise raw MSE at 400 ms | At 800 ms | Mean normalized MSE at 800 ms | Maximum individual raw / normalized MSE |
|---|---:|---:|---:|---:|
| Command history, frame 12 | 1.63609 | 3.05452 | 0.77162 | 3.68196 / 0.93095 |
| Reactive, frame 132 | 1.55709 | 3.09213 | 0.79086 | 3.64560 / 0.93892 |

Across all horizons the minimum candidate difference is zero. Replay/source MSE divided by mean between-candidate MSE is zero for the 30 post-prefix replay comparisons and **undefined** for the 18 shared-prefix comparisons (denominator ≤1e−12). Absolute differences are retained alongside ratios. The optional float32 cosine calculation has small negative roundoff even for identical features; those values are preserved and are not interpreted as negative distances. Equality/MSE provides the relevant zero-error evidence. These diagnostics could not have rescued failed strict acceptance and do not establish that visual discrepancies would be harmless.

| Resource | Measured | Cap |
|---|---:|---:|
| Simulated time, including settling | 36.6 s | 36.6 s |
| Execution wall time | 104.90 s | 900 s |
| Aggregate CPU | 138.50 core-s (0.0385 core-h) | 4 core-h |
| Sampled aggregate RSS | 7.86 GiB | 16 GiB |
| Sampled total device VRAM | 2.99 GiB | 8 GiB |
| Retained bytes at closeout | 56,632,148 (54.01 MiB) | 1 GiB |
| Peak additional observed bytes before final closeout files | 56,537,572 | 2 GiB |
| Minimum RecoveryStorage free | 35.94 GiB | ≥12 GiB |
| Minimum workspace free | 5.50 GiB | ≥4 GiB |

Final closeout files add a small amount beyond the last periodic write sample; retained closeout bytes also remain far below the peak-write cap. RAM/VRAM are sampled measurements, not OS-enforced limits. Read-only report preparation after exit is outside the recheck owner's measured runtime; no further encoder computation was performed.

The original pilot failure remains intact: eight physical states passed restoration/repeat checks but **0/384** original source-replay images matched. This successor does not retrospectively qualify those branch images. Nor do two corrected states establish complete determinism, articulated clearance, all source classes or unseen-layout validity. No comparative audit selections or regrets have been computed.

Evidence: [bounded result and SHA bindings](go2_decision_headroom_rgb_recheck_result_v2_2026-09-23.json). Native root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_decision_headroom_rgb_restore_recheck_v1_attempt_001`. Per-branch restoration comparisons, original source identities, command tapes, physical trajectories, camera definitions and software identities are retained there or bound by the manifest. This is checkpoint-(a) evidence, not Phase 2 authority.
