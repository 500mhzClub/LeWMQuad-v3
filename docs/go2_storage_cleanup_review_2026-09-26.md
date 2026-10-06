# Storage cleanup options — 26 September 2026

Read-only assessment; no artifacts deleted. Sizes below are allocated GiB and exclude sealed material. The current navigation pilot is complete and stopped at storage admission.

## Recommended candidate: older development depth

These eight roots contain 63,050 primary/auxiliary depth NPZ files totalling **98.71 GiB**. They are historical diagnostic/reference recordings, explicitly pinned by the development retention policy. Releasing those pins requires a deliberate retention decision; this review does not authorize retirement.

All roots are relative to `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.

| Root | Depth files | Allocated GiB |
|---|---:|---:|
| `go2_pair_local_plane_preferred_learned_round_trip_native_layout03_4800_v1_attempt_001` | 9,610 | 15.38 |
| `go2_pair_local_plane_preferred_reactive_round_trip_native_layout03_4800_v1_attempt_001` | 9,610 | 15.18 |
| `go2_plane_consensus_learned_round_trip_native_layout00_v1_attempt_001` | 7,210 | 10.42 |
| `go2_plane_consensus_reactive_round_trip_native_layout00_v1_attempt_001` | 7,210 | 11.23 |
| `go2_progress_rejoining_learned_round_trip_native_layout05_4800_v1_attempt_001` | 9,626 | 15.17 |
| `go2_progress_rejoining_learned_round_trip_native_layout06_4800_v1_attempt_001` | 6,708 | 10.97 |
| `go2_fixed_transfer_preferred_reactive_round_trip_native_layout05_4800_v1_attempt_001` | 3,466 | 5.84 |
| `go2_fixed_transfer_preferred_reactive_round_trip_native_layout06_4800_v1_attempt_001` | 9,610 | 14.51 |

The proposed scope is depth only. Keep all RGB, physics, commands, trajectories, configurations, models, results and failure diagnoses. A retirement would need exact inventories and per-run `depth_retention.json` markers. These roots currently have no such markers, and the counted depth files have one hard link each.

Tradeoff: historical full-depth sensor replay would no longer be available. Regeneration is not guaranteed to reproduce the exact historical closed-loop recording. The layout-5/6 pairs were retained specifically as false-arrival and registration-failure diagnostic references. Their results remain part of the historical population even if depth is retired. Before execution, check that no current reader or pending replay requires the selected depth.

The policy explicitly preserves these references: [retention policy](go2_development_artifact_retention_2026-09-14.md). Current navigation preregistration also records no existing artifact retirement. Routine unpinned retirement authority is not authority to discard these pins.

## Other findings

- Navigation development artifacts occupy approximately **883.85 GiB**; the current capability pilot occupies only **1.33 GiB**. Keep the current pilot and its failures, fixed C4 checkpoint and replay evidence.
- Pip cache is only **14 MiB**; Genesis GSD cache is below **1 MiB**. Earlier large cache cleanups have already happened.
- Triton, Gstaichi, COMGR and Quadrants caches total approximately **1.76 GiB**. They are secondary candidates, not sufficient alone to admit the next screen; this review has not qualified their removal.
- The approximately **181.95-GiB** temporal cache contains scientific data, not merely disposable downloads.
- RecoveryStorage environments total **24.62 GiB**. Workspace paths resolve to these environments, so they are not duplicate copies. The active Genesis environment explicitly imports the **14.60-GiB** world-model environment through a `.pth` file: preserve it. The **7.50-GiB** `genesis_render_vulkan` environment is a possible future review candidate, but its removal has not been qualified.
- Preserve current training inputs, pretrained encoders, frozen checkpoints, the closed audit's retained panel/snapshots and all sealed material.

## Capacity effect

The recorded next-screen shortfall is **3.81 GiB**. The reduced programme projects **86.11 GiB** additional recordings/videos; approximately **90 GiB** of reclamation provides room for that current projection, but not an unlimited number of harness versions. Releasing the eight historical depth pins would reclaim approximately **98.71 GiB**, enough for this projection while retaining numerical results and non-depth evidence.

See [pilot result and budget](go2_navigation_capability_result_2026-09-25.md). No cleanup or new simulation was executed during this review.

## Execution addendum — approved and completed 26 September

The user subsequently instructed **"do depth"**, releasing the eight listed
historical depth pins. Retired **63,050 files / 98.71 GiB**.
All **63,342 original non-depth files** were hash-verified unchanged.
RecoveryStorage now has **113.48 GiB free**, approximately
**101.36 GiB** above the 12-GiB reserve and 128-MiB guard.
This clears the recorded storage shortfall for the next screen and current
86.11-GiB programme projection. No simulations were launched by this cleanup.

Receipt: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/depth_retirement_historical_references_2026-09-26/result.json`.
Per-root markers and the exact deletion inventory preserve the retirement record.
The earlier read-only assessment above remains the original proposal.
