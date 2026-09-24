# Checkpoint (a), V2 — stop and approval submission

All work specifically authorized by the two-state recheck amendment is complete. The recheck had not started on receipt; all pre-execution changes were implemented, including timestamp-only replacement of the reactive snapshot and source replay after candidate branches. No pre-execution amendment was omitted. Optional feature summaries also completed within the original caps.

| Deliverable | Outcome / evidence |
|---|---|
| Amended renderer recheck | **PASS: 96/96 exact source-replay RGB matches**, physical tolerances and candidate repeat agreement; 42 branches, no extra attempt |
| Measured execution budget | 36.6 simulated s; 104.90 wall s; 0.0385 core-hours; 7.86 GiB RAM; 2.99 GiB device VRAM; 54.01 MiB retained at closeout |
| Renderer provenance review | September matched assays, near-goal and stalled-turn diagnostics avoid the verified rewind path; legacy whole-population source-version coverage remains explicitly unresolved |
| Stage A hold analysis | Four completed traces; no new physics/models/regret; most holds involve logged movement ineligibility, with score-loss and explicit-override cases separated |
| Added analytical cases | **13/13 pass**; original fixture failure and erratum preserved |
| Phase 2 readiness | **Not yet qualified; no execution approval** |

Read the [recheck report](go2_decision_headroom_rgb_recheck_result_v2_2026-09-23.md), [provenance report](go2_renderer_provenance_readonly_2026-09-23.md), [hold report](go2_stage_a_holds_exploratory_2026-09-23.md) and [revised protocol/budget](go2_decision_headroom_protocol_v2_checkpoint_a_2026-09-23.md). Machine-readable results and source/input SHA bindings accompany each report.

The proposed protocol/configuration is `go2_decision_headroom_protocol_v2_checkpoint_a_2026-09-23.json`, SHA-256 **5617f2ddd0d37a5bc90a305a50ecca397494d8f799038895ea789c09f1060825**. It binds the protocol text, recheck source/input identities and completed reports. The recheck's immutable pre-execution manifest SHA is **ac8061a77319b6eb88035a9566d95740906a9d3652d365e7da615023d4a2586e**.

Requested next approval scope: the **remaining Phase 1 qualification only** in that identified proposal—four missing source/layout cells, four states, 84 branch attempts, 137.6 simulated seconds; 1 hour wall / 16 core-hours / 16 GiB RAM / 8 GiB total VRAM / 2 GiB retained / 3 GiB peak writes, retaining 12/4-GiB filesystem reserves. This includes bounded physical-clearance and original-source-input qualification. No completed Stage A or recheck assignment would be restarted. Any failure stops without retry, extension, fitting, controller repair or artifact retirement.

The conditional reduced-audit design is also documented, but requires further qualification and exact new layout/implementation bindings before a separate Phase 2 submission. The original eight-state physical evidence and two-state RGB pass do not establish universal determinism. Current disk clearance does not yet certify moving articulated legs/body or actual between-sample motion. Those limitations are not hidden by the passing renderer test.

**Stopped at checkpoint (a).** The user’s amendment section 6 and the authoritative handoff’s approval-checkpoint section explicitly require this stop. Neither a passing result nor this commit grants Phase 2 permission. After any separately approved fixed audit, deliver the versioned branch panel, results and one-step decision memo, then stop again without implementing the recommendation.
