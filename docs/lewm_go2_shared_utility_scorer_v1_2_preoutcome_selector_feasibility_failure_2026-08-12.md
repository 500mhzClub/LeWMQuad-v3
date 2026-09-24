# Go2 shared utility scorer V1.2: terminal pre-outcome selector-feasibility failure

Disposition: **STOP_PRE_OUTCOME**.

The final public binding receipt is complete and decision-ready. It exhaustively reduced all 1,284 allowed scenes across eight required families and three required strata. Twenty-three of the 24 family–stratum criteria pass. The sole failure is `small_enclosed_maze × completion_enriched`: **0 eligible distinct scenes, with 5 required (`0 < 5`)**, after all 182 allowed small-maze scenes were scanned.

This is not a scorer failure and is not a response to scorer performance. No candidate outcomes were loaded, no new state or branch identities were created, no branches were attempted, no frames were rendered, no target latents were encoded, and scorer training never started. Three pre-existing valid outcome-free 15-state shards remain on disk, but there is no active 120-state manifest and they were not made into a successor identity set. Phase 1 revalidation, successor-contract issuance, active identity issuance, outcome generation, and every scientific or downstream stage remain unreached.

## Binding evidence

The scientific verdict in this report comes only from the final public selector-feasibility receipt:

- Path: `.generated/go2_branch_corpus_v1_2/scorer_fit/state_selector_feasibility_receipt.json`
- Schema/status: `go2_scorer_fit_state_selector_feasibility_receipt_v1` / `FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY`
- Raw SHA-256: `28e852792b5de24b2d008c5bb3f95521da668927e555deb9eb3c508bb6b0e59f`
- Semantic self-digest: `2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7`
- Clean source commit: `4dd906e7e82dd7c1530622a00319d1455c86b8e5`
- Clean source binding: `ba1ca3e11b2694bdac79a981c1960b82e26ce089a0851dde0f54ba553f1027d5`
- Bound implementations: `bc0115cd46d33fe3b465489f8bd2014683747acf985743459e8c5f784c5e6504`
- Exclusion binding: `ff8ad38473edf26479480cd81c4bf21b704427eb6eef6e17be57d4e9aa51bca8`
- Scene-shard lineage: `33245a4fddea0ae304659100b98d6214bc93e25d78c4062e39be353605e5f1a9`

The adjacent frozen task census is bound, but is not used as a second verdict source:

- Raw SHA-256: `7ff35ec9feb864b1e9d6ef138a67874e6cce23e447e3de00d12773ca2ee56811`
- Semantic digest: `0ee5fb6d073e6e8db33b0f63ce9b70b8346ba12f29f729f06c06de5982fbe109`
- Scene-task-set digest: `ffe558fcf8d94bc0166f680c6362eaf54d3f77481c308081c8917c7eca975659`

The selector-feasibility runtime was 37,337.498021 seconds (10:22:17.498021). The final receipt is 1,194,515 bytes and the task census is 1,300,740 bytes, 2,495,255 bytes combined. This is not a total-run storage claim; scene-shard storage is deliberately excluded.

## Exact eligibility census

Every cell requires five distinct eligible scenes.

| Family | Scanned/allowed | General | Safety enriched | Completion enriched |
|---|---:|---:|---:|---:|
| `large_enclosed_maze` | 208/208 | 174/5 PASS | 174/5 PASS | 8/5 PASS |
| `local_composite_motifs` | 180/180 | 145/5 PASS | 143/5 PASS | 37/5 PASS |
| `loop_alias_stress` | 144/144 | 139/5 PASS | 139/5 PASS | 45/5 PASS |
| `medium_enclosed_maze` | 324/324 | 275/5 PASS | 275/5 PASS | 8/5 PASS |
| `open_obstacle_field` | 116/116 | 108/5 PASS | 49/5 PASS | 30/5 PASS |
| `rough_local_dynamics` | 66/66 | 63/5 PASS | 37/5 PASS | 16/5 PASS |
| `small_enclosed_maze` | 182/182 | 143/5 PASS | 143/5 PASS | **0/5 FAIL** |
| `visual_sensor_stress` | 64/64 | 55/5 PASS | 55/5 PASS | 13/5 PASS |

The exact per-family rejection maps are preserved in the companion JSON. For the failed small family they are: `already_in_disallowed_contact=2695`, `boundary_refused=3`, `completion_bearing_gt_75deg=9121`, `completion_geodesic_gt_0_75m=19900`, `completion_snapshot_goal_claimed=2195`, `completion_unreachable=6532`, `insufficient_proprioceptive_history=6`, `no_completion_enriched_goal=10211`, `no_reachable_landmark=8773`, and `no_stratum=148`. These are reducer rejection-event counts, not mutually exclusive scene counts. Exhaustive eligible-state and eligible-goal totals are unavailable: the receipt retains one projected evidence row from the first eligible state and its chosen goal per eligible scene/stratum. The decision claim is therefore the exact eligible distinct-scene census, not an unstored state/goal total.

## What the selector amendment resolved—and what it exposed

The authorized amendment did its intended job for the two families blocked by the superseded shared-landmark/graph-hop conjunction:

| Family | Completion eligible/required | Hop 0 | Hop > 0 | Verdict |
|---|---:|---:|---:|---:|
| `open_obstacle_field` | 30/5 | 30 | 0 | PASS |
| `rough_local_dynamics` | 16/5 | 16 | 0 | PASS |

Across all 157 completion-eligible scenes, 53 have diagnostic graph hops equal to zero and 104 have positive hops. Continuous geodesic distance for all completion-eligible scenes spans 0.11416539667902033–0.7496507633607222 m with mean 0.6582144947892661 m. Hop-zero scenes span 0.11416539667902033–0.7494468325520873 m with mean 0.5864912603025937 m; positive-hop scenes span 0.4434570640162134–0.7496507633607222 m with mean 0.6947657585180511 m. Full per-family min/Q1/median/mean/Q3/max summaries are in the companion JSON.

The amendment therefore succeeded for rough/open and uncovered a separate, genuine infeasibility under the amended frozen selector and exhaustive allowed corpus: the small-maze completion cell has no eligible scenes at all. The prior graph-discretization failure remains superseded; it is not being relabelled as this result.

## Lineage and terminal boundary

Active prospective lineage is bound to allocation amendment `4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc` (raw `1790429d6c02deebc794aa255be3b8c93ac5278de9c8c94920ee13b877fb5f38`), the current static allocation preflight `46efa42e3bdcad6df6cdcd4e404c2e8a796a9a331109a433cfbfffcfa18bf60d` (raw `a7f23011cdfec1f7a1938bfff57b4e6aa5f32b4e69082236c57d37a9ffd50256`), selector amendment `69e11a3efe665c4591fa29748b2f13ad08938b92acde763bda10608f93768628` (raw `907f23421cc0c4e22746b6fecc580bf4509b2cc904ef7f212800d2597795d663`), successor selection `8cf65cc016c28ad34f1e50246561e72ee9d0f9c1c253fe8e32a4203a35b73ebe`, allocator `bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e`, and candidate bank `85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9`.

The abandoned invalid-45 scene list (`5d5c4fef96e5132ad443c4fbd2778ad7d13fb9190328a498ca56490d53e041fe`) remains excluded by amendment `6d644c34b822fb5fb8e30906875047d1677aa730c2db584470cabdbe8bf6abc3`.

The earlier allocation failure (`550c52f9a3ff04f8a564f6f28e75e9d36fc8bc0f73da4795b95dedc3ad2e3cab`) and initial selector failure (`47c2bcc7cfaf79b328cd5a1cf2823554f2553fc419e020559fde1351df2ca75f`) remain preserved and superseded. The predecessor selection (`341de51facbb34b7361175bb713bbcef0fedb9cfc837a5adb6e2c888829a1df1`) was not reused as a successor result. The physical scorer contract still at its canonical path is the semantically superseded predecessor `06263907d8f8df0fe735f95da26c10fab9dff4af6827562622aa66463b456c0b` / artifact `116a7e77a7888788048a9fddcb3b7a1eaf62ea655890503ea09e08ebc91b898d` (raw `3ea9d04c4bf19e21713bac9d724581beac5b0ba41c1c4eb2ee5c017c785d0de2`, source `38e7fc84b83d815311ead732afa138c2179ccb11`); current validators reject it. The clean-launch receipt is likewise a stale predecessor (`7ab90a7fc6cdde04a0982701b008bc9d00b47ea8c0baecf47f775dcef6d64520`, raw `34dd039a3e6a3f9739ee0270c34c0f12eb6b831b6084acf5bf239f86832a939d`, source `38e7fc84b83d815311ead732afa138c2179ccb11`). The prospective successor scorer digest `8b1528cdd8b2c9ef0ee02cbdbc73040c4381968667903645be55fce8857e5aa4` was never issued as a contract because feasibility failed first.

The prior monolithic source attempt at commit `1779082a7bfef543e2dcb450d20c27ef220111f1` terminated with exit 139 / SIGSEGV. Its invalid-attempt record is preserved only for provenance (semantic digest `97fd6460e6d26c7dc1d6aa3e8c98a0c369b2c6faf12b32761aa4c95b4edc6dfa`, raw SHA-256 `377474e0e029c3f4ef289b3299896897f8304b2b9b4744dec0550bef084c4018`). It is non-binding, supplied no selector verdict, and contributed no scientifically reusable progress. The complete isolated run at commit `4dd906e7e82dd7c1530622a00319d1455c86b8e5` is the sole verdict source.

An independent exact rebuild of the final receipt passed and found no material implementation defect. A boundary audit confirmed the scorer tree has no phase-1 or phase-2 receipt, active manifest, branch rows, corpus receipt, frames, latents, training output, qualification output, package, or transfer output. This was validation only; it ran no science or downstream stage.

No automatic retry, selector change, contract change, performance-motivated response, or downstream phase is authorized by this report. Any successor would require separately reviewed prospective authority.

Companion JSON self-digest: `81637cdf3889dc0856ea97aee9a644f182855ef49c4e466eee3f8aed4134a0b8`.
