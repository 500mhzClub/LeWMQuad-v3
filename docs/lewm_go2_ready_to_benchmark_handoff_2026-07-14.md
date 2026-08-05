# Go2 ready-to-benchmark handoff

Date: 2026-07-14 05:09 BST

Status: **stopped by user request; no active experiment or test process**

## Objective

Continue toward a reviewed JEPA plus online-memory navigation stack that can
cover novel mazes and claim their beacons. The requested stopping point is
"ready to benchmark": a frozen trained candidate, a real reviewed development
runner, passing integration gates, and a fresh opaque held-out set. This
handoff does not claim that point has been reached.

## Plain-language state

The raw supervision dataset has passed its exact independent audit. Its labels
can now be treated as faithfully built, subject to a future training contract
explicitly granting dataset-use authority.

The small Camera V11 fit learned the task extremely accurately but failed one
of 26 frozen checks: all-cell raster NLL was `0.07255925759673118` against a
required maximum of `0.06`. Camera V12 added the preregistered gate-aligned
raster NLL, but source review blocked it before execution because nested review
bindings accepted undeclared fields. Camera V13 is the governance-only repair.
Its implementation was interrupted while being reconstructed and is **not a
frozen or runnable candidate**.

There is still no reviewed entrypoint that connects a qualified shared V5
checkpoint to the two-resolution online memory, explorer, router, follower,
and claim evaluator. The legacy closed-loop benchmark is not that entrypoint.

## Terminal Raw V13 PASS

The only authorized Raw V13 audit ran once, CPU-only, with six workers. It
exited `0` after approximately 59 seconds; both GPUs remained unused. No retry
or other mutator ran.

- Dataset: `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
- Audit receipt: `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v13.json`
- Audit receipt file SHA-256: `0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76`
- Audit receipt canonical content SHA-256: `0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca`
- Dataset manifest file SHA-256: `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`
- Dataset manifest content SHA-256: `74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`
- Population: 5,172 pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 scene shards, 354 source files
- Split pair counts: train/checkpoint/calibration = 4,262/495/415
- All 24 precommitted geometry samples passed, covering all eight families in all three roles
- Sample-results SHA-256: `a051b9a0a10f14413105f2f1cc3c36ad10a43ec20071f0577efcc99fc321d356`
- RGB reads/decodes, checkpoint/model/G2/held-out/hardware/production opens: zero
- Raw V13 authorization file SHA-256: `8a12c5f8d6c6e64a418052cf01177dd25049d6d373f7e87cd52c5d2a5b2bf587`
- Authorization content SHA-256: `4b179c33de00399652f4f915285ca99a4d47cfa95d31878d1f91ca7e8fd9d0e8`
- Authorization source-map SHA-256: `88f748865ff132bc7afd6fe85def14d7f3180ce86b1304ce195c7220f75b8996`

The PASS proves the dataset bytes and sampled geometry. It does not by itself
authorize training, selection, G2, navigation, held-out use, or deployment.
Do not rerun Raw V13.

## Camera evidence before V13

Camera V11 terminally completed one exact GPU0 attempt and passed 25/26 checks.
Relevant values were:

- Raster balanced accuracy: `0.9939025862808951`
- Unknown/free/occupied recall: `0.9894560565651553` / `0.9922517022775299` / `1.0`
- Hit balanced accuracy: `0.9999430156137219`
- Hit-depth median/p95 MAE: `0.002992570400238037` / `0.010697221755981447` metres
- Ground balanced accuracy: `0.9998668840676225`
- Sole failure, raster NLL: `0.07255925759673118 > 0.06`

Camera V12 retained V11 science and added exact
`0.25 * derived_raster_cell_nll`. Its source review was a terminal BLOCK, not a
numeric attempt:

- V12 amendment SHA-256: `77de8c69b1bef69ab3d1b976567eb20371f53d47d81af757ef8c7fdaade93c1b`
- V12 review file SHA-256: `076855183730bcff58b507d8fde6c613a023b633681c7516daaf0d80b5e27158`
- V12 review content SHA-256: `4a56c46ede9482f72b5ae304734e12a706d8f7075873b4e5de135f9fa6cc289d`
- Blocking defect: nested source/proof bindings used `Mapping` and `.get(...)`
  without requiring a plain `dict` with exactly `{"path", "file_sha256"}`.
- No V12 exact attempt or output exists.

The source-free Camera V13 amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_strict_review_binding_successor_amendment_2026-07-14.md`

Its SHA-256 is
`2eaaaa7b896dd42bed02d5a75072d1933b11ad4cce5e8d83f35f1d137ba89633`.
It permits only the strict review-binding repair and preserves every V12
tensor operation, loss, schedule, metric, threshold, and lifecycle rule.

## Interrupted Camera V13 source

The fixed implementation role is
`/root/camera_v12_gate_aligned_implementer`. That agent was interrupted by the
stop request. All current V13 files must be treated as mutable, partial, and
unreviewed. No tests were completed after the reconstruction began, no handoff
was produced, no independent source review exists, and no V13 output directory
exists.

Forensic hashes at the stop boundary, **not approved source identities**:

| Role | Path | Stop-boundary SHA-256 |
|---|---|---|
| policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `100f2cce42d1966a71fa55e10a90f35cd8f36a9de9b76778772d337ae11907f5` |
| trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `92d6fef2a32498b4dc80566f73422b3735d2d9bbb39612b8a8946d7aa3a34d43` |
| verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `7fe1fa1f107478303c10cecd0b591388e1fdb042e14f0ad289f0b36ee399686b` |
| executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `77d7782078dc8b089f97144117d7dd0d8d0116dbfbe55a8b665335ee9de55a54` |
| synthetic proof | `lewm/tests/n5_gate_aligned_raster_nll_v13_synthetic_execution.py` | `19c6a1897b247760653c1329e46d389ab7a1b760074967f0e29ace9a19fd36b3` |
| science test | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py` | `2ebac0d62fa6c67e97ff174b301882cce73bda3b0f11bfa008ef23ff20745596` |
| lifecycle test | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py` | `851e661a951df3de931828ff797d3d718eba6acbc850cec34f08e6a93ba89352` |

Known unfinished condition at interruption: the policy's
`preflight_source_review` still contained permissive `Mapping`/`.get(...)`
checks near its source/proof loops. The implementation handoff file was absent.
Do not review or execute these bytes.

The retained V12 model/loss file is the scientific source of truth:

`lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py`

Its frozen SHA-256 is
`735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662`.
V13 must bind and import that exact file; it must not create or expect a V13
model/loss copy.

## Full Training V3 readiness audit

The interrupted read-only audit found that Full Training V2 is a usable
structural base, but it is not eligible for the new pipeline. V3 must:

1. Bind the terminal Raw V13 PASS report, Builder V9, Auditor V13, and their
   review, authorization, and fingerprint-witness chain instead of legacy V1
   audit paths and schemas.
2. Replace `ordered_first_hit_nll` with the retained hierarchical first-hit
   objective and include Camera V13's additive gate-aligned raster NLL if V13
   passes.
3. Freeze and test reduction semantics: compute current and next B=4 scalar
   losses separately, average the pair 0.5/0.5, then average four microbatch
   scalars equally per update. Never pool nonlinear groups over a synthetic
   B=16 batch.
4. Add a distinct strict pre-G2 candidate checkpoint schema. The current
   `qualified_checkpoint.pt` path incorrectly claims the post-G2 V5 schema
   before lifecycle/provenance/G2 qualification exists.
5. Replace legacy `development_fit_v2` camera bindings with the future Camera
   V13 two-seed N5/N16/N32/N320 ladder and its primary N320 checkpoint.

Raw bindings, loss/reduction code, schema code, V3 namespaces, and CPU tests can
be prepared before the camera ladder completes. Camera gate, ladder, and N320
hash fields must remain absent/unresolved until their real artifacts exist.

## Navigation benchmark readiness audit

There is no valid current benchmark entrypoint for the reviewed V5 design.

- `scripts/benchmark_go2_memory_closed_loop.py` and
  `scripts/run_go2_generalized_learned_local_suite.sh` are legacy. They load
  old hidden-target/primitive/local-policy checkpoints and do not construct the
  reviewed 0.05 m physical to 0.10 m configuration projection/coordinator.
- `lewm/planning/two_resolution_navigation_development_integration_v3.py`
  remains API/test-only and rejects non-synthetic construction.
- `lewm/planning/native_learned_physical_projection_v5.py` is also
  synthetic-only.
- The available 138 train plus 24 visible development scenes are valid for
  iteration. Use the 24 development scenes for fast probes.
- The legacy 18-scene `phase4_full18` table is development-only and
  leakage-contaminated. It cannot support a new held-out claim.
- The old sealed 30-scene role was opened and is permanently invalid as a final
  blind evaluation set. A fresh opaque held-out split must be created only
  after the development stack is frozen.

The top downstream implementation blocker is a real development runner that
loads the qualified shared V5 checkpoint, runs one learned inference per tick,
projects physical evidence into revisioned two-resolution memory, invokes the
frontier/target router and follower, and logs canonical coverage, visibility,
claims, collisions, and an actual-open ledger.

## Restart order

1. Resume the same fixed Camera V13 implementation role. Reconstruct every V13
   production/proof file from the frozen V12 closure and amendment. Make the
   nested source and proof checks require `type(binding) is dict` and exact
   keys before consuming either value.
2. Run CPU-only V13 tests, then the frozen V12 `202/202` and V11 `190/190`
   suites, isolated-child smoke, compilation, whitespace, AST parity, no-open,
   and adversarial extra/missing/subclass/nonstring/role-swap cases.
3. Freeze source hashes and create the V13 implementation handoff.
4. Assign a different eligible agent to independently review the exact closure.
   The author cannot self-review.
5. Only after a canonical PASS, assign a third agent to the one authorized
   Camera V13 N5 execution on discrete GPU0. Keep GPU1 unused. There is no
   retry. Require all 26 frozen checks, especially raster NLL at or below
   `0.06`.
6. If Camera V13 passes, preregister/review and execute the fresh-init two-seed
   N16/N32/N320 scaling ladder. Do not warm-start or tune against observed
   gate values.
7. In parallel where source-only, implement and review Full Training V3 with
   the five corrections above. Bind real camera artifacts only after they
   exist.
8. Train the matched JEPA and no-JEPA controls, perform allowed development
   selection/calibration, and qualify the shared checkpoint through G2.
9. Implement and independently review the real development runner. Prove its
   one-inference-per-tick, memory revision, route/claim, reset, and logging
   contracts with synthetic tests before development simulation.
10. Iterate coverage, sighting, routing, and claiming only on development
    scenes. Once frozen, materialize a new opaque held-out set and run the final
    multi-seed benchmark without tuning on it.

## Resource and stop state

- Subagents interrupted: Camera V13 implementer, Full Training V3 readiness
  auditor, navigation benchmark readiness auditor.
- Active audit/training/pytest/benchmark processes: none observed at stop.
- Camera V13 canonical output directory: absent.
- `.generated` mutators: none active.
- CPU work should use at most six workers with native math threads capped at
  one.
- Neural training/inference should use discrete GPU0 only. Keep the Raphael
  iGPU/GPU1 unused.
- Serialize every `.generated` mutator even when source and review work runs in
  parallel.

The worktree contains extensive pre-existing modified and untracked research
files. Do not reset, clean, or revert unrelated changes. Treat the forensic V13
hashes above only as a stop marker, never as approval.
