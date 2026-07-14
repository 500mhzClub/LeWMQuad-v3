# Camera-ray N5 hierarchical-first-hit V9 pre-implementation amendment

Date: 2026-07-13

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and boundary

Camera V8 completed its sole exact seed-`20260710`, N=5 attempt and failed the
unchanged numerical gate. V8 is terminal. This amendment does not retry V8,
reuse its checkpoint, weaken its thresholds, repair its result, or authorize a
later fit rung.

The read-only diagnosis preceding this amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_v8_numeric_failure_diagnosis_and_successor_design_2026-07-13.md`

Its SHA-256 is
`ece7c960f49748776cd73f029e144f91f4c0723908e234a7e71173047777ee9a`.

This amendment is the first V9 artifact. No V9 policy, model/loss module,
trainer, verifier, executor, test, handoff, review, output root, checkpoint, or
metric existed when these bytes were frozen.

## Frozen predecessor evidence

| Role | Path | SHA-256 |
| --- | --- | --- |
| V8 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_isolated_verifier_amendment_2026-07-13.md` | `9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211` |
| V8 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py` | `99a2777d3ba2ad8baf62b98944f05aa1affb2e74834f337a2ba0644e9c03c84c` |
| V8 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py` | `f163aaf04722bb118796912bcfcdf1f4e24b7e54990e41a9d164acc08b233500` |
| V8 independent review | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_independent_review_2026-07-13.json` | `fd095eea8b1f2a0cde67f77a3bd2338f8f13e3a81d824777475600a258758f0f` |
| V8 result | `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v8/attempts/seed_20260710/n5/result.json` | `fc89f0e9cfdabc13e4bbac0053dabf559d2491c6c45e82c6d7cdc0468ba7e2d0` |
| V8 metric verification | `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v8/metric_verifications/seed_20260710_n5.json` | `b28cbd3795d090d652504a4721216689f160577e7947e3671b04688e39ae6b89` |
| V8 gate | `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v8/gates/seed_20260710_n5.json` | `cfe39b64e496bbd7bf4a2b0144bffee884c9b2ceca18d1d5275f41492633c081` |

The V8 metric and gate content SHA-256 values are
`c3bf90bc16bff983232d9a23de20a881637233e5e3b4723f5134769a2d5c7090`
and `11f02aa3fb51b217d4b2a18544582f42f4593a44c01b28cd733df4f6873f4ddf`.
The V8 gate records 19 passes and seven failures, with all retry, checkpoint,
N16, later-rung-design, V5-training, G2, held-out, runtime, hardware,
production, and promotion licenses false.

V9 may rehash the listed predecessor bytes as static authority. It may not load
the V8 checkpoint, read unverified V8 numeric state, or use any V8 value for
initialization, checkpoint selection, calibration, post-processing, or metric
repair.

## Frozen scientific change

V9 changes exactly the first-hit training objective and the fixed convergence
budget. It preserves:

- the observable-camera-ray V4 target and physical semantics;
- seed `20260710` and the exact same five train frames, one per family;
- fresh `ObservableCameraRayEvidenceV4Model` initialization;
- image normalization, camera calibration, model architecture, parameter
  count, ordered hazard output, bounded offset output, ground-query branch, and
  differentiable physical rasterizer;
- AdamW, learning rate `1e-4`, weight decay `1e-4`, float32, no autocast,
  gradient clipping norm `1.0`, and full-panel batch size five;
- within-bin offset, balanced ground-clear, and hierarchical raster losses;
- matched-RGB and wrong-RGB-with-target-calibration evaluation;
- all existing N5 metrics, arithmetic, and 26 thresholds; and
- R9700 GPU0 only, Raphael GPU1 forbidden, no HSA override, at most five RGB
  workers, and one native math thread per process.

### Hierarchical first-hit loss

Let the existing ordered output define normalized `P(no_hit)` and
`P(hit_at_d)` for depth bin `d`. Define:

1. `P(hit) = sum_d P(hit_at_d)`;
2. `presence_nll` as the mean of the nonempty target-no-hit mean NLL and the
   nonempty target-hit mean NLL, so the two binary states receive equal weight;
3. `conditional_depth_nll` as the equal-weight mean over represented target
   depth bins of `-log(P(hit_at_d) / P(hit))` on target-hit rays; and
4. `hierarchical_first_hit_nll = 0.5 * presence_nll + 0.5 *
   conditional_depth_nll`.

All probability arithmetic stays in log space with finite checks. Empty groups
retain a zero-valued gradient term. Conditional depth must be invariant to a
common change in hit-presence mass, and presence must be invariant to how hit
mass is distributed among depth bins.

The four top-level losses are weighted exactly `0.25` each:

- `hierarchical_first_hit_nll`;
- `target_bin_offset_smooth_l1`;
- `ground_clear_distance_state_balanced_bce`; and
- `derived_raster_hierarchical_bce`.

No old loss may be reported under the new name, and no new loss may be reported
under the old `ordered_first_hit_nll` name. Training, result, checkpoint,
verification, and gate schemas must bind the new loss contract explicitly.

### Convergence schedule

V9 uses exactly 4,000 full-panel optimizer updates and 20,000 frame exposures.
The deterministic schedule contains each of the five frames exactly once per
update. Diagnostics are recorded at update 1, every 100 updates, and update
4,000. The update-4,000 state is the only eligible checkpoint; best-loss,
gate-based, early-stopped, averaged, or repaired checkpoint selection is
forbidden.

The exact schedule and its SHA-256 must be constructed and frozen in reviewed
source before execution. A source-only CPU test must prove update count, frame
exposures, full-panel membership, trace indices, final-only selection, and
determinism.

## Required implementation boundary

The V9 candidate must add, at minimum:

1. a pure loss module or versioned model-training module implementing and
   validating the hierarchical first-hit breakdown;
2. a standard-library-only policy binding every predecessor, source, schema,
   science, schedule, threshold, resource, access, cleanup, and license field;
3. one exact trainer that starts from fresh initialization and records the new
   objective without reading V8 model state;
4. one independent metric verifier that loads the V9 checkpoint only for
   recomputation, recomputes matched and wrong-RGB evidence, and invokes the
   unchanged frozen N5 metric gate;
5. one lifecycle-owning executor that preserves V8's no-follow
   descriptor-relative transaction, complete owned-directory journal,
   terminal cleanup, and fresh isolated compute-only verifier child; and
6. CPU-hidden author/adversarial tests plus a frozen handoff for a different
   reviewer.

The canonical root, if later authorized, is:

`.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v9`

The root, its parents below the already-existing camera root, source review,
reservation, attempt, checkpoint, result, metric, and gate must all have one
fixed path and one-attempt semantics. Publication must be exclusive, durable,
descriptor-relative, journaled, and fully revalidated after every rename and
at the final quiet boundary.

## Required source-only evidence

Before different-agent review, CPU-only tests with all accelerators hidden must
prove:

1. exact presence and conditional-depth arithmetic against hand-computed
   distributions;
2. equal hit/no-hit influence even when many hit-depth bins are represented;
3. presence invariance to conditional redistribution and conditional-depth
   invariance to presence mass;
4. correct finite gradients for target hit, target no-hit, and every represented
   depth bin, including extreme logits;
5. the old `1/(G+1)` no-hit group weight is not retained accidentally;
6. unchanged raw model outputs, calibration, physical raster semantics, metric
   accumulation, wrong-RGB mapping, and 26 thresholds;
7. exactly 4,000 updates, 20,000 exposures, and final-update-only selection;
8. fresh initialization with no V8 checkpoint or result input;
9. exact isolated-child environment, request/response binding, parent-only
   publication, no fallback, timeout/nonzero/signal/stderr/extra-output failure,
   terminal cleanup, and no retry; and
10. zero canonical output, experiment RGB/data, checkpoint, model-output, GPU,
    G2, held-out, selection, calibration, runtime, hardware, production, or
    promotion access during authoring and review.

## Review and execution sequence

1. Freeze this amendment before any V9 source.
2. A non-amendment author constructs and freezes V9 source and a handoff without
   exact/data/GPU work.
3. A reviewer different from both amendment and implementation authors audits
   the complete frozen candidate and publishes canonical PASS or BLOCK evidence
   last.
4. Only a PASS review may authorize one exact V9 attempt. The command must be
   serialized against every other `.generated` mutator.
5. The exact attempt passes only if all unchanged N5 checks pass. Failure is
   terminal and grants no retry or downstream license.
6. A pass may license design and different-agent review of the next fit rung;
   it does not itself authorize N16, a second seed, V5 training, checkpoint
   promotion, G2, held-out, runtime, hardware, production, or deployment.

## Explicit non-authority

This amendment does not authorize implementation by `/root`, exact execution,
data/RGB opening, checkpoint loading, GPU use, V8 retry, V8 checkpoint use,
threshold or calibration change, capacity change, N16, second seed, shared-JEPA
training, selection, calibration, G2, held-out navigation, runtime, hardware,
production, promotion, or deployment.
