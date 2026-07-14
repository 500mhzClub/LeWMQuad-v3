# Observable camera-ray fit V4 N5 full-panel structural-trigger amendment

Date: 2026-07-13

Status: **frozen before successor implementation review or execution**

## Purpose

This additive amendment changes only the execution trigger in the frozen
full-panel successor preregistration. It does not change the experiment,
dataset subset, seed, model, objective, optimizer, budget, evaluation,
threshold, device, output namespace, attempt count, or authority boundary.

The predecessor preregistration is frozen at:

- path:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_successor_preregistration_2026-07-13.md`;
- file SHA-256:
  `0ad13e3897c70f90df6705538f4d86262ec53d3e096618a69563acdf63567c01`.

## Replaced trigger

The preregistration expected a narrow V3 finalizer to publish a structurally
valid unchanged-threshold N5 numeric failure. That trigger cannot occur. The
immutable N5 result fails the frozen structural loss invariant before metric
finalization: its matched total differs from one quarter of its four stored
components by `+3.067543719037502e-09`, exceeding the frozen absolute
tolerance of `1e-9`.

The exact terminal record is:

- path:
  `docs/lewm_go2_observable_camera_ray_fit_v4_prepublication_structural_invalidation_2026-07-13.json`;
- file SHA-256:
  `1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560`;
- content SHA-256:
  `7bdaae6ebb13b7d90290dfe07f5d48f403d29cad977f4a56c9ac7b8cfbcb8602`.

That record, rather than a V3 metric failure gate, is the sole predecessor
that licenses source implementation and different-agent review of the one
fresh full-panel successor. The invalid N5 result and checkpoint remain
ineligible for metric finalization, checkpoint use, later-rung authority, or
warm start.

## Unchanged experiment

The one authorized attempt remains exactly:

- output:
  `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5`;
- the same frozen seed-`20260710`, `N=5` subset;
- fresh `ObservableCameraRayEvidenceV4Model` initialization;
- AdamW, `400` optimizer updates, all five frames in every update;
- existing seeded concatenated-randperm schedule;
- learning rate `1e-4`, weight decay `1e-4`;
- float32, no autocast, global gradient clip `1.0`;
- the same four V4 losses weighted exactly `0.25` each;
- final-update selection only;
- matched-RGB and wrong-RGB evaluation at batch size one;
- GPU0 R9700 only, with GPU1/Raphael forbidden;
- one native thread per process and no more than five RGB workers;
- the existing frozen N5 scientific thresholds without exception.

The implementation must bind both this amendment and the exact terminal JSON
record before any RGB decode, model construction, GPU query, attempt
reservation, or output creation. It must use exclusive creation and publish a
terminal completion or failure receipt. A different agent must review and
hash-freeze every execution, verification, and finalization source before the
one command is licensed.

## Result rule

A structurally valid numeric failure is terminal and licenses no retry. A pass
licenses only design and different-agent review of the later-rung successor
schedule. It does not directly license `N=16`, seed `20260711`, V5 training,
G2, navigation evaluation, runtime, hardware, held-out access, or promotion.

No implementation or verifier may mutate, repair, reserialize, or canonically
finalize the invalid predecessor result. Counterfactual in-memory repair is
diagnostic evidence only and cannot enter successor inputs or output lineage.

## Access boundary

Implementation and review may read the frozen train-role N5 subset, reviewed
V4 sources, the terminal invalidation record, and the preregistration. They may
not open G2, held-out, sealed, checkpoint-selection, probability-calibration,
runtime, hardware, physical executor/reset, navigation benchmark, or
production-promotion inputs. Execution remains unavailable until the required
different-agent source review passes.
