# Go2 categorical-radial N32 V4 result and adjudication

Date: 2026-07-11

Status: authoritative fit-only negative result. No holdout, checkpoint-selection,
probability-calibration, physical-nontrain, G2, runtime, or sealed payload/model
output was opened.

## Bound result

The authoritative seed-20260710 result is:

- path:
  `.generated/go2_categorical_radial_n32/v4/seed_20260710_result.json`;
- file SHA-256:
  `d4736b76e354c63268ee7698cacc0ae1834b888407c32095f22b562ce1726789`;
- canonical content SHA-256:
  `719841ac72d09f6240be59a26fdcab059ed070bc4b7cccf3fa79ddbfa2be5103`;
- frozen V4 binding SHA-256:
  `bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3`.

The strict artifact records `classification=fit_gate_failed`, no qualifying
optimizer stage, no holdout decision, and no G2, G3, runtime, full-training, or
promotion license. All 20 evaluations from update 100 through update 2,000
failed the all-family gate. In particular, updates 1,800, 1,900, and 2,000 all
failed, so the exact terminal-three rule failed.

The access ledger records 320 fit-image decodes and 20 fit-label-shard opens.
Both train-role holdout panels remained unauthorized with zero artifact hashes,
byte opens, model calls, or model outputs. Checkpoint-selection,
probability-calibration, G2, and all physical-nontrain counts are zero.

## Terminal result

At update 2,000 the aggregate report passed every threshold:

- hierarchical balanced NLL: `0.0126775` (`<=0.03`);
- UNKNOWN/KNOWN balanced accuracy: `0.991480` (`>=0.99`);
- conditional FREE/OCCUPIED balanced accuracy: `0.999844` (`>=0.99`);
- UNKNOWN/FREE/OCCUPIED recall:
  `0.984616 / 0.999428 / 0.983672` (each `>=0.98`);
- cross-scene and same-scene wrong-view NLL deltas:
  `4.54186 / 3.28850` (each `>=0.25`).

The mandatory family rule failed in three of five families:

| Family | Failed metric | Result | Exact consequence |
| --- | --- | ---: | --- |
| open obstacle field | OCCUPIED recall | `0.959083` | 586/611 correct; 25 OCCUPIED cells became UNKNOWN; 13 must be recovered to reach 0.98 |
| rough local dynamics | UNKNOWN recall | `0.970032` | 200,880/207,086 correct; 6,206 UNKNOWN cells became known; 2,065 must be recovered to reach 0.98 |
| rough local dynamics | UNKNOWN/KNOWN balanced accuracy | `0.984491` | below 0.99 by `0.005509` |
| small enclosed maze | OCCUPIED recall | `0.977273` | 2,365/2,420 correct; 55 OCCUPIED cells became UNKNOWN; 7 must be recovered to reach 0.98 |

Large and medium enclosed mazes passed. Every critical failed-family metric
reached its best recorded value at update 2,000, but no evaluation ever passed
the complete gate. The precommitted budget was fully consumed; this trend does
not authorize extra updates.

## Failure class

The failed cells are not principally FREE-versus-OCCUPIED errors. Conditional
FREE/OCCUPIED confusion at update 2,000 was:

- open obstacle field: `[[50049,3],[0,611]]`;
- rough local dynamics: `[[51749,3],[0,3306]]`;
- small enclosed maze: `[[7537,12],[0,2420]]`.

All OCCUPIED targets in these conditional tables rank OCCUPIED over FREE. The
joint OCCUPIED misses are rejected as UNKNOWN. Rough terrain fails in the
opposite direction: too many UNKNOWN targets are admitted as known. The
remaining conflict is therefore spatially varying UNKNOWN-versus-KNOWN
grounding, not conditional obstacle classification.

The exact V4 output factorization is not sufficient. Against the width-24 V2
control, V4 worsened every critical terminal recall:

| Candidate | Open OCCUPIED | Rough UNKNOWN | Small OCCUPIED | Aggregate hierarchical NLL |
| --- | ---: | ---: | ---: | ---: |
| V2 width 24 | `0.978723` | `0.974518` | `0.982645` | `0.0110524` |
| V3 width 32 | `0.960720` | `0.972079` | `0.975620` | `0.0121352` |
| V4 explicit output hierarchy | `0.959083` | `0.970032` | `0.977273` | `0.0126775` |

V4 also worsened aggregate UNKNOWN/KNOWN weighted NLL from V2's `0.0211737`
to `0.0240046`. The two output factors remove cross-gradients only at the last
two-channel head; both objectives still share the encoder and complete context
decoder. This result rejects last-layer coupling as the sufficient cause.

The earlier V1 ceiling terminal had NLL `0.055060` and
UNKNOWN/FREE/OCCUPIED recall `0.954089/0.984418/0.916895`, but it received only
62.5 effective epochs. V2 supplied the registered 500-epoch exposure control
and moved the failure to the narrow family-specific boundary above. Exposure
was therefore tested rather than assumed.

## Hypothesis adjudication

**Representation: supported as the next causal hypothesis.** The current
radial context runs along body-centered polar columns. Because the camera is
0.326 m forward of the body origin, a body-polar column is not one camera ray;
its image bearing changes with range. Observable-physical-v3 labels also
aggregate multiple vertical rays and physical-cell supports. The full-ray
context can therefore mix evidence that is not observationally collinear.

**Objective weighting: possible secondary issue, not the next licensed
change.** Training minimizes globally weighted cell risk while promotion
requires every family to pass. A family-robust objective could change the
trade, but selecting weights from these failed family values would be
post-result tuning and would add no missing camera-aligned information. It is
not evidence that the existing representation can express the required
spatially varying boundary under the registered optimization.

**Label observability: unresolved prerequisite risk.** The v3 target uses
multi-point visible-floor support, sparse 3-D first-surface witnesses, and a
collision-geometry fail-closed veto. Exact 0.10 m labels may contain
image-resolution aliasing, and any veto caused by collision geometry absent
from the rendered observation would be unlearnable from RGB. This result does
not authorize relabeling or gate relaxation. A fit-only source/geometry audit
must prove the next target remains observable before trained output.

**Capacity: the simple hypothesis is rejected.** N=1/4/16 passed, while the
registered width-32 V3 change worsened all critical metrics and reduced the
number of passing families. This does not prove that every architecture has
enough capacity; it does reject another undirected width retry.

**Exposure: rejected as the next intervention.** V2, V3, and V4 each received
500 effective epochs and 160,000 frame presentations. The V4 failure clause
explicitly does not license more exposure.

**Calibration and thresholds: rejected.** A common KNOWN bias has an empty
feasible interval because open OCCUPIED recall requires more admission while
rough UNKNOWN recall requires less. The frozen proof SHA-256 is
`e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a`.
Post-hoc family thresholds and gate changes are forbidden.

## Single next intervention

Preregister one camera-centered, true-frustum categorical representation. Its
range context must follow actual camera rays rather than body-polar columns,
retain multiple vertical evidence channels, and deterministically aggregate
or gather those predictions into the unchanged 64 x 64 body-local
observable-physical-v3 target. A scalar first-hit depth profile is not an
admissible substitute because it cannot represent the target's multi-height,
partial-support, occlusion, and obstacle-witness semantics.

Keep the fit data, fixed deployment camera calibration, encoder size, loss,
optimizer schedule, controls, gates, and conditional-access rules fixed. Run a
fit-only observability/geometry audit before freezing implementation, then
repeat the N=1/4/16/N32 ladder. Only an exact seed-20260710 fit pass may open
the two train-role holdouts, and only a fully favorable seed-20260710 result
may authorize seed 20260711.

Seed 20260711 is currently forbidden.
