# Physics-rate contact-proxy reconciliation V1

## Decision

**`PHYSICS_RATE_FULL_GEOMETRY_SCORE_NO_GO`**

The secondary finding is
`PREVIOUS_FALSE_POSITIVES_EXPLAINED_BY_TEMPORAL_ALIASING`.

This is a post-outcome development diagnostic. It preserves the historical
`H1_SAMPLED_DISALLOWED_CONTACT` target beside the new development target
`H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`; it does not rewrite any completed
result. Both are simulated separation/contact-avoidance proxies. Material
hazard remains `SEVERITY_UNRESOLVED`: no injury, property-damage, human,
fragile-infrastructure, acceptable-impact, or closed-loop claim is supported.

## Frozen evidence and dual labels

The diagnostic started from commit
`5d440911774682f351b8ab7192c89b453226328b` and used the 24 calibration and
24 held-out `wide-*` states, 12 candidates per state, five H1 command ticks,
and 250 persisted 2 ms physics steps per branch. The input geometry ledger
SHA-256 is
`827263fa58aaf782daddcca9c935173f46a0b4c44a672549cbc2daf8b4a7eea5`.
No replay, learned inference, or new identity was needed.

The exact sampled-versus-physics confusion matrices are:

| Split | sampled+/physics+ | sampled+/physics− | sampled−/physics+ | sampled−/physics− |
|---|---:|---:|---:|---:|
| Calibration | 31 | 0 | 59 | 198 |
| Held-out | 53 | 0 | 88 | 147 |

Thus the sampled label has perfect specificity but only 0.3444 calibration
and 0.3759 held-out sensitivity to any physics-step contact. It misses 65.56%
and 62.41% of physics-positive branches, respectively. Held-out prevalence
rises from 53/288 sampled positives to 141/288 physics-rate positives.

The held-out traces contain 3,628 positive physics steps grouped into 523
contiguous events. Event duration is 1–33 steps (median 6, mean 6.94); the
244 events on sampled-negative/physics-positive branches are 1–20 steps
(median 6, mean 6.52). Only 8.20% of those missed events last one physics
step, so the mismatch is not mostly single-step numerical flicker. Nearest-
geometry body-region counts among the 88 missed branches are trunk 36,
front limb 28, and rear limb 35; a branch can contribute more than one
region. Impulse and relative speed are unavailable in this persisted panel.

Held-out sampled/physics/missed branch counts by family are large 7/25/18,
loop 3/27/24, medium 22/48/26, and small 21/41/20. Temporal aliasing is
therefore material in every family.

## Physics-rate geometry results

Risk remains negative minimum clearance. Thresholds were selected using only
the 24 calibration states, then frozen.

| Condition | clearance threshold (m) | AUC | AP | recall/FNR | negative retention | retained states | selected contacts | false abstentions | progress / oracle | regret | top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full articulated scene | −0.004124 | 0.9375 | 0.8960 | 0.9504 / 0.0496 | 0.8095 | 11/24 | 1 | 6 | 0.2013 / 0.1704 m | 0.1776 | 0.4706 |
| Front depth sweep | 0.868471 | 0.8118 | 0.7922 | 1.0000 / 0 | 0.1633 | 2/24 | 0 | 15 | 0.1739 / 0.1704 m | 0.0526 | 0.1176 |
| LiDAR sweep | 0.119507 | 0.8458 | 0.7721 | 0.9645 / 0.0355 | 0.6463 | 9/24 | 2 | 8 | 0.1934 / 0.1704 m | 0.1018 | 0.3529 |
| Depth + LiDAR sweep | 0.119507 | 0.8548 | 0.7877 | 0.9645 / 0.0355 | 0.6463 | 9/24 | 2 | 8 | 0.1934 / 0.1704 m | 0.1018 | 0.3529 |

Progress fractions above one do not represent safety success: unsafe
selections and different availability denominators make them possible. The
full condition fails its 0.99 AUC/AP/recall upper-bound requirements, retains
only 11 states, admits one all-positive state, and selects one physics-contact
candidate. Consequently sensor conditions cannot qualify as a solution.

Full-condition recall/retention by family is large 0.7600/0.8723, loop
0.9630/0.9778, medium 1.0000/0.5000, and small 1.0000/0.7097. Retained states
are 4, 4, 1, and 2; the selected contact is in the large family. LiDAR and
fusion recall falls below 0.95 in large and loop, and only nine states remain
available overall. Depth collapses large and medium availability completely.

## Threshold-capacity frontiers

No held-out threshold satisfies a complete gate for any condition.

| Condition | thresholds | max retention at recall ≥0.95 | max states at recall ≥0.95 | max progress with zero selected contacts |
|---|---:|---:|---:|---:|
| Full articulated scene | 167 | 0.8095 | 11 | 0.2303 m |
| Front depth sweep | 217 | 0.3673 | 7 | 0.1739 m |
| LiDAR sweep | 211 | 0.7211 | 10 | 0.2614 m |
| Depth + LiDAR sweep | 212 | 0.7211 | 10 | 0.2614 m |
| Persisted wide learned score | 252 | 0.3401 | 8 | 0.2630 m |

The physics-rate label also reveals a candidate-bank limitation: seven of 24
held-out states have no physics-contact-negative candidate, so the requested
22-state full-geometry availability gate is unattainable on this panel even
with oracle labels. This does not change the frozen gate; it is part of the
diagnosis.

## Solver consistency

At the calibration-selected full-geometry threshold there are 35 held-out
decision disagreements. All are assigned `GEOMETRY_QUERY_MISMATCH` because
the reducer shares the Genesis URDF primitive source but does not reproduce
Genesis's exact MPR/GJK narrowphase: it uses analytic sphere/box, SAT box,
and a 33-sample capsule-axis query. Six physics-positive disagreements have
solver contact while this approximate query reports 2.4–6.9 mm positive
clearance. The other disagreements are query-positive/solver-negative.

The directly relevant Genesis 0.3.14 narrowphase uses zero margin for these
rigid primitives; no configured skin offset was identified. However,
penetration and contact-manifold evidence were not persisted, and the exact
solver query was not reproduced. Accordingly, neither collision margins nor
contact dynamics can yet be isolated as the residual cause. It would be
scientifically premature to authorize `ARTICULATED_CONTACT_DYNAMICS_STATE_V1`
from this approximate-query disagreement.

## Existing learned score

The persisted H1 `WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1` logits can be
lawfully aligned to the current panel without checkpoint execution. Against
the physics-rate target they yield AUC 0.7486, AP 0.7644, recall 0.9574,
negative retention 0.2313, seven retained states, two selected physics
contacts, and 0.1478 m progress. No held-out threshold passes its complete
gate.

Of 195 candidates that this model rejected while the sampled label called
them negative, 82 (42.05%) are physics-rate positive. This supports
`PREVIOUS_FALSE_POSITIVES_EXPLAINED_BY_TEMPORAL_ALIASING` descriptively, but
does not make the completed model pass. Persisted `DEPTH_ONLY`, `LIDAR_ONLY`,
and `DEPTH_PLUS_EMBODIED` logits are bound to the older `scale-*` rows and
have zero identity overlap with this `wide-*` panel; checkpoint execution was
prohibited, so no result is fabricated for them.

## Decision and next experiment

Temporal aliasing is confirmed and materially improves the correspondence of
full geometry (held-out AUC rises from the earlier sampled-target 0.7756 to
0.9375), but the strict full-geometry gate still fails. The exact primary
classification is therefore `PHYSICS_RATE_FULL_GEOMETRY_SCORE_NO_GO`.

`ARTICULATED_CONTACT_DYNAMICS_STATE_V1` is **not yet justified**. The next
bounded experiment should first freeze
`H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT` and run an exact Genesis-congruent
MPR/GJK clearance/contact-manifold persistence audit on the existing panel.
Only residual disagreement after matching primitive, margin, manifold, and
query conventions would justify a dynamics-state model. No learned geometry
predictor should be trained from this result.

The reusable 576-row ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/physics_rate_contact_proxy_reconciliation_v1/row_level_evidence_v1.npz`,
30,140 bytes, SHA-256
`3e5de8b6b4007f9ac066bb981e23f9fc59b28459caa23d93c9c222431b18b8ee`.
It contains both labels, event timing and durations, nearest-geometry link
attribution, every score and threshold decision, selected candidates, and
planning inputs. Evaluation took approximately 2.4 seconds and created 53,821 bytes across
the ledger and five frontier files.

No training, learned inference, simulation, replay, rendering, encoding, new
panel generation, JEPA access, memory, novelty, routing, or navigation work
occurred. Nothing remains running.
