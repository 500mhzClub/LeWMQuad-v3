# GEOMETRY_FUSION_CONTACT_ERROR_ATTRIBUTION_V1

Date: 2026-08-21  
Source commit: `06b1ffb8232476456f21fa8fd56284230f26d7c8`  
Status: `POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC`  
Preserved result: `GEOMETRY_MODALITY_POSITIVE_TENDENCY`

Primary classification: **`CONTACT_REQUIREMENT_ONTOLOGY_REQUIRES_REVISION`**

Secondary findings:

- `FUSION_SCORE_FRONTIER_NO_GO`
- `LIDAR_COMPLEMENTARY_TENDENCY`

## Decision

Threshold calibration alone cannot rescue the frozen depth-plus-embodied score: none of the 290 held-out thresholds satisfies the original complete safety–mobility gate. LiDAR is strongly complementary at the frozen scores—it rejects all eleven fusion false negatives and both selected fusion contacts—but the frozen branch evidence cannot establish that these contacts are materially hazardous, low-energy/recoverable, front/side/rear, or even directly visible at the inferred range return. Every contact lacks link-resolved contact position, calibrated disallowed-contact force/impulse, relative contact speed, penetration, damage relevance, and object fragility.

The single next action is therefore to complete and prospectively freeze the contact-hazard ontology before training another model or collecting another panel. Required evidence is link/body-region resolved disallowed contact, calibrated force and impulse, relative contact speed, separation or penetration, a region-specific damage model, object identity/fragility, duration, and task interruption consequence. The current binary labels and completed metrics remain unchanged.

## Frozen evidence reproduction

All three checkpoint files matched their required SHA-256 values. Their 576-row ledgers aligned exactly by state, candidate, split, family, labels, scores, and selections. Held-out aggregate metrics and every selected candidate reproduced at `atol=1e-12`, `rtol=1e-12`; no checkpoint inference was required.

| Condition | Ledger SHA-256 | AUC | Recall | FNR | Negative retention | Selected candidates |
|---|---|---:|---:|---:|---:|---|
| Depth | `d4df8e8a57f3ac94fbb91ec2f343e92fb82955b21912b30bc52bcadaae954510` | 0.7360 | 0.8537 | 0.1463 | 0.4242 | Exact reproduction |
| LiDAR | `cdd6706c078480f89df5bcd7e24872e339fcbfeb5aa3c8fa4c3b24c62d4253b9` | 0.7148 | 0.9431 | 0.0569 | 0.2424 | Exact reproduction |
| Depth + embodied | `bb0b4725650811a8fb7aa4a06c4ef989b35aeeebd86f32441ebc1fdc4edff900` | 0.8927 | 0.9106 | 0.0894 | 0.5636 | Exact reproduction |

## Post-hoc held-out threshold frontier

The frontier is diagnostic only. No threshold was adopted, and the calibration-selected threshold remains `0.0775665`.

- Thresholds evaluated: 290.
- Complete-gate points: 0.
- Maximum negative retention at recall ≥0.95: 0.3152.
- Maximum states retaining an action at recall ≥0.95: 17/24.
- Minimum regret at recall ≥0.95: 0, attained only at unusably restrictive availability points.

The best-retention point satisfying recall ≥0.95 was:

| Threshold | Recall | FNR | Retention | Admitted negative/positive | States retaining | Selected contact | False abstentions | Progress | Regret | Top-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.064802 | 0.9675 | 0.0325 | 0.3152 | 52 / 4 | 17/24 | 1 | 7 | 0.1685 m | 0.2002 | 0.4167 |

This misses retention, state availability, zero selection, abstention, regret, and top-three requirements. The maximum-progress point with zero selected contact retained only 2/165 contact-negative candidates in 2/24 states and falsely abstained in 22 states. Its high conditional progress is therefore not a useful operating point.

Frontier artifact:

- SHA-256: `cbbb678c6552907365d1b77eb5a5a9b422ec444f45c982e456768da390307968`
- Content digest: `d198a9234b5694b69025a5be65ea88a52717f0c3a817375a26d88e71a4b4d26b`

## Complete contact inventory

The machine-readable inventory contains all 123 held-out contact-positive branches, all eleven fusion false negatives/admitted positives, both selected contacts, eleven mechanically matched correctly rejected positives, and eleven matched retained negatives. Matching used family and structural onset/duration or candidate identity, never model performance.

- Inventory: `.generated/geometry_fusion_contact_error_attribution_v1/contact_error_inventory.json`
- SHA-256: `ba7bcbdbc93f33d22e83af3841bebc4cb0dc1c2ee18957ec97bd3b85dab88ff1`
- Contact positives: 123.
- Fusion false negatives: 11.
- Selected contact positives: 2.
- Persistent contacts: 68; transient contacts: 55.
- Contact followed by or overlapping stuck: 70.
- Meaningful absolute route progress: 111.

The eleven false negatives comprise five large-maze, four medium-maze, and two loop-alias branches. Ten are one-tick transient contacts and one persists for three ticks. Four overlap stuck.

| Branch | Family | Primitive | Onset / duration | Fusion / LiDAR score | Depth pre-onset proxy | LiDAR pre-onset proxy | Stuck | Progress | Attribution |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `scale-held-0-01:02` | Large | straight slow | 3 / 1 | 0.0504 / 0.5132 | No | Yes | Yes | −0.150 | `MULTIPLE` |
| `scale-held-0-02:00` | Large | straight fast | 3 / 1 | 0.0642 / 0.6479 | No | Yes | No | 0.434 | `MULTIPLE` |
| `scale-held-0-02:11` | Large | hold | 9 / 1 | 0.0648 / 0.2797 | No | Yes | No | 0.048 | `MULTIPLE` |
| `scale-held-0-03:06` | Large | sustained right turn | 3 / 1 | 0.0738 / 0.3099 | No | No | Yes | −0.127 | `UNRESOLVED` |
| `scale-held-0-04:04` | Large | sustained right arc | 15 / 1 | 0.0682 / 0.4059 | Yes | Yes | No | 0.093 | `GEOMETRY_VISIBLE_MODEL_MISS` |
| `scale-held-1-00:11` | Medium | hold | 3 / 1 | 0.0723 / 0.6316 | No | No | No | 0.005 | `UNRESOLVED` |
| `scale-held-1-01:06` | Medium | sustained right turn | 3 / 1 | 0.0651 / 0.7331 | No | Yes | Yes | 0.037 | `MULTIPLE` |
| `scale-held-1-01:07` | Medium | turn left then go | 2 / 1 | 0.0702 / 0.4356 | No | Yes | No | 0.175 | `MULTIPLE` |
| `scale-held-1-02:10` | Medium | reverse then turn | 13 / 1 | 0.0621 / 0.2245 | No | Yes | Yes | −0.067 | `MULTIPLE` |
| `scale-held-3-01:01` | Loop-alias | straight medium | 8 / 1 | 0.0652 / 0.7778 | Yes | Yes | No | −0.318 | `GEOMETRY_VISIBLE_MODEL_MISS` |
| `scale-held-3-01:03` | Loop-alias | sustained left arc | 4 / 3 | 0.0599 / 0.7373 | No | Yes | No | −0.246 | `MULTIPLE` |

The attribution counts are two `GEOMETRY_VISIBLE_MODEL_MISS`, seven `MULTIPLE`, and two `UNRESOLVED`. No row can be soundly labelled `FRONT_DEPTH_OUT_OF_FOV` or `GEOMETRY_OCCLUDED` because the contact point and body link are absent. No false negative was depth-event-only under the frozen 0.35 m planned-path proxy.

## Visibility and embodied evidence

For each contact branch, the inventory reports central front-depth and forward-LiDAR sector minimum ranges, the first 0.35 m threshold crossing, pre-onset/event status, and embodied change magnitudes. These are path-sector observability proxies—not contact-point visibility labels.

Among the eleven false negatives:

- 2/11 had front-depth proximity evidence before onset;
- 0/11 first became depth-visible only at onset;
- 9/11 had LiDAR forward-sector evidence before onset;
- 2/11 had neither proxy before or at onset.

Acceleration, angular velocity, joint acceleration, actuator torque, calf net contact force, and joint-velocity response changes are reported for every error. Calf force is normal locomotion telemetry, not a stored force for the disallowed body contact; it cannot establish impact severity or target equivalence.

## LiDAR–fusion complementarity

| Diagnostic | Result |
|---|---:|
| Fusion false negatives rejected by frozen LiDAR | 11/11 |
| LiDAR false negatives rejected by frozen fusion | 7/7 |
| Selected fusion contacts rejected by LiDAR | 2/2 |
| Pearson score correlation | 0.2662 |

Fusion positive scores range from 0.0504 to 0.9643 and negative scores from 0.0416 to 0.9069. LiDAR positive scores range from 0.1135 to 0.8131 and negatives from 0.1113 to 0.7865. Both have substantial class overlap, but their relatively low correlation and reciprocal error recovery establish `LIDAR_COMPLEMENTARY_TENDENCY`.

This does not qualify `WIDE_AREA_GEOMETRY_REQUIRED`: materially hazardous status and exact field-of-view/occlusion causality cannot be established from the stored evidence. It does justify retaining LiDAR-plus-embodied as a prospective option after the hazard ontology is repaired.

## Exact selected-contact diagnoses

### `scale-held-0-02:00`

- Scene: `large_enclosed_maze_f16f20f8156f`; candidate 0, straight fast.
- Contact: tick 3, one tick, commanded speed 0.30 m/s.
- Fusion score 0.06419, below 0.07757; LiDAR score 0.64794, above 0.18362.
- Front path-sector minimum through onset: 0.603 m; no 0.35 m front proxy crossing.
- LiDAR path-sector minimum through onset: 0.254 m, first informative tick 1.
- No stuck; realized route progress 0.434 m.
- Actual versus nominal endpoint displacement error 0.059 m; yaw error 0.097 rad.
- Base displacement speed at the contact tick: 0.237 m/s. Relative contact speed is unavailable.
- Attribution: `MULTIPLE`; severity: `SEVERITY_UNRESOLVED`.

### `scale-held-0-03:06`

- Scene: `large_enclosed_maze_9eeeebf63964`; candidate 6, sustained right turn.
- Contact: tick 3, one tick, commanded yaw rate −0.45 rad/s.
- Fusion score 0.07383; LiDAR score 0.30990. Both frozen filters reject under their respective thresholds only for LiDAR; fusion admits it.
- Front path-sector minimum through onset: 0.675 m; LiDAR forward-sector minimum: 0.436 m; no 0.35 m proxy crossing.
- Overlaps stuck; realized route progress −0.127 m.
- Actual versus nominal endpoint displacement error 0.148 m; yaw error 0.076 rad.
- Base displacement speed at contact tick: 0.049 m/s. Relative contact speed is unavailable.
- Attribution and severity: `UNRESOLVED` / `SEVERITY_UNRESOLVED`.

Both contacts would have been rejected by the frozen LiDAR-only decision, but neither can be classified as materially hazardous from the retained evidence.

## Contact severity and requirement ontology

Prospective definitions—not revised labels—are:

1. `MATERIAL_HAZARDOUS_CONTACT`: calibrated energy/impulse, region-specific damage, separation violation, or fragile-object consequence exceeds a frozen hazard limit.
2. `RECOVERABLE_LOW_ENERGY_CONTACT`: prospectively demonstrated below all such limits, without damage, loss of control, unsafe separation, or material task interruption.
3. `SEVERITY_UNRESOLVED`: the evidence establishes neither definition.

All 123 current positives are `SEVERITY_UNRESOLVED`; none is retrospectively reclassified. Duration, action, base motion, stuck overlap, endpoint error, and route progress are descriptive but cannot substitute for a calibrated hazard model.

Operational safety and task performance remain separate requirements. Avoiding contact does not establish task completion, while becoming stuck or failing to progress is safety-related but is not interchangeable with collision severity.

## Runtime and custody

The reducer ran in 11.37 seconds and produced approximately 624 KiB of new diagnostic evidence, dominated by the complete inventory and frontier. No training, checkpoint inference, simulation, rendering, encoding, geometry regeneration, label change, threshold adoption, JEPA access, navigation, memory, novelty, or beacon work occurred. Nothing remained running when the diagnostic was finalized.
