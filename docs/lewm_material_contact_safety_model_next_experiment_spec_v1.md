# MATERIAL_CONTACT_SAFETY_MODEL_V1 — blocked prospective specification

Status: **not authorised to run**
Blocking result: `CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT`

## Required instrumentation action

Before another safety model or fresh evaluation panel is created:

1. Extend every environment asset with fixed, reviewed fields for mechanical material, mass, mobility, fragility, safety criticality, person proxy, permitted/prohibited contact, and damage observation.
2. Establish robot-link and object-specific consequence limits from an approved hazard analysis. At minimum, bind calibrated force/impulse or a validated substitute, relative impact velocity, duration/repetition, penetration/separation, and stability/fall consequences.
3. Calibrate or validate Genesis solver force/impulse against the intended Go2/platform contact instrumentation, or explicitly narrow all claims to simulation separation hazards.
4. Define damage/stability acceptance for trunk/head, hip, thigh, calf and underside contacts and separate person/fragile-infrastructure requirements. Do not transfer a rigid-wall threshold to those objects.
5. Add deterministic consequence fixtures containing at least one known material hazard and one demonstrated recoverable contact for each relevant object/body-region class. Freeze the ontology and thresholds before collection.
6. Only then create fresh scene-disjoint fit/calibration/held-out identities. Persist raw contact points, event reductions, branch labels, scores, thresholds, and selected actions.

No percentile of the present event distribution may serve as the sole safety rationale.

## Model experiment after the blocker is resolved

The primary hard target is prospective `MATERIAL_HAZARDOUS_CONTACT`. Secondary outputs are recoverable low-severity contact, impulse/severity, body region, and contact followed by stuck. Stuck remains a separate recoverability/task-performance variable.

Prospectively compare true-future depth, LiDAR, enhanced embodied state, depth plus embodied state, and—only if justified by the visibility audit—LiDAR plus embodied state. Hard rejection uses only calibrated material-hazard risk. Recoverable-contact risk is a soft tie-break or recovery-demand signal. Deterministic kinematic route progress remains unchanged.

Evaluation must jointly require high material-hazard recall, useful safe-action retention, zero selected material hazards, bounded false abstention, retained route progress, and per-family generalisation. A reject-all policy is not success. A fresh calibration and held-out panel collected after the ontology freeze is mandatory.

Until the instrumentation action above is complete, `MATERIAL_CONTACT_SAFETY_MODEL_V1` remains blocked and no model threshold may be selected.
