# Protected Contact Scope Decision Memo V1

Date: 2026-08-26

Decision: `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`

## Decision

Preserve the full existing simulated disallowed robot/environment contact scope.
Do not remove a protected link, body region, contact direction, or swept volume.
Continue only the established exclusions for ordinary calf/foot-ground support
and robot self-contact. Abnormal non-calf ground contact and calf contact with
non-ground geometry remain disallowed.

This is a conservative custody decision, not a finding that every simulated
contact is materially hazardous. The available evidence cannot establish which
contacts are materially hazardous, recoverable low severity, or acceptable for
an intended deployment. Those events remain `SEVERITY_UNRESOLVED` unless an
independently sufficient categorical consequence is present.

## Context-first scope categories

The primary decision unit is the operational/contact context and its
consequence, object, and contact phase. Robot region, link, and collision
primitive are subsequent attribution fields; a region alone never establishes
permission. The exact requirements vocabulary is:

- `HARD_PREACTION_SEPARATION`: person, fragile, safety-critical, prohibited,
  demonstrated-damage, fall, stability, entrapment, or control-loss
  context, across every region;
- `CONDITIONAL_RECOVERABLE_CONTACT`: allowed only after approved physical and
  recovery bounds demonstrate bounded recoverability and acceptable task
  consequence;
- `MONITOR_AND_RECOVER`: no-contact stuck, oscillation, repeated veto or
  abstention, inadequate progress, or incomplete-task context;
- `PERMITTED_SUPPORT_OR_SELF_CONTACT`: only established locomotion-support and
  robot-self-contact exclusions;
- `SEVERITY_OR_REQUIREMENT_UNRESOLVED`: default when severity, permission, or
  task requirements are missing.

No current row is assigned `CONDITIONAL_RECOVERABLE_CONTACT`. Wall, landmark,
and abnormal-ground contact contexts remain
`SEVERITY_OR_REQUIREMENT_UNRESOLVED` under the conservative separation proxy.
No-contact stuck, abstention, oscillation, and task failures are
`MONITOR_AND_RECOVER`. Person, fragile, critical, prohibited, or demonstrated
damage/fall/stability/entrapment contexts are
`HARD_PREACTION_SEPARATION` regardless of robot region.

The 13 protected link groups aggregate 27 protected collision primitives. The
historical calf-plane link exemption is consequently broader than literal foot
support. It remains preserved for label custody, but prospective support
permission is unresolved without primitive identity and contact-phase detail.

## Why this decision is requirements-derived

The repository defines plausible consequences—injury, robot/property damage,
fall, instability, entrapment, loss of control, unacceptable separation,
stuck behavior, progress loss, and incomplete inspection—but lacks the approved
application, object properties, region-specific consequence limits, physical
calibration, stopping response, recovery budget, and stakeholder acceptance
needed to authorize a narrower scope.

The relevant authorities are:

- `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:15-57`;
- `docs/lewm_contact_hazard_instrumentation_contract_v1.md:13-36`;
- `docs/lewm_material_contact_safety_model_next_experiment_spec_v1.md:6-27`;
- `docs/SAINTS_Year_1_Progression_Document-3.pdf`, p.8 and pp.15-17,
  23-26;
- `docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md:3-27`;
- `docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md:3-5,28-52`;
- `docs/lewm_control_commitment_horizon_and_viability_v1_result.md:13-20,47-64,147-164`;
- `docs/lewm_deployment_valid_strong_braking_mode_v1_result.md:86-120`.

No sensor coverage, sensor error, held-out condition, or row-level outcome is a
decision input.

## Alternatives considered

### Narrow to directly observable regions

Rejected. Observability is a property of a sensor/representation, not a
consequence-based definition of acceptable contact. This would make the safety
scope depend on implementation weakness.

### Protect trunk only

Rejected. Hip, thigh, calf, underside, and limb contacts can plausibly produce
joint damage, instability, entrapment, repeated ineffective motion, asset
damage, or loss of control. No approved evidence shows those consequences are
acceptable.

### Exclude rear or side contacts

Rejected. Contact direction does not establish force, damage, stability,
fragility, or recoverability. No directional consequence limits exist.

### Treat all rigid-wall contacts as recoverable

Rejected. Current assets lack mechanical material, mass, fragility,
safety-criticality, permitted-contact, and damage fields. A simulator wall
threshold cannot transfer to real inspection infrastructure or robot damage.

### Treat every disallowed contact as materially hazardous

Rejected as a material-safety claim. Preserving a conservative binary proxy is
appropriate, but it must not be relabelled as injury, property-damage, human, or
deployment safety without consequence evidence.

### Preserve the complete proxy while requirements remain unresolved

Selected. It prevents retrospective label relaxation and retains all evidence
needed for a later approved hazard analysis.

## Per-region decision

- Trunk/head-hosting body and underside: preserve.
- Hips: preserve.
- Thighs: preserve.
- Calves against non-ground geometry: preserve.
- Front, side, and rear contacts: preserve.
- Ordinary calf/foot-ground support: retain existing exclusion only.
- Robot self-contact: retain existing exclusion only.
- Person, fragile, safety-critical, prohibited-contact, damage, fall, and
  instability consequences: retain as prospective categorical hard hazards;
  their application binding remains unresolved.

These region bullets do not replace the context-first categorization above.

## Perfect-sensing decision test

Even a perfect, zero-latency, full-body sensor would not provide approved impact
limits, object consequence, inspection consequence, stopping behavior,
recoverability, or stakeholder risk tolerance. The same decision therefore
holds under perfect sensing. Conversely, sensing difficulty supplies no basis
for excluding a region.

## Search/inspection decision boundary

Physical safety and task capability remain separate and jointly necessary.
Unsafe progress is inadmissible; indefinite stop, repeated veto, or reject-all
behavior is not a successful search/inspection policy. The repository's local
waypoint metrics remain development measures rather than application-derived
safety acceptance criteria.

## Required next authority

The next authorized work is requirements acquisition and assurance preparation:

1. define the application and ODD;
2. obtain stakeholder input, including consequences of incomplete inspection;
3. define object and link consequence limits;
4. validate simulator/physical instrumentation;
5. qualify a stopping and minimum-risk response;
6. define recovery and operator-escalation requirements;
7. allocate requirements across learned and non-learned components;
8. approve verification and acceptance criteria.

Stage B is not authorized. No model training, new panel, sensor qualification,
predictor opening, label change, region search, memory, navigation, routing, or
beacon work follows from this memo.

## Classifications

Primary:

- `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`

Secondaries:

- `SIMULATED_CONTACT_PROXY_SCOPE_ONLY`;
- `DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED`;
- `PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED`;
- `RECOVERABILITY_REQUIREMENTS_UNRESOLVED`;
- `MISSION_PROGRESS_REQUIREMENTS_PRESENT`;
- `DISTRIBUTED_BODY_SENSING_CANDIDATE`;
- `FULL_BODY_EXTERNAL_RANGE_SENSING_IMPLAUSIBLE`.

`DISTRIBUTED_BODY_SENSING_CANDIDATE` and
`FULL_BODY_EXTERNAL_RANGE_SENSING_IMPLAUSIBLE` are downstream engineering
dispositions, not rationales for narrowing the scope and not authorization to
select hardware. The execution gate remains `STAGE_B_NOT_AUTHORIZED`.
