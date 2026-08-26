# Protected Contact Scope Requirements Review V1

Date: 2026-08-26

Status: Stage-A requirements-only review

Primary classification: `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`

## Purpose and evidence boundary

This review asks which robot/environment contacts must remain protected for the
bounded Unitree Go2 search-and-inspection research use case. It is intentionally
independent of sensor performance. Sensor coverage, per-link error, condition
metrics, held-out outcomes, and row-level evaluation evidence are not inputs to
the scope decision. A contact is not removed from the protected scope because
it is difficult to observe, and good observability would not by itself establish
that the contact is harmless.

This is not a hazard-analysis approval, an operational safety case, or authority
to change labels. It preserves the existing simulated disallowed-contact proxy
until consequence, recovery, task, and stopping requirements are supplied by an
approved system-hazard process.

## Preserved predecessor authority

The following completed findings are immutable aggregate context, not inputs
to the Stage-A scope-selection rationale:

> Exact Genesis per-link geometry reconstructs immediate contact, successor
> contact, zero-versus-nonzero safe-action availability, and the complete
> two-ply viability decision.

> Under the current full-body, 13-link protected-contact scope, no tested one-,
> two-, three-, or diagnostic four-origin range arrangement reproduced that
> decision sufficiently.

The preserved classifications are:

- `MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO`;
- `SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO`;
- `SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO`;
- `TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL`;
- `DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED`;
- `STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL`;
- `MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL`;
- `REPLANNING_INTERFACE_UNRESOLVED`;
- `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`;
- `ASSUMED_SENSOR_CONTRACT`.

They are preserved from result commit
`99cfa17cddb2aaddde69b8bdb6c3ea4a8e5ca849`. The only Stage-A inference drawn
from them is the already-known aggregate context that full-body range sensing
failed; no condition, link, error, threshold, calibration, held-out, or row
outcome enters the requirements decision.

## Source authority

The requirements evidence is:

- `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:7-21,23-57,59-61`:
  claim boundary, prospective hazard classes, conservative low-energy screens,
  current environment limits, and branch annotations.
- `docs/lewm_contact_hazard_instrumentation_contract_v1.md:13-36,38-40`:
  required physics-step evidence, unresolved consequence fields, event
  reduction, object-consequence schema, and deterministic fixtures.
- `docs/lewm_material_contact_safety_model_next_experiment_spec_v1.md:6-27`:
  the blocked instrumentation and consequence requirements that must precede a
  material-contact model or new panel.
- `docs/lewm_rollout_safety_and_trajectory_cleanup_2026-06-13.md:148-226`:
  deployment-valid obstacle inputs, swept-body clearance, execution-calibrated
  envelopes, physical labels, and joint safety/task promotion requirements.
- `docs/lewm_control_commitment_horizon_and_viability_v1_result.md:13-20,47-64,147-164`:
  the distinction between short commitment and a fail-safe, the unresolved
  100 ms replanning interface, and the required response and platform-parity
  evidence. No outcome metric from that document is used here.
- `docs/lewm_deployment_valid_strong_braking_mode_v1_result.md:86-120`:
  prospective stopping-envelope and platform-mode requirements. No outcome
  metric from that document is used here.
- `docs/lewm_go2_generalization_execution_contract_2026-07-09.md:27-46,68-91`:
  search-task completion, safety as a non-compensable constraint, deployment
  inputs, and the privileged-evidence boundary.
- `docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md:3-27`:
  bounded local intent, replanning, waypoint completion, and separate
  collision/fall/stuck, abstention, progress, and efficiency outcomes.
- `docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md:3-5,28-52`:
  separate geometry, progress, safety, completion, and support outputs; hard
  admissibility; and brake/abstain behavior when no candidate is admissible.
- `docs/lewm_planner_design_decision_memo_2026-08-19.md:5-20,23-29`:
  the local-waypoint boundary and prohibition on inferring beacon discovery,
  global routing, physical transfer, or safe navigation.
- `docs/lewm_planner_evaluation_first_protocol_2026-08-19.md:3-27`:
  evaluation ordering and stop rules. Its numeric gates are development gates,
  not stakeholder-approved safety requirements.
- `docs/SAINTS_Year_1_Progression_Document-3.pdf`, p.5 lines 12-48, p.7 lines
  11-27, p.8 lines 6-48, pp.15-17, p.23 lines 18-49, and pp.24-26:
  the bounded research use case, search/inspection non-performance hazards,
  AREA safeguards, stakeholder engagement, and SACE/AMLAS evidence mapping.

## Intended use case and claim boundary

The current case study is RGB-based high-level navigation by a Go2 in bounded
indoor environments. Local action consequences and local waypoint following are
separate from persistent place memory, global graph routing, beacon discovery,
and physical locomotion control. Search or inspection is an intended possible
application, not a deployment-approved mission definition.

The system-level safety problem includes both:

1. operational harm: contact, collision, fall, instability, entrapment, control
   loss, property or robot damage, and operation outside the calibrated domain;
2. safety-related non-performance: stationary behavior, repeated veto or
   abstention, oscillation, inadequate coverage, incomplete inspection, or
   failure to complete a task that could leave a hazard undetected.

Neither side may substitute for the other. Progress cannot compensate for an
unacceptable physical hazard, and zero movement cannot satisfy a meaningful
search or inspection mission.

## Context-first scope unit and exact requirement categories

The scope unit is a contact or non-contact operational context first, followed
by its consequence evidence, contact phase, environment object, and only then
the robot region, protected link, and collision primitive. A region name alone
is not a scope unit. The exact prospective requirement categories are:

1. `HARD_PREACTION_SEPARATION`: the planner must preserve separation before
   action execution. This applies across every robot region when the context is
   a person, fragile asset, safety-critical asset, prohibited contact, or has
   demonstrated damage, fall, stability, entrapment, or control-loss
   consequence.
2. `CONDITIONAL_RECOVERABLE_CONTACT`: contact may be admitted only under
   prospectively approved physical-severity and recovery bounds, with a
   demonstrated bounded response and no unacceptable task consequence.
3. `MONITOR_AND_RECOVER`: a system response is required for no-contact stuck,
   oscillation, repeated veto or abstention, inadequate progress, incomplete
   inspection, or related task failure. If contact also occurs, its contact
   context receives a separate category.
4. `PERMITTED_SUPPORT_OR_SELF_CONTACT`: only an established locomotion-support
   or robot-self-contact exclusion is permitted; this category is not a general
   low-severity-contact class.
5. `SEVERITY_OR_REQUIREMENT_UNRESOLVED`: the physical severity, permission, or
   application requirement is not established. This is the default when the
   evidence required for another category is absent.

No current scientific row is assigned `CONDITIONAL_RECOVERABLE_CONTACT`.
Physical consequence bounds and recovery requirements are unavailable. Fixed
wall, landmark, and abnormal-ground contexts remain
`SEVERITY_OR_REQUIREMENT_UNRESOLVED`, while the conservative simulated
disallowed-contact proxy continues to require separation from them. They must
not be retrospectively relabelled as recoverable.

## Historical event ontology preserved

The prospective ontology remains exactly:

1. `MATERIAL_HAZARDOUS_CONTACT`: contact with a credible direct injury,
   robot/property damage, stability/fall, severe entrapment, control-loss, or
   unacceptable-separation consequence.
2. `RECOVERABLE_LOW_SEVERITY_CONTACT`: complete evidence establishes low
   energy, short duration, no damage relevance, no destabilisation, no material
   progress loss, and recoverability.
3. `SEVERITY_UNRESOLVED`: available evidence establishes neither of the first
   two classes.
4. Recoverability/task-performance failure: stuck, repeated ineffective
   contact, route-progress loss, excessive recovery, or mission incompletion;
   this remains separate from contact severity.
5. `NO_DISALLOWED_CONTACT`: no contact included by the unchanged simulated
   disallowed-contact definition.

The separate annotations remain:

- `STABILITY_HAZARD`;
- `DAMAGE_RELEVANT`;
- `HUMAN_OR_FRAGILE_OBJECT_RELEVANT`;
- `STUCK_OR_ENTRAPMENT_CONSEQUENCE`;
- `TASK_PROGRESS_CONSEQUENCE`.

Ordinary calf/foot-ground support and robot self-contact remain the only
established `PERMITTED_SUPPORT_OR_SELF_CONTACT` exclusions. Abnormal non-calf
ground contact and calf contact with non-ground geometry remain disallowed and
requirements-unresolved. No other regional exclusion is supported.

The 13 protected link groups aggregate 27 protected collision primitives
(`docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_preregistration_2026-08-26.md:26-32`). The historical calf-plane link exemption is therefore broader than a
literal foot-support event: without collision-primitive identity and contact
phase, a calf-link/ground-plane match does not prove permitted stance support.
Prospective support semantics remain unresolved until they are defined at
geometry and contact-phase granularity. This review preserves the historical
label for custody but does not endorse the broad link-level exemption as a
deployment requirement.

The existing low-energy screens of duration at most 0.040 s, relative normal
speed at most 0.20 m/s, integrated normal impulse at most 0.50 N s,
penetration at most 0.002 m, and repeated-contact count at most one are
development screens only. They are not manufacturer, human-injury,
infrastructure-damage, or deployment limits and cannot authorize a narrower
scope.

## Hazard-derived candidate requirements

| Requirement | Requirements-only statement | Status |
|---|---|---|
| `PCS-SR-CONTACT-001` | Reject contact that exceeds approved link/object consequence limits or has a categorical person, fragile, safety-critical, prohibited-contact, damage, fall, or stability consequence. | `EVIDENCE_MISSING` |
| `PCS-SR-CLEARANCE-001` | Preserve clearance sufficient for the candidate motion plus a validated stopping distance and uncertainty margin. | `EVIDENCE_MISSING` |
| `PCS-SR-STABILITY-001` | Prevent or terminate actions with unacceptable fall, destabilisation, entrapment, or loss-of-control consequence. | `EVIDENCE_MISSING` |
| `PCS-SR-VIABILITY-001` | Do not enter a state lacking a bounded contact-negative response or a qualified stopping fallback within the implemented commitment cycle. | `PARTIALLY_SPECIFIED` |
| `PCS-SR-FALLBACK-001` | On invalid operating assumptions or no admissible movement, enter a prospectively defined, executable minimum-risk state with bounded acknowledgement and response time. | `EVIDENCE_MISSING` |
| `PCS-SR-RECOVERY-001` | Detect stuck, entrapment, repeated contact, or ineffective motion and recover within an approved time/contact/progress budget. | `EVIDENCE_MISSING` |
| `PCS-SR-MONITOR-001` | Detect unreliable perception, prediction, localisation, or operation outside the calibrated domain early enough for an effective response, with bounded false alarms and task cost. | `EVIDENCE_MISSING` |
| `PCS-SR-TASK-001` | Meet application-derived minimum progress, coverage, inspection completeness, and completion requirements; reject-all, repeated stopping, and indefinite abstention do not satisfy this requirement. | `EVIDENCE_MISSING` |
| `PCS-SR-TRACE-001` | Trace each system requirement to its hazard, operating context, allocated learned/non-learned component, verification evidence, and effective response. | `PARTIALLY_SPECIFIED` |

`PARTIALLY_SPECIFIED` means that the repository states the form of the
requirement but lacks approved quantitative acceptance evidence. It does not
mean the requirement has been satisfied.

## Per-region requirements rationale

The authoritative historical inventory is the Genesis 13-link/27-component
contract. Fixed head components collapse into `base`; fixed lower-leg and foot
components collapse into each `_calf` link. The ROS/Gazebo xacro has a
different lower-leg/head collision description, so physical collision-envelope
parity is unresolved.

| Protected link | Frozen collision components | Mechanical/function role | Stability, payload, or entrapment relevance | Ordinary support already excluded? | Deployment-valid recovery | Current evidence limitation |
|---|---|---|---|---|---|---|
| `base` | g0 trunk box 0.3762 x 0.0935 x 0.114 m; g1 head capsule r=0.05 m, l=0.09 m; g2 head sphere r=0.047 m | Central chassis, four leg roots, fixed head/sensor-bearing envelope | Body-pose/load transfer; payload/electronics exposure; broad rigid envelope may wedge | No | None established | No structural/payload limit, damage model, or physical-envelope parity |
| `FL_hip` | g3 capsule r=0.046 m, l=0.04 m | Front-left proximal ab/adduction and load transfer | Joint obstruction can affect stance/control; proximal snagging is plausible | No | None established | No joint-contact load, saturation, entrapment, or recovery limit |
| `FR_hip` | g4, same primitive | Front-right proximal ab/adduction and load transfer | Same right-side function and unresolved consequences | No | None established | Same missing evidence |
| `RL_hip` | g5, same primitive | Rear-left proximal ab/adduction and load transfer | Rear stance/control and proximal entrapment relevance | No | None established | Same missing evidence; no approved front/rear severity distinction |
| `RR_hip` | g6, same primitive | Rear-right proximal ab/adduction and load transfer | Same right-rear function and unresolved consequences | No | None established | Same missing evidence |
| `FL_thigh` | g7 box 0.11 x 0.0245 x 0.034 m | Front-left upper-leg pitch, swing/stance, load transfer | Contact may block swing/stance or alter support | No | None established | No gait-phase, contact-load, damage, or recovery criterion |
| `FR_thigh` | g8, same primitive | Front-right upper-leg pitch, swing/stance, load transfer | Same right-side function and unresolved consequences | No | None established | Same missing evidence |
| `RL_thigh` | g9, same primitive | Rear-left upper-leg pitch, swing/stance, load transfer | Contact may block swing/stance or alter support | No | None established | Same missing evidence; no approved front/rear severity distinction |
| `RR_thigh` | g10, same primitive | Rear-right upper-leg pitch, swing/stance, load transfer | Same right-rear function and unresolved consequences | No | None established | Same missing evidence |
| `FL_calf` | g11 main capsule r=0.012 m/l=0.12 m; g12 lower capsule r=0.011/l=0.065; g13 distal capsule r=0.0155/l=0.03; g14 foot sphere r=0.022 | Front-left lower leg and terminal support | Support, stability, entanglement, repeated contact, recovery | Historical link/Plane exclusion covers all g11-g14, not merely foot g14 | None established beyond restricted reverse/yaw action families | No primitive/phase support semantics, entanglement limit, or recovery proof |
| `FR_calf` | g15 main capsule r=0.013/l=0.12; g16/g17 lower capsules; g18 foot sphere | Front-right lower leg and terminal support | Same right-side function and unresolved consequences | Historical link/Plane exclusion covers g15-g18 | None established | Same missing evidence |
| `RL_calf` | g19 main capsule r=0.013/l=0.12; g20/g21 lower capsules; g22 foot sphere | Rear-left lower leg and terminal support | Support, stability, entanglement, repeated contact, recovery | Historical link/Plane exclusion covers g19-g22 | None established | Same missing evidence; no approved front/rear severity distinction |
| `RR_calf` | g23 main capsule r=0.013/l=0.12; g24/g25 lower capsules; g26 foot sphere | Rear-right lower leg and terminal support | Same right-rear function and unresolved consequences | Historical link/Plane exclusion covers g23-g26 | None established | Same missing evidence |

The inventory records mechanical facts separately from engineering hazard
hypotheses. It does not infer that a region is safe, hazardous, or recoverable
from its link name.

| Region or contact class | Requirements rationale | Stage-A disposition |
|---|---|---|
| Trunk, head-hosting body, and underside | Potential structural, electronics, payload, stability, control-loss, and large-area contact consequences; no approved region-specific limit exists. | Preserve as protected simulated contact. |
| Hips | Joint/actuator, entrapment, stance, and control-authority consequences are plausible; no approved damage or recoverability limit exists. | Preserve. |
| Thighs | Large swept limb regions can transmit load, destabilise the platform, become trapped, or damage nearby assets; no consequence evidence supports exclusion. | Preserve. |
| Calves against non-ground objects | Can cause obstacle contact, entanglement, repeated ineffective motion, or destabilisation. The ontology explicitly keeps these contacts disallowed. | Preserve. |
| Ordinary calf/foot-ground support | Required for locomotion and explicitly excluded by the established ontology. | Continue existing exclusion only. |
| Front, side, and rear contacts | Direction alone does not establish severity or recoverability. No directional consequence evidence supports a reduced scope. | Preserve all directions. |
| Person, fragile, safety-critical, or prohibited-contact object | The ontology treats these object/consequence fields as categorically material when present. The current maze assets do not provide them. | Hard-hazard category prospectively; current application binding unresolved. |
| Fixed wall or landmark | Visual class and rigidity do not establish harmlessness, platform damage tolerance, or transfer to real infrastructure. | Preserve; do not transfer a wall threshold. |

These rows are secondary attribution after context classification. In
particular, person, fragile, critical, prohibited, demonstrated-damage,
fall/stability, or entrapment contexts are
`HARD_PREACTION_SEPARATION` across all regions. Wall, landmark, and abnormal
ground contexts remain `SEVERITY_OR_REQUIREMENT_UNRESOLVED`. No-contact stuck,
abstention, oscillation, and task failures are `MONITOR_AND_RECOVER`.

## Perfect-sensing counterfactual

Assume, counterfactually, that every current and future clearance/contact event
for every body region were observed perfectly with zero latency and exact
attribution. The protected-contact requirements would still be unresolved:

- perfect sensing would not supply an object material, fragility, person, or
  safety-critical classification;
- it would not define a permissible force, impulse, velocity, penetration,
  duration, repetition, fall, damage, or entrapment consequence;
- it would not establish the consequence of missed inspection or inadequate
  progress;
- it would not implement or validate a physical stopping or recovery response;
- it would not define the operating domain or stakeholder risk tolerance.

Therefore observability is downstream of the protected-scope decision. A
perfect sensor cannot justify narrowing the scope, and an imperfect sensor
cannot justify excluding the contacts it misses.

## Missing evidence and freeze blockers

Stage B is blocked on all of the following:

1. a defined application and operational design domain, including people,
   assets, environment, mission, speeds, payload, supervision, and operating
   assumptions;
2. stakeholder-reviewed consequences of incomplete inspection and quantitative
   progress, coverage, revisit, deadline, and completion requirements;
3. object material, mass, mobility, fragility, safety-criticality, person-proxy,
   permitted-contact, and observed-damage fields;
4. approved robot-link/object consequence limits for force or a validated
   substitute, impulse, relative velocity, duration, repetition,
   penetration/separation, stability, fall, entrapment, and damage;
5. validation of simulator evidence against intended physical instrumentation,
   or an explicit permanent restriction to a simulated separation proxy;
6. platform-equivalent stopping behavior, request/acknowledgement latency,
   stopping distance/time, controller variability, and uncertainty margin;
7. definitions and evidence for stuck, successful separation, restored task
   capability, recovery budget, repeated-contact limit, and operator handoff;
8. allocation of system requirements to learned and non-learned components,
   stakeholder-approved acceptance criteria, and appropriately independent
   verification.

Until these blockers are resolved, no contact class, protected link, body
region, direction, or swept volume may be removed; no contact label may be
changed; and no sensor, predictor, training panel, or navigation evaluation may
be selected using a narrowed scope.

## Classification

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

The final two classifications are downstream engineering dispositions only.
They neither define nor narrow the protected-contact requirement and do not
authorize a sensor architecture. The requirements-review gate is
`STAGE_B_NOT_AUTHORIZED`; that gate is not a secondary classification.
