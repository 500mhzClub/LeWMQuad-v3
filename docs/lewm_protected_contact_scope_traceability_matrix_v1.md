# Protected Contact Scope Traceability Matrix V1

Date: 2026-08-26

Status: Stage-A candidate traceability; not an approved hazard log

Primary classification: `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`

Authorized secondary classifications:

- `SIMULATED_CONTACT_PROXY_SCOPE_ONLY`;
- `DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED`;
- `PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED`;
- `RECOVERABILITY_REQUIREMENTS_UNRESOLVED`;
- `MISSION_PROGRESS_REQUIREMENTS_PRESENT`;
- `DISTRIBUTED_BODY_SENSING_CANDIDATE`;
- `FULL_BODY_EXTERNAL_RANGE_SENSING_IMPLAUSIBLE`.

The final two are downstream engineering dispositions and do not define the
scope or authorize hardware. The gate remains `STAGE_B_NOT_AUTHORIZED`.

## Status vocabulary

- `SUPPORTED_REQUIREMENT_FORM`: the repository supports the stated form of the
  requirement, but not necessarily its acceptance threshold.
- `PARTIALLY_SPECIFIED`: interfaces or qualitative behavior are stated, but
  consequence limits or verification evidence are incomplete.
- `EVIDENCE_MISSING`: the requirement cannot yet be quantified or accepted.
- `PRESERVED_PROXY`: the existing simulated disallowed-contact scope remains
  unchanged pending evidence.
- `OUT_OF_CURRENT_APPLICATION_SCOPE`: explicitly excluded from the present
  research case study; this is not evidence of safety in that domain.
- `STAGE_B_BLOCKED`: no empirical scope comparison or label change is authorized.

## Context-first scope unit and exact categories

The traceability key is
`(operational/contact context, consequence evidence, contact phase, environment object, region, link, primitive)`.
Region or link alone never determines permission. The exact requirement-category
vocabulary is:

| Category | Prospective assignment rule | Current-row status |
|---|---|---|
| `HARD_PREACTION_SEPARATION` | Person, fragile asset, safety-critical asset, prohibited contact, or demonstrated damage, fall, stability, entrapment, or control-loss consequence, across every body region. | No such object metadata exists in the current maze corpus; rule remains prospective and hard. |
| `CONDITIONAL_RECOVERABLE_CONTACT` | Complete approved physical-severity bounds and bounded recovery evidence establish that contact is recoverable with no unacceptable task consequence. | No current row is assigned this category; required physical and recovery bounds are unavailable. |
| `MONITOR_AND_RECOVER` | No-contact stuck, oscillation, repeated veto/abstention, inadequate progress, or incomplete task requires monitoring and bounded recovery. Contact, if present, is classified separately. | Requirement form present; quantitative recovery/task bounds unavailable. |
| `PERMITTED_SUPPORT_OR_SELF_CONTACT` | Only established locomotion-support or robot-self-contact exclusions. | Limited to the existing support/self exclusions; not authority for other low-severity contact. |
| `SEVERITY_OR_REQUIREMENT_UNRESOLVED` | Severity, permission, or task requirement lacks sufficient approved evidence. | Wall, landmark, and abnormal-ground contact contexts remain here, under the unchanged conservative separation proxy. |

No current row is assigned `CONDITIONAL_RECOVERABLE_CONTACT` because the
required physical-severity and recovery bounds are unavailable.

## Context and assumptions

| ID | Context or assumption | Source | Status |
|---|---|---|---|
| `PCS-CTX-001` | Research case study is RGB-based high-level Go2 navigation in bounded indoor environments; search/inspection is a possible application, not an approved deployment mission. | `docs/SAINTS_Year_1_Progression_Document-3.pdf`, p.5 lines 12-48 and p.15 lines 18-24 | `SUPPORTED_REQUIREMENT_FORM` |
| `PCS-CTX-002` | Safety includes operational harm and safety-related task non-performance. | Same PDF, p.8 lines 6-13 and p.24 lines 34-41 | `SUPPORTED_REQUIREMENT_FORM` |
| `PCS-CTX-003` | Current maze assets are fixed ground, walls, and landmarks and lack person, fragility, material, mass, permitted-contact, and damage fields. | `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:53-57` | `PRESERVED_PROXY` |
| `PCS-CTX-004` | Protected geometry is represented as 13 link groups aggregating 27 collision primitives; link identity alone is not the prospective scope unit. | `docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_preregistration_2026-08-26.md:26-32` | `SUPPORTED_REQUIREMENT_FORM` |
| `PCS-A-001` | The current binary target is a simulated contact/separation proxy, not material-impact, human, property, or deployment safety. | `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:7-21,47-49` | `SUPPORTED_REQUIREMENT_FORM` |
| `PCS-A-002` | Sensor performance is not authority for selecting the protected scope. | This Stage-A review; supported by the hazard-derived threshold requirement in the progression PDF, p.24 lines 34-41 and p.25 lines 4-8 | `SUPPORTED_REQUIREMENT_FORM` |

## Hazard-to-scope traceability

The following is the normative context-first matrix requested by the review.
`NOT REPRESENTED IN CURRENT TEST ENVIRONMENT` records an evidence boundary,
not absence of a deployment hazard.

| Hazard | Safety requirement | Robot region | Object/context | Required treatment | Evidence | Assumption | Residual limitation |
|---|---|---|---|---|---|---|---|
| Person collision | `PCS-SR-CONTACT-001`: preserve separation before execution. | All 27 components/all 13 aggregate links. | Person or person proxy; `NOT REPRESENTED IN CURRENT TEST ENVIRONMENT`. | `HARD_PREACTION_SEPARATION` | Progression hazardous-scenario discussion; contact ontology person field. | Person consequence is categorically unacceptable unless an approved system hazard process states otherwise. | No person model, person detector, injury criterion, or stakeholder-approved separation distance. |
| Fragile or safety-critical infrastructure contact | `PCS-SR-CONTACT-001`: preserve separation before execution. | All regions, including head/payload envelope. | Fragile, critical, prohibited-contact, or damage-sensitive asset; `NOT REPRESENTED IN CURRENT TEST ENVIRONMENT`. | `HARD_PREACTION_SEPARATION` | Hazard analysis and instrumentation object-consequence schema. | Object class and prohibition are known before action. | No such object metadata, damage model, material property, or acceptance rule exists in the maze. |
| Trunk/payload impact | `PCS-SR-CONTACT-001`, `PCS-SR-STABILITY-001`. | Base g0 and fixed head/payload-envelope g1-g2. | Current fixed wall/landmark or a deployment asset. | `SEVERITY_OR_REQUIREMENT_UNRESOLVED`; historical simulated veto preserved. | Genesis 13/27 inventory; platform manifest; hazard analysis. | Simulated envelope is evidence, not a validated physical payload envelope. | Missing payload load case, component damage limit, impact calibration, physical-envelope parity, and object consequence. |
| Proximal-limb entrapment | `PCS-SR-STABILITY-001`, `PCS-SR-RECOVERY-001`. | Hips g3-g6 and thighs g7-g10. | Gap, wall, landmark, or moving structure; only fixed boxes represented. | `SEVERITY_OR_REQUIREMENT_UNRESOLVED`; hard if demonstrated entrapment/control loss. | URDF function; stuck/entrapment ontology annotation. | Joint obstruction or wedging is plausible but not established by link identity alone. | No direct entrapment definition, torque/saturation consequence, recovery envelope, or dynamic-object evidence. |
| Distal-limb/calf contact | `PCS-SR-CONTACT-001`, `PCS-SR-STABILITY-001`, `PCS-SR-RECOVERY-001`. | Calf/lower-leg/foot g11-g26. | Non-ground wall/landmark, or abnormal/unknown ground contact. | `SEVERITY_OR_REQUIREMENT_UNRESOLVED`; ordinary support alone may be `PERMITTED_SUPPORT_OR_SELF_CONTACT`. | Historical calf-link/Plane exclusion and 27-component inventory. | Primitive/contact phase can distinguish intended support from abnormal contact. | Current H1 exclusion is link-level; it does not supply prospective support phase, severity, or entanglement criteria. |
| Destabilisation or fall | `PCS-SR-STABILITY-001`: prevent unacceptable stability consequence. | Any region whose contact or command changes support/body state. | Any object and operating mode. | `HARD_PREACTION_SEPARATION` when consequence is demonstrated or prospectively required; otherwise unresolved. | Contact ontology stability/fall fields; progression hazard scenarios. | Stability consequence can be detected and bounded prospectively. | No approved attitude/support/fall limit, physical calibration, or minimum-risk response. |
| Repeated contact | `PCS-SR-RECOVERY-001`: terminate/replan and bound repetition. | Any contacting component. | Repeated external contact of unresolved or known severity. | Contact keeps its primary severity category; mandatory recovery overlay. | Ontology repetition and task annotations. | A repeat budget and effective response can be specified. | Existing one-event screen is developmental, not approved; no recovery-cycle or escalation limit. |
| Contact followed by stuck | `PCS-SR-RECOVERY-001`: detect, stop, separate, and restore capability. | Any contacting component. | Contact plus subsequent ineffective commanded motion. | Contact remains unresolved/hard as applicable; recovery overlay required. | Stuck detector and prospective ontology. | Stuck is a useful monitor signal. | Stuck does not distinguish entrapment, controller failure, deliberate hold, or physical recoverability. |
| Loss of task progress | `PCS-SR-TASK-001`: bound ineffective motion and preserve mission progress. | System/task level; not a link-only classification. | No-contact or contact-associated local-waypoint operation. | `MONITOR_AND_RECOVER` for the task event; physical contact is classified separately. | Progression safety-related non-performance text; local-waypoint specification. | Application can define minimum meaningful progress. | No stakeholder-approved progress, deadline, intervention, or mission-loss criterion. |
| Repeated abstention | `PCS-SR-TASK-001`, `PCS-SR-FALLBACK-001`. | Integrated system. | Repeated veto/stop with no physical contact. | `MONITOR_AND_RECOVER` | Progression document and factorised planner design. | Abstention is observable and a bounded fallback can be implemented. | No acceptable abstention duration/count, operator handoff, or mission consequence. |
| Failure to complete inspection/search | `PCS-SR-INSPECTION-001`. | Mission system; not a robot-region permission. | Missed asset/area, incomplete coverage, or unmet completion deadline. | `MONITOR_AND_RECOVER` plus task-level reporting/escalation. | Progression responsible-research and inspection discussion. | Intended deployment stakeholders can define inspection sufficiency. | `NOT REPRESENTED IN CURRENT TEST ENVIRONMENT`; no approved coverage/revisit/deadline or missed-hazard consequence. |

Every physical-contact row above remains independent of sensor observability.
The category would be unchanged under perfect full-body sensing.

| Hazard ID | Hazardous scenario and consequence | Current protected scope | Candidate requirement | Required verification/evidence | Status |
|---|---|---|---|---|---|
| `PCS-HZ-001` | Robot contact causes injury, robot/property damage, prohibited separation, or safety-critical-object damage. | Every historically disallowed robot/environment body contact; categorical person/fragile/safety-critical/prohibited consequences when fields exist. | `PCS-SR-CONTACT-001` | Approved object/link consequence limits, object fields, raw contact evidence, consequence fixtures, stakeholder acceptance. | `EVIDENCE_MISSING` |
| `PCS-HZ-002` | Contact or command causes instability, fall, entrapment, or loss of control. | Trunk/underside, hips, thighs, calves, and all directions remain protected. | `PCS-SR-STABILITY-001` | Stability/fall definition, body attitude and support evidence, physical or validated simulator limits, recovery response. | `EVIDENCE_MISSING` |
| `PCS-HZ-003` | Apparently low-severity or repeated contact leads to stuck behavior, ineffective movement, route loss, or mission failure. | Contact severity and recoverability remain separate; repeated ineffective contact is not harmless. | `PCS-SR-RECOVERY-001` | Stuck/entrapment criteria, contact repetition limit, recovery time/cycles, successful separation and restored-capability evidence. | `EVIDENCE_MISSING` |
| `PCS-HZ-004` | False free-space or insufficient-clearance belief causes collision/fall. | Full swept protected body, not merely a global or front clearance. | `PCS-SR-CLEARANCE-001` | Candidate swept envelope, calibrated stopping envelope, uncertainty, timing, and full-body attribution. | `PARTIALLY_SPECIFIED` |
| `PCS-HZ-005` | The robot enters a state with no contact-negative response or executable stop. | Every committed state/action boundary. | `PCS-SR-VIABILITY-001` | Implemented cycle time, observation/prediction/command replacement, qualified route response or stop, latency evidence. | `PARTIALLY_SPECIFIED` |
| `PCS-HZ-006` | False place merge or unresolved localisation routes the robot to an unsafe region. | System-level route and memory boundary; not reducible to contact sensing. | `PCS-SR-LOCALISATION-001` | False-merge/closure evidence, belief uncertainty, transition consistency, effective conservative response. | `EVIDENCE_MISSING` |
| `PCS-HZ-007` | Operation continues outside the calibrated operating domain. | Complete integrated system. | `PCS-SR-OOD-001` | Formal ODD, monitor validity, detection lead time, minimum-risk response, operator handoff. | `EVIDENCE_MISSING` |
| `PCS-HZ-008` | Repeated veto, stop, oscillation, or abstention prevents useful search/inspection. | Task-performance requirement remains separate from the contact veto. | `PCS-SR-TASK-001` | Application-defined coverage, progress, completion, deadline, false-abstention and intervention-cost acceptance. | `EVIDENCE_MISSING` |
| `PCS-HZ-009` | Incomplete inspection leaves a hazard undetected or an area uninspected. | Mission-level coverage and completion; not a sensor-contact scope reduction. | `PCS-SR-INSPECTION-001` | Asset/hazard priorities, coverage semantics, revisit rules, detection confidence, missed-area consequence, DNV/domain stakeholder review. | `EVIDENCE_MISSING` |
| `PCS-HZ-010` | Monitor acts too late, has excessive false alarms, or invokes an ineffective response. | Perception, prediction, memory, planning, controller, and fallback interaction. | `PCS-SR-MONITOR-001` | Hazard-linked lead time, calibration, false alarms, task cost, and response-effectiveness trials. | `EVIDENCE_MISSING` |
| `PCS-HZ-011` | Simulator or local controller behavior does not transfer to physical Go2 stopping or impact response. | Physical platform integration boundary. | `PCS-SR-TRANSFER-001` | Platform-equivalent mode, request/ack timing, stopping traces, telemetry, physical calibration, controlled trials. | `EVIDENCE_MISSING` |

## Per-region traceability

| Region/contact | Hazard links | Evidence needed before exclusion or reduced treatment | Disposition |
|---|---|---|---|
| Trunk/head-hosting body | `PCS-HZ-001`, `002`, `004` | Structural/electronics/payload limits, contact energy and damage model, stability and stopping consequence. | `PRESERVED_PROXY` |
| Underside | `PCS-HZ-001`, `002`, `003` | Grounding/entrapment consequence, structural limits, recoverability evidence. | `PRESERVED_PROXY` |
| Hips | `PCS-HZ-001`, `002`, `003` | Joint/actuator damage, control-authority, entrapment and recovery limits. | `PRESERVED_PROXY` |
| Thighs | `PCS-HZ-001`, `002`, `003`, `004` | Load transfer, swept contact, stability, entrapment, asset-damage and recovery evidence. | `PRESERVED_PROXY` |
| Calves with non-ground objects | `PCS-HZ-001`, `002`, `003`, `004` | Object-specific consequence, entanglement, repeated-contact and stability evidence. | `PRESERVED_PROXY` |
| Ordinary calf/foot-ground support | Normal locomotion support, not a disallowed environmental collision under the existing ontology. | Any change would require a new locomotion/contact ontology. | Existing exclusion preserved. |
| Front/side/rear | All physical-contact hazards | Direction-specific consequence and recovery limits; visibility is not a consequence rationale. | All preserved. |
| Person/fragile/safety-critical object | `PCS-HZ-001` | Application binding, stakeholder input, object detection/identity assurance, categorical response. | Prospectively categorical hard hazard; current binding unresolved. |
| Fixed walls/landmarks | `PCS-HZ-001` through `005` | Physical material/damage tolerance, robot damage limit, stopping and recovery evidence. | `PRESERVED_PROXY`; no wall-threshold transfer. |

The ordinary-support row preserves historical custody only. The current
calf-plane exemption is link-level and therefore broader than literal foot
support. A prospective `PERMITTED_SUPPORT_OR_SELF_CONTACT` assignment requires
collision-primitive identity and contact-phase semantics that distinguish
stance support from abnormal calf/ground contact. Until then, those prospective
semantics are `SEVERITY_OR_REQUIREMENT_UNRESOLVED`.

## Requirement allocation candidates

| Requirement ID | System requirement form | Candidate allocation | Source | Status |
|---|---|---|---|---|
| `PCS-SR-CONTACT-001` | Avoid material hazardous contact under the approved ODD. | System hazard logic; learned components may supply calibrated evidence but do not define severity. | `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:23-51` | `EVIDENCE_MISSING` |
| `PCS-SR-CLEARANCE-001` | Maintain clearance for committed motion plus validated stop and uncertainty. | Geometry/clearance estimator, planner admissibility, controller, stop guard. | `docs/lewm_deployment_valid_strong_braking_mode_v1_result.md:86-101` | `PARTIALLY_SPECIFIED` |
| `PCS-SR-STABILITY-001` | Avoid unacceptable fall, instability, entrapment, or control loss. | Planner, locomotion controller, stability monitor, fallback. | `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:23-51` | `EVIDENCE_MISSING` |
| `PCS-SR-VIABILITY-001` | Preserve a one-cycle response or qualified stop at every admitted state. | State eligibility, planner, 100 ms interface, controller. | `docs/lewm_control_commitment_horizon_and_viability_v1_result.md:147-164` | `PARTIALLY_SPECIFIED` |
| `PCS-SR-FALLBACK-001` | Enter a bounded executable minimum-risk state when assumptions fail. | OOD monitor, supervisor, platform stop, operator interface. | Progression PDF p.24 lines 34-41 and p.26 lines 12-18 | `EVIDENCE_MISSING` |
| `PCS-SR-RECOVERY-001` | Recover from contact/stuck without unacceptable repetition or task loss. | Recovery supervisor, planner, controller, operator escalation. | `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:23-51` | `EVIDENCE_MISSING` |
| `PCS-SR-MONITOR-001` | Detect hazard-linked unreliability in time for an effective response. | Perception/predictor/memory monitors plus system supervisor. | Progression PDF p.23 lines 18-41 | `EVIDENCE_MISSING` |
| `PCS-SR-TASK-001` | Meet minimum application-derived progress, coverage, and completion. | Local planner, memory/router, mission manager. | Progression PDF p.8 lines 6-13; `docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md:17-27` | `EVIDENCE_MISSING` |
| `PCS-SR-TRACE-001` | Trace hazards, requirements, component allocation, evidence, assumptions, and response. | SACE system argument; AMLAS for learned component allocations. | Progression PDF pp.23-26 | `PARTIALLY_SPECIFIED` |

## Perfect-sensing counterfactual trace

| Counterfactual evidence | What it could establish | What remains unresolved |
|---|---|---|
| Exact full-body contact and clearance at every physics step | Contact occurrence, attribution, timing, and geometric separation in the simulator. | Material consequence, acceptable severity, object fragility, human relevance, damage, fall tolerance, recoverability, stopping response, ODD, and mission consequence. |
| Exact future action outcomes | A representation-sufficiency upper bound and deterministic candidate consequence in simulation. | Pre-action predictability, deployed sensing, approved risk threshold, fallback execution, and physical transfer. |
| Exact route/task outcomes | Whether a tested action progressed under the chosen proxy task. | Stakeholder-approved inspection completeness, missed-hazard consequence, and acceptable intervention burden. |

The counterfactual cannot change any `PRESERVED_PROXY` disposition.

## Stage-B entry criteria

Stage B remains `STAGE_B_BLOCKED` until the eight blocker groups in
`docs/lewm_protected_contact_scope_requirements_review_v1.md` are resolved and
reviewed prospectively. Stage B must not be a retrospective search over sensor
results, body regions, or labels. It would require a new authorization and a
frozen application, ODD, hazard log, consequence model, recovery contract,
stopping contract, requirements allocation, acceptance criteria, and evidence
schema before any empirical materialisation.
