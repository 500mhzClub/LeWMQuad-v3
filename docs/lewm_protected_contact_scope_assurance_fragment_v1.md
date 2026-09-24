# Protected Contact Scope Assurance Fragment V1

Date: 2026-08-26

Status: incomplete SACE/AMLAS-oriented Stage-A fragment

Primary classification: `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`

## Assurance boundary

This fragment records the current argument structure and explicit defeaters. It
is not a safety case, an AMLAS conformance claim, a SACE completion claim, or an
assurance that the robot is safe. It follows the progression document's rule
that experimental evidence becomes safety evidence only when linked to a
defined hazard, operating context, allocated requirement, and effective system
response (`docs/SAINTS_Year_1_Progression_Document-3.pdf`, p.23 lines 18-49).

SACE is the outer system argument. AMLAS may structure evidence for learned
perception, prediction, risk/task heads, or learned place models, but contact
severity, graph/controller behavior, platform stopping, recovery, ODD handling,
and mission completion remain system responsibilities. A self-supervised JEPA
and hybrid learned/algorithmic online memory would require explicit AMLAS
tailoring and review (same PDF, pp.24-26).

## Top-level claim and present status

`PCS-G0`: Within a defined bounded indoor search/inspection ODD, the integrated
Go2 system avoids unacceptable physical consequences while satisfying the
minimum safety-related task performance required by the application.

Status: `UNSUPPORTED_TOP_LEVEL_CLAIM`.

The ODD, unacceptable-consequence limits, application completion requirements,
fallback, recovery, platform response, and stakeholder acceptance are not yet
defined sufficiently to support `PCS-G0`.

## Contexts

- `PCS-C1`: present research case study is RGB-based high-level Go2 navigation
  in bounded indoor environments; it is not deployment ready.
- `PCS-C2`: search/inspection is a prospective application family; the exact
  mission, assets, hazards, deadlines, people, and operating conditions are
  unresolved.
- `PCS-C3`: the current target is a simulated disallowed-contact/separation
  proxy, not material-impact, injury, property-damage, human, or physical-Go2
  safety.
- `PCS-C4`: safety-related non-performance includes failure to progress, cover,
  inspect, or complete the task, not only collision exposure.
- `PCS-C5`: the existing simulated scope excludes ordinary calf/foot-ground
  support and robot self-contact but includes abnormal non-calf ground and calf
  non-ground contact.
- `PCS-C6`: the 13 protected link groups aggregate 27 protected collision primitives. Region/link identity is attribution, not the primary requirements unit.
- `PCS-C7`: the historical calf-plane link exemption is broader than literal foot
  support. Prospective permitted-support semantics require primitive and
  contact-phase detail that is not yet defined.

## Context-first category allocation

The exact category vocabulary allocated under `PCS-G1` is:

- `HARD_PREACTION_SEPARATION` for person, fragile, safety-critical, prohibited,
  demonstrated-damage, fall, stability, entrapment, or control-loss
  contexts across all regions;
- `CONDITIONAL_RECOVERABLE_CONTACT` only where approved physical limits and a
  bounded recovery response establish recoverability without unacceptable task
  consequence;
- `MONITOR_AND_RECOVER` for no-contact stuck, oscillation, repeated veto or
  abstention, inadequate progress, or incomplete task;
- `PERMITTED_SUPPORT_OR_SELF_CONTACT` only for the established support/self
  exclusions;
- `SEVERITY_OR_REQUIREMENT_UNRESOLVED` wherever severity, permission, or task
  requirements are missing.

No current row can support `CONDITIONAL_RECOVERABLE_CONTACT`. Wall, landmark,
and abnormal-ground contexts remain `SEVERITY_OR_REQUIREMENT_UNRESOLVED` under
the existing conservative separation proxy. The broad calf-plane link
exemption is preserved only for historical label custody; prospective support
permission is unresolved without collision-primitive and contact-phase
semantics.

## Argument strategy

`PCS-S0`: Decompose `PCS-G0` into system claims for hazard identification,
physical consequence control, state viability and fallback, recoverability,
task non-performance, out-of-context operation, learned-component assurance,
and integrated verification. Preserve all unresolved contacts in the simulated
proxy until these claims have approved requirements and evidence.

### `PCS-G1`: Hazard and scope completeness

All credible contact, stability, entrapment, control-loss, recovery, ODD, and
inspection non-performance hazards are identified for the intended application,
and the protected-contact scope covers them.

Status: `UNSUPPORTED`.

Available evidence:

- prospective taxonomy and annotations in
  `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:23-51`;
- candidate scenarios in the progression PDF, p.24 lines 26-41;
- per-region preservation and hazard links in
  `docs/lewm_protected_contact_scope_traceability_matrix_v1.md`.

Defeaters:

- no approved application/ODD or complete stakeholder hazard analysis;
- no human, fragile-asset, safety-critical-infrastructure, movable-object, or
  damage-consequence binding;
- no evidence supporting removal of any body region or direction.
- no context/primitive/contact-phase semantics sufficient to distinguish
  permitted foot support from every calf-plane link contact.

### `PCS-G2`: Physical consequence requirements

Material hazardous contact is identified and prevented using approved
object/link-specific consequence limits, while recoverable-low-severity contact
is classified only from complete evidence.

Status: `UNSUPPORTED`.

Available evidence:

- required physical fields in
  `docs/lewm_contact_hazard_instrumentation_contract_v1.md:13-36`;
- blocked action list in
  `docs/lewm_material_contact_safety_model_next_experiment_spec_v1.md:6-17`.

Defeaters:

- no approved force/impulse or validated substitute;
- no relative-speed, duration, repetition, penetration, fall, damage, or
  entrapment acceptance limits;
- no platform calibration or region/object consequence model;
- development low-energy screens are not deployment limits.

### `PCS-G3`: Viability, stopping, and minimum-risk response

Every admitted action preserves a bounded contact-negative response or
qualified stop, and ODD or monitor failure leads to an executable minimum-risk
state within a validated time and distance.

Status: `UNSUPPORTED`.

Available requirement forms:

- `docs/lewm_control_commitment_horizon_and_viability_v1_result.md:47-64,147-164`;
- `docs/lewm_deployment_valid_strong_braking_mode_v1_result.md:86-120`.

Defeaters:

- no qualified 100 ms end-to-end replanning interface;
- no platform-equivalent stopping behavior or acknowledgement latency;
- no stopping envelope, uncertainty margin, or minimum-risk-state definition;
- no operator handoff or controller/plant variability evidence.

### `PCS-G4`: Recoverability

Contact, stuck, or entrapment is detected and resolved within an approved
recovery budget without unacceptable repeated contact, instability, damage, or
task loss.

Status: `UNSUPPORTED`.

Available evidence form:

- separate recoverability/task annotation in
  `docs/lewm_contact_hazard_analysis_and_ontology_v1.md:23-51,59-61`.

Defeaters:

- no accepted stuck, entrapment, successful-separation, or restored-capability
  definitions;
- no maximum recovery time/cycles, repeated-contact budget, recovery envelope,
  or escalation policy;
- no evidence linking low-severity contact to successful mission recovery.

### `PCS-G5`: Search/inspection non-performance

The integrated system meets application-derived minimum progress, coverage,
inspection completeness, and completion requirements without allowing physical
hazard to be offset by utility.

Status: `UNSUPPORTED`.

Available requirement forms:

- progression PDF, p.8 lines 6-13 and p.24 lines 34-41;
- `docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md:17-27`;
- `docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md:32-52`.

Defeaters:

- no application-derived coverage, deadline, completion, revisit, or
  missed-hazard consequence;
- no stakeholder acceptance of intervention burden or false abstention;
- local-waypoint development thresholds are not deployment requirements;
- beacon discovery, global routing, and inspection semantics are outside the
  local-waypoint evidence boundary.

### `PCS-G6`: Learned-component assurance

Each learned component satisfies measurable ML requirements allocated from the
system hazards and requirements within its stated operating context.

Status: `NOT_YET_ALLOCATED`.

Candidate AMLAS evidence classes:

- scope and ML safety requirements;
- scene-separated data and generation custody;
- learning/source/attempt evidence and component-specific baselines;
- verification under defined distribution shifts;
- deployment interface and out-of-context evidence.

Defeaters:

- system-level requirements and thresholds have not been hazard-derived;
- no authorized material-contact model or Stage-B panel exists;
- self-supervised JEPA and online hybrid memory require AMLAS tailoring;
- an ML predictor cannot compensate for an absent executable system response.

### `PCS-G7`: Integrated verification and assurance confidence

Evidence is relevant, unbiased, reproducible, independently reviewed where
appropriate, and linked to the claim through an explicit argument.

Status: `PARTIALLY_SUPPORTED_PROCESS`, `UNSUPPORTED_SAFETY_CLAIM`.

Existing process evidence includes immutable roles, preregistration, source
hashes, attempt receipts, retained negative results, and raw-evidence
requirements. These provide confidence in experimental custody, not proof of
acceptable risk. The remaining work listed in the progression PDF p.26 lines
12-18—formal ODD, validated hazardous scenarios, quantitative requirements,
stakeholder acceptance, minimum-risk response, and appropriate independence—
remains open.

## Perfect-sensing defeater analysis

`PCS-A1`: Suppose every protected body region has exact, zero-latency current
and future contact/clearance evidence.

Under `PCS-A1`, geometric detection evidence for parts of `PCS-G2` and `PCS-G3`
would improve, but `PCS-G0` would remain unsupported. Perfect sensing does not
define severity, damage, acceptable contact, ODD, mission consequence,
recoverability, stop execution, operator response, or stakeholder tolerance.
Consequently:

- no region may be excluded based on sensing difficulty;
- no region may be declared acceptable based only on sensing success;
- sensor evaluation remains downstream of an approved requirements allocation.

## Assurance classification

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

The distributed-sensing and external-range classifications are downstream
engineering dispositions. They do not support the protected-scope argument and
do not authorize hardware selection. `STAGE_B_NOT_AUTHORIZED` is the current
assurance gate, not a secondary classification.

## Authorized next step

Requirements acquisition, stakeholder engagement, ODD definition, hazard-log
completion, consequence-limit approval, stopping/recovery specification, and
SACE/AMLAS allocation may proceed under separate authority. Stage B empirical
scope testing is not authorized. No label change, sensor selection, training,
new panel, predictor, memory, navigation, routing, or beacon work is authorized
by this fragment.
