# Contact hazard analysis and prospective ontology V1

Date: 2026-08-21
Status: prospective development requirement
Source commit: `c4790b85ebd0f58846c1cb73772be7d030896d95`

## Claim boundary

The completed binary-contact experiments and their classifications remain unchanged:

- `CONTACT_REQUIREMENT_ONTOLOGY_REQUIRES_REVISION`;
- `FUSION_SCORE_FRONTIER_NO_GO`;
- `LIDAR_COMPLEMENTARY_TENDENCY`.

This ontology is prospective. It does not retrospectively declare any historical contact harmless or materially hazardous, and it was not selected using classifier scores. Contact-event detection is not material-hazard prediction. Operational hard safety, recoverability, and task-performance safety remain separate requirements. Avoiding collision does not establish task completion; a stationary or reject-all robot does not satisfy search or inspection safety.

## Requirement inputs

The project inputs inspected were the frozen v1.2 physical contact definition in `lewm/oracle/go2_branch_oracle_v1_2.py`; `docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md`; `docs/lewm_planner_design_decision_memo_2026-08-19.md`; `docs/lewm_rollout_safety_and_trajectory_cleanup_2026-06-13.md`; `docs/lewm_geometry_fusion_contact_error_attribution_v1_result_2026-08-21.md`; and `docs/lewm_go2_scientific_execution_authority_threat_model_2026-07-13.md`. Together they require actual robot/environment contact evidence, separate safety and progress outputs, hard rejection that cannot be offset by utility, explicit abstention, and raw evidence rather than caller-authored aggregate claims.

The official Unitree Go2 specification describes an approximately 15 kg aluminium-alloy/high-strength-plastic robot and approximately 45 N·m peak joint torque, but supplies no allowable body-impact force, impulse, contact velocity, penetration, human-injury, or infrastructure-damage limit. Those product figures therefore do not justify a contact-severity threshold. See the [official Unitree Go2 specification](https://www.unitree.com/mobile/go2/).

## Hazard decomposition

The following are different requirement classes:

1. `MATERIAL_HAZARDOUS_CONTACT`: contact with a credible direct injury, robot/property damage, stability/fall, severe entrapment, control-loss, or unacceptable-separation consequence. This may support a hard veto.
2. `RECOVERABLE_LOW_SEVERITY_CONTACT`: complete physical and object-consequence evidence establishes low energy, short duration, no damage relevance, no destabilisation, no material progress loss, and recoverability.
3. `SEVERITY_UNRESOLVED`: the evidence establishes neither class.
4. Recoverability/task-performance failure: stuck, repeated ineffective contact, route-progress loss, excessive recovery, or mission incompletion. This remains separately annotated and may require monitoring, recovery, replanning, or a soft penalty.
5. `NO_DISALLOWED_CONTACT`: no robot/environment contact under the unchanged historical definition.

Ordinary calf/foot–ground support contact and robot self-contact remain excluded. Abnormal non-calf ground contact and calf contact with non-ground geometry remain disallowed.

## Deterministic classification

An event is materially hazardous when any prospectively supplied object/consequence field identifies a person proxy, fragile or safety-critical object, prohibited contact, recorded damage, loss of stability, or fall. It is recoverable-low-severity only when all object consequence fields are known and all of the following conservative development screens are satisfied:

| Screen | Value | Unit | Role and limitation |
|---|---:|---|---|
| Duration | ≤ 0.040 | s | Brief-event screen; not a human or damage limit. |
| Relative normal speed | ≤ 0.20 | m/s | Conservative low-energy screen; not a certified impact limit. |
| Integrated normal impulse | ≤ 0.50 | N·s | Development screen from calibrated event evidence; not a manufacturer limit. |
| Penetration | ≤ 0.002 | m | Simulator separation-quality screen; not a physical crush limit. |
| Repeated-contact count | ≤ 1 | events | Excludes repeated ineffective contact from “recoverable low severity.” |

It must additionally have no stuck or route-progress consequence. These five values are conservative engineering assumptions used only to test whether a completely evidenced event is plainly low-energy. A statistical percentile was not used to define safety. They do not apply to people, fragile objects, or safety-critical infrastructure.

There is deliberately no numeric `MATERIAL_HAZARDOUS_CONTACT` force/impulse threshold in V1. The project and manufacturer inputs do not supply a defensible robot-, object-, or human-specific limit. Events above the low-energy screen but without a categorical consequence therefore remain unresolved.

Separate annotations are `STABILITY_HAZARD`, `DAMAGE_RELEVANT`, `HUMAN_OR_FRAGILE_OBJECT_RELEVANT`, `STUCK_OR_ENTRAPMENT_CONSEQUENCE`, and `TASK_PROGRESS_CONSEQUENCE`.

## Current environment scope

The four frozen maze families contain a fixed ground plane, fixed rigid wall boxes, and fixed landmark boxes. Across the inspected manifests there are no movable obstacles and no asset fields identifying people/person proxies, fragility, safety-critical infrastructure, mechanical material, mass, permitted/prohibited contact, or damage consequence. Visual `material_id` values are rendering categories, not mechanical properties.

Accordingly, a wall-contact threshold from this simulator cannot be transferred to people, fragile assets, or real inspection infrastructure. This missing consequence model is decisive for the V1 readiness result.

## Branch labels

A branch prospectively records: any material hazard, any recoverable contact, any unresolved contact, maximum event severity, event count, cumulative impulse, contact followed by stuck, and no disallowed contact. Recoverability is never collapsed into the hard-contact target.
