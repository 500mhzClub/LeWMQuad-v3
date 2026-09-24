# Contact hazard instrumentation contract V1

Date: 2026-08-21
Physics cadence: 0.002 s
Policy horizon: 15 ticks / 750 physics steps per branch

## Identity and replay

The development panel is exactly the original 48 `purpose-*` states and 576 registered candidates. Every replay restores the registered pre-action state, executes the exact post-slew candidate, and verifies action trace, H3 endpoint pose, 15-tick binary contact trace, frozen stuck trace, and aggregate outcome. A mismatch is preserved and excluded; no replacement identity is allowed.

The two historical selected-contact branches are separate `POST_HOC_DESCRIPTIVE_CASE_STUDY` receipts and are excluded from ontology development.

## Raw physics-step record

For each disallowed robot/environment contact point, the recorder stores:

- state, scene, family, candidate, tick, policy-step, physics-step, robot link, environment link/entity/object, and object class;
- world and robot-body contact point, solver normal, penetration, simultaneous-point count, and robot side/octant;
- solver force vector and magnitude, normal-force component, and 2 ms normal-impulse increment;
- tangential force/impulse, derived exactly from the stored force vector and normal during reduction;
- immediately-preceding link point velocity decomposed into normal and tangential relative speed; all current maze objects are fixed;
- base linear/angular velocity response, projected-gravity change, joint velocity/acceleration response, actuator control torque, support-contact force, fall/stability flags, frozen stuck consequence, candidate command, and route-progress consequence.

Genesis exposes contact point, normal, penetration, and solver force. `integrated_*_impulse_n_s` is transparently force integrated at 0.002 s; it is not represented as a native or hardware-calibrated impulse sensor. World/body pose and dynamics are label-construction evidence, not authorised deployment inputs.

Unavailable or unresolved fields are explicit: controller saturation limit, physical recovery time beyond H3, nominal-pose deviation at every physics step, object mass/mechanical material/fragility/safety criticality, observed damage, and hardware calibration of solver force.

## Exclusion and event grouping

The unchanged historical 1e-3 N numerical floor is used only to reproduce the binary contact definition. It is not a severity limit. Ordinary calf–ground support and robot self-contact are excluded.

A contact event is a contiguous or near-contiguous sequence with the same branch, robot link, and environment object, with no more than two empty physics steps (4 ms) between contacts. The rule was frozen before development reduction. Event reduction reports start/end, duration, peak force, integrated normal/tangential impulse, peak relative normal/tangential speed, maximum penetration, body side/link, object class, stability, stuck, and progress consequence. Raw points remain available.

## Object consequence fields

The prospective schema includes fixed/movable status, mass, mechanical material, fragility, safety-critical status, person proxy, permitted/prohibited contact, and damage observation. The current assets provide fixed status, object identity/class, and a visual material only. Missing fields remain null and force `SEVERITY_UNRESOLVED` unless an independently sufficient categorical hazard consequence is present.

## Fixtures

The deterministic suite covers foot–ground exclusion, self-contact exclusion, low-energy brush, sustained/high-speed/repeated contact, contact followed by stuck, instability, hypothetical fragile/person contact, missing evidence, body-region assignment, 4 ms event grouping, and byte-identical JSON regeneration. Fixture content digest: `7087e9e97db21880ae860e712b004aad58ec925898b1bafcce48561911c63063` (file SHA-256: `67b4ac762be27a57255d6a3c452273d4752eee22ccd6cb3dc4b9f6c12be17838`).

## Evidence bindings

- Raw point index: `docs/lewm_contact_hazard_raw_contact_event_index_v1.json`
- Event ledger: `docs/lewm_contact_hazard_event_ledger_v1.json`
- Branch ontology ledger: `docs/lewm_contact_hazard_branch_ontology_ledger_v1.json`
- Raw cache content index digest: `b0897c9fbc1e739495a0b1184ede11639d6932e6a5fa37761efc246f8b45d610`
- Raw compressed evidence: 9,732,524 bytes

The committed index binds each immutable compressed raw-point shard by SHA-256. The event and branch ledgers are sufficient for deterministic aggregate reduction without simulation.
