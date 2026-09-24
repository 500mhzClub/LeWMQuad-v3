# Frozen pose transfer: longer physical observations, explicit coverage failures

The four predeclared nominal estimators all tracked all 586 frames of the fresh
fit and validation trials. Joint RGB-D obtains full-body initial-floor coverage
from deployment-shaped sensor inputs after sufficient actual travel. No model
was fitted or selected using these trials, and all predictions preceded native
scoring. These are two tapes in ONE geometry, not independent-maze evidence.

| Result | Fit joint | Fit gyro | Validation joint | Validation gyro |
| --- | ---: | ---: | ---: | ---: |
| Admitted frames | 586 | 586 | 586 | 586 |
| Maximum position error | 23.162 mm | 16.654 mm | 20.861 mm | 17.381 mm |
| Maximum orientation error | 4.242 mrad | .618 mrad | 3.646 mrad | .731 mrad |
| Keyframes including initial anchor | 75 | 75 | 59 | 56 |
| Admitted coverage queries | 586 | 249 | 586 | 215 |
| Coverage-contract rejections | 0 | 337 | 0 | 371 |
| First sensor-estimated 27/27 floor-coverage frame | 266 | None | 264 | None |
| Full-coverage frames | 320 | 0 | 322 | 0 |

Native-pose diagnostics first obtain full coverage at frame 262 in both trials.
Joint estimates lag that event by .4/.2 s. Against native-pose footprint queries,
joint produces zero false-covered shape queries and 41/20 missed-covered queries
among 15,822 shape queries per trial. This is conditional observability evidence,
not a bound on future error or proof of safe body/floor separation. Missing gyro
queries are UNKNOWN, not correctly uncovered or obstacle observations.

## Preserved failure and numerical diagnosis

The original frozen-transfer coordinator terminated when a diagnostic coverage
query rejected a rotation, before it persisted predictions or evaluation. Its
launch and failure remain immutable. A separately named
[explicit-coverage-status successor](go2_longer_motion_frozen_pose_coverage_status_v1_2026-09-06.md)
preserved the exact models, scorer, input tapes, initial surface and coverage
predicate. It records coverage rejection separately from pose survival. This is
failure accounting, not another independent trial or a silently repaired result.

The estimator allows rotation numerical defects up to 1e-8; the floor consumer
requires 1e-12. In the completed successor, gyro coverage first rejects at frames
249/215. Direct global gyro integration has maximum matrix defects only
2.63e-14/5.73e-14, whereas its keyframe-composed pose reaches 1.94e-12/3.86e-12.
The audit reconstructs the saved composition R_anchor (G_anchor^T G_current).
Finite-precision G_anchor^T is not its exact inverse; repeated anchor composition
accumulates numerical error. A single transpose-versus-inverse diagnostic differs
by up to 2.50e-14/5.67e-14; the composed pose differs from direct gyro by up to
9.52e-13/1.90e-12. No inverse replacement, pose projection or tolerance relaxation
was applied to runtime outputs. Joint rotations remain within 1.16e-14.

The gyro baseline is actually more accurate in these nominal trials. Its missing
full-coverage result is an interface/numerical limitation, not evidence that joint
RGB-D has better perceptual or navigation performance. Preserve that distinction
in any future comparison.

## Audit, tests and identities

The reference audit checked all 2,344 accepted states, 339,252 raw-depth lifted
point pairs and 261 promoted parent links. Independent quaternion-eigen joint
fits agree with the production SVD fits within 3.11e-15. Direct gyro is exactly
reconstructed; saved pose and translation compositions agree. All 1,636 admitted
coverage queries agree with direct cell enumeration rather than the prefix-sum
counter, and all 708 rejections reproduce their original predicate and reason.
Surface construction is shared; correspondence identity and physical error
hypotheses are not independently established by this audit.

The 2,156-test regression passed; 21 additional focused tests passed across the
acquisition audit, frozen scorer and explicit-coverage helper. The latter includes
the real numerical tolerance mismatch, preservation of negative/unknown coverage,
unchanged model and scorer identity and propagation of unexpected exceptions.

Root: `.generated/go2_longer_motion_frozen_pose_coverage_status_v1_attempt_001`.
Launch: 532 source bindings, 11,295 inputs plus native/OpenCV.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | b022ac51531226205e09b9dd42ef631d761aec2149955589c4960051c77ba7e6 |
| result.json | 23b3861c028e8207c748dc55f520f84a8cf941b67ac94d62fd05d735ba289c34 |
| predictions.json | 2cab75bad84d601ec5fc7271045308696dc48bb257cac196af084dfdf83ccadb |
| evaluation.json | f70dc9b696d38bb5ffa97c25a6ad60efd67c75c1f75c0f273536cea15b8c69e8 |
| pose_chain_coverage_audit_launch.json | 2c3b87e93f0077e2db8ec7d120d83dee711964be4404c07f48d3eabbeccd0a7f |
| pose_chain_coverage_audit.json | 82bc0e7278a704359151fa04a6fe4b056dffd86e2065f75012284789d96466e9 |

## Next steps toward actual navigation

1. Resolve the gyro/reference rotation contract without hiding a rejection:
   use a reviewed rotation representation/composition rule or explicitly bound any
   projection correction. Preserve global history and quantify numerical changes
   separately from sensor uncertainty. This is an engineering comparison on known
   tapes, not fresh validation and not a reason to favor the joint model.
2. Use fitting-only data to diagnose task-relevant body/surface gap and prospective
   motion errors, rather than assigning the 23-mm global maximum to every local
   query. Freeze an action/error fitting procedure before fresh reserved validation.
   Include measured yaw drift, turning sway and stopping displacement; keep camera,
   joint, timing and surface hypotheses explicit. Current validation is exposed.
3. Resolve the startup observability problem: full initial-floor coverage appears
   only after roughly 26 s of externally supervised forward motion. A deployable
   controller cannot borrow that open-loop prelude. Evaluate deployment-valid
   downward/wider RGB-D coverage or a validated contact/proprioception-grounded
   local support model under a NEW sensor/initialization protocol. Do not silently
   extend the evaluator-only setup region or assume unseen floor.
4. Integrate the resulting floor/non-floor and action evidence in short closed-loop
   forward/turn/brake tasks with online memory, then complete true maze discovery,
   wrong-branch recovery and home return. Establish matched JEPA/supervised/geometry,
   genuine multi-step rollout and memory factors across independent layouts/seeds,
   timed operation and bounded hardware evidence.

The learned low-level gait is unchanged. The estimator and coverage stack here
are geometric. No JEPA navigation contribution, deployable safety envelope,
successful maze return or ultimate-goal completion has been established.
